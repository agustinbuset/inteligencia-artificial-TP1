"""Entrena árbol de decisión y/o Random Forest; selecciona variantes por validación."""
import argparse
import gc
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits

from scripts.evaluacion_modelos import comprobar_datos, evaluar, guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metodo', choices=['arbol_decision', 'random_forest', 'ambos'], default='ambos')
    parser.add_argument('--datos', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'outputs/modelos')
    parser.add_argument('--arboles', type=int, default=100)
    parser.add_argument('--trabajos', type=int, default=2)
    parser.add_argument('--reanudar', action='store_true',
                        help='Recupera modelos guardados, comprobando parámetros y métricas.')
    args = parser.parse_args()
    if args.arboles < 1 or args.trabajos < 1:
        parser.error('--arboles y --trabajos deben ser positivos.')
    methods = ['arbol_decision', 'random_forest'] if args.metodo == 'ambos' else [args.metodo]
    if not args.reanudar and any((args.salida / method).exists() for method in methods):
        parser.error('Una salida ya existe; elegir otra raíz con --salida para preservarla.')
    names = pd.read_csv(args.datos / 'atributos_finales.csv')['atributo']
    # Con 172 columnas, float32 denso ocupa aproximadamente 1,43 GB en train.
    # Fue más rápido que CSC en una medición sobre 100.000 filas del dataset.
    X = load_npz(args.datos / 'X_train.npz').astype('float32').toarray()
    y = pd.read_csv(args.datos / 'y_train.csv')['Severity'].to_numpy()
    V = load_npz(args.datos / 'X_validation.npz').astype('float32').toarray()
    v = pd.read_csv(args.datos / 'y_validation.csv')['Severity'].to_numpy()
    comprobar_datos(X, y, names)
    comprobar_datos(V, v, names)
    for method in methods:
        output = args.salida / method
        output.mkdir(parents=True, exist_ok=args.reanudar)
        rows = []
        for variant, weight in [('sin_pesos', None), ('balanceada', 'balanced')]:
            folder = output / variant
            folder.mkdir(exist_ok=args.reanudar)
            common = dict(max_depth=20, min_samples_leaf=5, class_weight=weight, random_state=42)
            if method == 'arbol_decision':
                model = DecisionTreeClassifier(**common)
            else:
                model = RandomForestClassifier(**common, n_estimators=args.arboles,
                    max_features='sqrt', bootstrap=True, max_samples=0.25,
                    n_jobs=args.trabajos, verbose=1)
            recovered = args.reanudar and (folder / 'modelo.joblib').exists()
            previous = None
            if recovered:
                previous = json.loads((folder / 'ajuste.json').read_text(encoding='utf-8'))
                expected = model.get_params()
                actual = previous['parametros']
                if any(actual[key] != value for key, value in expected.items() if key != 'verbose'):
                    raise ValueError(f'{folder}: parámetros diferentes; elegir otra salida.')
                model = joblib.load(folder / 'modelo.joblib')
                for matrix, labels, split in [(V, v, 'validation'), (X, y, 'train')]:
                    report = classification_report(labels, model.predict(matrix), labels=[1,2,3,4],
                                                   output_dict=True, zero_division=0)
                    saved = json.loads((folder / f'metricas_{split}.json').read_text(encoding='utf-8'))
                    if report != saved:
                        raise ValueError(f'{folder}: los datos o métricas de {split} no coinciden.')
                elapsed = previous['resultado']['segundos_ajuste']
                notices = previous['advertencias']
                print(f'Recuperando {method}/{variant}; verificando métricas', flush=True)
            else:
                if any(folder.iterdir()):
                    raise ValueError(f'{folder}: salida parcial sin modelo recuperable.')
                print(f'Entrenando {method}/{variant}: {len(y):,} filas', flush=True)
                start = time.perf_counter()
                with threadpool_limits(limits=args.trabajos), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(X, y)
                elapsed = time.perf_counter() - start
                notices = [str(w.message) for w in caught]
            # Desactivar mensajes de joblib durante evaluaciones posteriores.
            if method == 'random_forest':
                model.verbose = 0
            metrics = evaluar(model, V, v, folder, 'validation')
            train_metrics = evaluar(model, X, y, folder, 'train')
            if recovered and any(metrics[key] != previous['resultado'][key] for key in metrics):
                raise ValueError(f'{folder}: las métricas del modelo recuperado no coinciden.')
            row = dict(variante=variant, segundos_ajuste=elapsed, **metrics,
                       f1_macro_train=train_metrics['f1_macro'])
            joblib.dump(model, folder / 'modelo.joblib', compress=3)
            guardar_json(folder / 'ajuste.json', dict(parametros=model.get_params(), resultado=row,
                advertencias=notices, representacion_ajuste=(previous.get('representacion_ajuste',
                'dispersa_csc_float32') if recovered else 'densa_float32')))
            pd.DataFrame({'atributo': names, 'importancia': model.feature_importances_}).sort_values(
                'importancia', ascending=False).to_csv(folder / 'importancias.csv', index=False)
            if method == 'arbol_decision':
                (folder / 'reglas_primeros_niveles.txt').write_text(export_text(
                    model, feature_names=list(names), max_depth=3), encoding='utf-8')
            else:
                guardar_json(folder / 'estructura.json', dict(arboles=len(model.estimators_),
                    profundidades=[int(tree.get_depth()) for tree in model.estimators_],
                    hojas=[int(tree.get_n_leaves()) for tree in model.estimators_]))
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / 'comparacion_validation.csv', index=False)
            print(f'Resultado {method}: {row}', flush=True)
            del model
            gc.collect()
        best = max(rows, key=lambda row: row['f1_macro'])
        guardar_json(output / 'experimento.json', dict(datos=str(args.datos.resolve()),
            filas_train=len(y), filas_validation=len(v), atributos=len(names), clases=[1,2,3,4],
            seleccion=best, criterio='Mayor F1 macro de validación', test_utilizado=False,
            sklearn=sklearn.__version__, python=platform.python_version(),
            representacion_evaluacion='densa_float32',
            configuracion_por_variante='Consultar ajuste.json en cada subcarpeta'))
        lines = [f'# {method}: validación', '',
            '| Variante | Accuracy | F1 macro | Balanced accuracy | F1 macro train |',
            '| --- | ---: | ---: | ---: | ---: |']
        lines += [f"| {r['variante']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
                  f"{r['balanced_accuracy']:.4f} | {r['f1_macro_train']:.4f} |" for r in rows]
        lines += ['', f"Seleccionada: {best['variante']}. Test no utilizado."]
        (output / 'resumen.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
