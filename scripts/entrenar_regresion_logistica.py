"""Entrena dos variantes de regresión logística y selecciona por F1 macro de validación."""
import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import load_npz
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from scripts.rutas import DATOS_PREPARADOS, PROYECTO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--salida', type=Path,
                        default=PROYECTO / 'outputs/modelos/regresion_logistica')
    args = parser.parse_args()
    if args.salida.exists():
        parser.error('La salida ya existe; elegir otra carpeta para conservar resultados.')
    X = load_npz(args.datos / 'X_train.npz').tocsr()
    y = pd.read_csv(args.datos / 'y_train.csv')['Severity'].to_numpy()
    V = load_npz(args.datos / 'X_validation.npz').tocsr()
    v = pd.read_csv(args.datos / 'y_validation.csv')['Severity'].to_numpy()
    names = pd.read_csv(args.datos / 'atributos_finales.csv')['atributo']
    for matrix, labels in [(X, y), (V, v)]:
        if matrix.shape != (len(labels), len(names)) or not np.isfinite(matrix.data).all():
            raise ValueError('Matrices y etiquetas incompatibles o valores no finitos.')
        if set(np.unique(labels)) != {1, 2, 3, 4}:
            raise ValueError('Se esperaban las cuatro clases de Severity.')
    args.salida.mkdir(parents=True)
    results = []
    best = None
    for name, weight in [('sin_pesos', None), ('balanceada', 'balanced')]:
        print(f'Entrenando {name}: {len(y):,} filas, {X.shape[1]} atributos', flush=True)
        model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                   class_weight=weight, max_iter=1000,
                                   tol=1e-4, random_state=42)
        start = time.perf_counter()
        with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X, y)
        elapsed = time.perf_counter() - start
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        pred = model.predict(V)
        report = classification_report(v, pred, labels=[1, 2, 3, 4],
                                       output_dict=True, zero_division=0)
        row = dict(variante=name, convergio=converged, iteraciones=int(model.n_iter_.max()),
                   segundos_ajuste=elapsed, accuracy=report['accuracy'],
                   f1_macro=report['macro avg']['f1-score'],
                   balanced_accuracy=report['macro avg']['recall'],
                   f1_weighted=report['weighted avg']['f1-score'])
        results.append(row)
        folder = args.salida / name
        folder.mkdir()
        (folder / 'ajuste.json').write_text(json.dumps(dict(
            parametros=model.get_params(), resultado=row,
            advertencias=[str(w.message) for w in caught]), indent=2), encoding='utf-8')
        joblib.dump(model, folder / 'modelo.joblib')
        (folder / 'metricas_validation.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        pd.DataFrame(confusion_matrix(v, pred, labels=[1, 2, 3, 4]),
                     index=pd.Index([1, 2, 3, 4], name='real'),
                     columns=['pred_1', 'pred_2', 'pred_3', 'pred_4']).to_csv(
                         folder / 'matriz_confusion_validation.csv')
        pd.DataFrame(model.coef_.T, index=names,
                     columns=[f'clase_{c}' for c in model.classes_]).to_csv(folder / 'coeficientes.csv')
        print(json.dumps(row), flush=True)
        if converged and (best is None or row['f1_macro'] > best['f1_macro']):
            best = row
        pd.DataFrame(results).to_csv(args.salida / 'comparacion_validation.csv', index=False)
    metadata = dict(datos=str(args.datos.resolve()), filas_train=len(y), filas_validation=len(v),
                    atributos=len(names), clases=[1, 2, 3, 4],
                    seleccion=best, criterio='Mayor F1 macro de validación entre ajustes convergentes',
                    test_utilizado=False, sklearn=sklearn.__version__, python=platform.python_version(),
                    configuracion_por_variante='Consultar ajuste.json en cada subcarpeta')
    (args.salida / 'experimento.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = ['# Regresión logística: validación', '',
             '| Variante | Convergió | Accuracy | F1 macro | Balanced accuracy |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in results:
        lines.append(f"| {row['variante']} | {row['convergio']} | {row['accuracy']:.4f} | "
                     f"{row['f1_macro']:.4f} | {row['balanced_accuracy']:.4f} |")
    lines.extend(['', f"Seleccionada: {best['variante'] if best else 'ninguna'}. "
                  'Criterio: F1 macro entre ajustes convergentes. Test no utilizado.'])
    (args.salida / 'resumen.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if best is None:
        raise RuntimeError('Ninguna variante convergió; revisar antes de seleccionar un modelo.')


if __name__ == '__main__':
    main()
