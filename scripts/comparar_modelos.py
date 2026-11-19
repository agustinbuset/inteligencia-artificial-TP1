"""Congela la selección por validación y evalúa los cuatro modelos en prueba."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import classification_report, confusion_matrix

from scripts.evaluacion_modelos import comprobar_datos, evaluar, guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO

METODOS = ['regresion_logistica', 'arbol_decision', 'random_forest', 'red_neuronal']
ETIQUETAS = {'regresion_logistica': 'Regresión logística', 'arbol_decision': 'Árbol de decisión',
             'random_forest': 'Random Forest', 'red_neuronal': 'Red neuronal'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--modelos', type=Path, default=PROYECTO / 'outputs/modelos')
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'outputs/comparacion_modelos')
    args = parser.parse_args()
    if args.salida.exists():
        parser.error('La salida existe; elegir otra carpeta para conservar la evaluación previa.')
    chosen = {}
    for method in METODOS:
        folder = args.modelos / method
        meta = json.loads((folder / 'experimento.json').read_text(encoding='utf-8'))
        if Path(meta['datos']).resolve() != args.datos.resolve() or meta['test_utilizado']:
            raise ValueError(f'{method}: procedencia o protocolo incompatible.')
        rows = pd.read_csv(folder / 'comparacion_validation.csv')
        if 'convergio' in rows:
            rows = rows[rows['convergio'].eq(True)]
        winner = rows.loc[rows['f1_macro'].idxmax()]
        if winner['variante'] != meta['seleccion']['variante']:
            raise ValueError(f'{method}: selección inconsistente.')
        chosen[method] = dict(variante=winner['variante'],
            f1_macro_validation=float(winner['f1_macro']),
            accuracy_validation=float(winner['accuracy']),
            segundos_ajuste=float(winner['segundos_ajuste']),
            modelo=str((folder / winner['variante'] / 'modelo.joblib').resolve()))
    # La elección del ganador se fija antes de leer los archivos de prueba.
    winner = max(chosen, key=lambda method: chosen[method]['f1_macro_validation'])
    args.salida.mkdir(parents=True)
    guardar_json(args.salida / 'seleccion_previa_test.json', dict(
        modelos=chosen, ganador_validation=winner, criterio='Mayor F1 macro de validación'))
    names = pd.read_csv(args.datos / 'atributos_finales.csv')['atributo']
    V = load_npz(args.datos / 'X_validation.npz').tocsr()
    v = pd.read_csv(args.datos / 'y_validation.csv')['Severity'].to_numpy()
    X = load_npz(args.datos / 'X_test.npz').tocsr()
    y = pd.read_csv(args.datos / 'y_test.csv')['Severity'].to_numpy()
    comprobar_datos(V, v, names)
    comprobar_datos(X, y, names)
    result = []
    for method, selection in chosen.items():
        model = joblib.load(selection['modelo'])
        folder = args.salida / method
        folder.mkdir()
        pred = model.predict(V)
        report = classification_report(v, pred, labels=[1,2,3,4], output_dict=True, zero_division=0)
        previous = args.modelos / method / selection['variante']
        saved = json.loads((previous / 'metricas_validation.json').read_text(encoding='utf-8'))
        if report != saved:
            raise ValueError(f'{method}: métricas no reproducibles al recargar el modelo.')
        matrix = pd.read_csv(previous / 'matriz_confusion_validation.csv', index_col=0).to_numpy()
        if not np.array_equal(matrix, confusion_matrix(v, pred, labels=[1,2,3,4])):
            raise ValueError(f'{method}: matriz de confusión de validación inconsistente.')
        metrics = evaluar(model, X, y, folder, 'test')
        result.append(dict(modelo=method, variante=selection['variante'],
            f1_macro_validation=selection['f1_macro_validation'],
            segundos_ajuste=selection['segundos_ajuste'], **metrics))
        print(f'{method}: {metrics}', flush=True)
        del model
    pd.DataFrame(result).to_csv(args.salida / 'comparacion_test.csv', index=False)
    train_y = pd.read_csv(args.datos / 'y_train.csv')['Severity']
    majority = int(train_y.mode().iloc[0])
    guardar_json(args.salida / 'baseline_mayoritaria_test.json', classification_report(
        y, np.full(len(y), majority), labels=[1,2,3,4], output_dict=True, zero_division=0))
    guardar_json(args.salida / 'verificacion.json', dict(modelos_recargados=True,
        metricas_validation_recalculadas=True, matrices_validation_verificadas=True,
        ganador_elegido_antes_de_test=winner, filas_test=len(y), atributos=len(names)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(result))
    ax.bar(positions - .18, [r['f1_macro_validation'] for r in result], .36, label='Validación')
    ax.bar(positions + .18, [r['f1_macro'] for r in result], .36, label='Prueba')
    ax.set_xticks(positions, [ETIQUETAS[r['modelo']] for r in result])
    ax.set_ylabel('F1 macro')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.salida / 'comparacion_f1_macro.png', dpi=150)
    plt.close(fig)
    lines = ['# Comparación de los cuatro clasificadores', '',
        f'Ganador fijado por F1 macro de validación: **{winner}**.', '',
        '| Modelo | Variante | F1 macro validación | F1 macro prueba | Accuracy prueba | Balanced accuracy prueba |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['modelo']} | {r['variante']} | {r['f1_macro_validation']:.4f} | "
        f"{r['f1_macro']:.4f} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} |" for r in result]
    lines += ['', '![F1 macro por modelo](comparacion_f1_macro.png)', '',
        'No se ajustaron parámetros después de consultar prueba. El preprocesamiento ya tenía',
        'una evaluación histórica en test; por eso no es un conjunto nunca observado en el proyecto.',
        'La partición aleatoria mide una mezcla temporal y geográfica similar, no regiones o años nuevos.']
    (args.salida / 'resumen.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
