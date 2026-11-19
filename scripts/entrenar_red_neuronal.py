"""Entrena dos arquitecturas de red neuronal (perceptrón multicapa) y selecciona por validación.

La red se entrena con retro-propagación del error: en cada lote se calcula la
pérdida (entropía cruzada), se propaga el gradiente desde la capa de salida
hacia las ocultas y el optimizador Adam actualiza los pesos.
"""
import argparse
import gc
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
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from scripts.evaluacion_modelos import comprobar_datos, evaluar, guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO

# Dos arquitecturas fijadas antes de evaluar: una capa oculta y dos capas ocultas.
ARQUITECTURAS = [('una_capa', (64,)), ('dos_capas', (128, 64))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'outputs/modelos/red_neuronal')
    parser.add_argument('--epocas', type=int, default=100, help='Máximo de pasadas por training.')
    parser.add_argument('--trabajos', type=int, default=4)
    args = parser.parse_args()
    if args.epocas < 1 or args.trabajos < 1:
        parser.error('--epocas y --trabajos deben ser positivos.')
    if args.salida.exists():
        parser.error('La salida ya existe; elegir otra carpeta para conservar resultados.')
    names = pd.read_csv(args.datos / 'atributos_finales.csv')['atributo']
    X = load_npz(args.datos / 'X_train.npz').astype('float32').toarray()
    y = pd.read_csv(args.datos / 'y_train.csv')['Severity'].to_numpy()
    V = load_npz(args.datos / 'X_validation.npz').astype('float32').toarray()
    v = pd.read_csv(args.datos / 'y_validation.csv')['Severity'].to_numpy()
    comprobar_datos(X, y, names)
    comprobar_datos(V, v, names)
    args.salida.mkdir(parents=True)
    rows = []
    for variant, layers in ARQUITECTURAS:
        folder = args.salida / variant
        folder.mkdir()
        # early_stopping reserva el 10 % de training (estratificado) para decidir
        # cuándo cortar; el conjunto de validación del proyecto no se usa para eso.
        model = MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='adam',
                              alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                              max_iter=args.epocas, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=5, tol=1e-4,
                              random_state=42)
        print(f'Entrenando red_neuronal/{variant} {layers}: {len(y):,} filas', flush=True)
        start = time.perf_counter()
        with threadpool_limits(limits=args.trabajos), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X, y)
        elapsed = time.perf_counter() - start
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        metrics = evaluar(model, V, v, folder, 'validation')
        train_metrics = evaluar(model, X, y, folder, 'train')
        row = dict(variante=variant, capas=str(layers), epocas=int(model.n_iter_),
                   corte_temprano=converged, segundos_ajuste=elapsed, **metrics,
                   f1_macro_train=train_metrics['f1_macro'])
        joblib.dump(model, folder / 'modelo.joblib', compress=3)
        guardar_json(folder / 'ajuste.json', dict(parametros=model.get_params(), resultado=row,
            advertencias=[str(w.message) for w in caught],
            pesos_por_capa=[list(w.shape) for w in model.coefs_],
            representacion_ajuste='densa_float32'))
        guardar_json(folder / 'curva_aprendizaje.json', dict(
            perdida_train=[float(x) for x in model.loss_curve_],
            accuracy_validacion_interna=[float(x) for x in model.validation_scores_]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.salida / 'comparacion_validation.csv', index=False)
        print(f'Resultado red_neuronal: {row}', flush=True)
        del model
        gc.collect()
    best = max(rows, key=lambda row: row['f1_macro'])
    guardar_json(args.salida / 'experimento.json', dict(datos=str(args.datos.resolve()),
        filas_train=len(y), filas_validation=len(v), atributos=len(names), clases=[1, 2, 3, 4],
        seleccion=best, criterio='Mayor F1 macro de validación', test_utilizado=False,
        sklearn=sklearn.__version__, python=platform.python_version(),
        configuracion_por_variante='Consultar ajuste.json en cada subcarpeta'))
    _graficar_curvas(args.salida, [variant for variant, _ in ARQUITECTURAS])
    lines = ['# Red neuronal (perceptrón multicapa): validación', '',
        '| Variante | Capas ocultas | Épocas | Accuracy | F1 macro | Balanced accuracy | F1 macro train |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['variante']} | {r['capas']} | {r['epocas']} | {r['accuracy']:.4f} | "
              f"{r['f1_macro']:.4f} | {r['balanced_accuracy']:.4f} | {r['f1_macro_train']:.4f} |"
              for r in rows]
    lines += ['', f"Seleccionada: {best['variante']}. Test no utilizado.", '',
              '![Curvas de aprendizaje](curvas_aprendizaje.png)']
    (args.salida / 'resumen.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _graficar_curvas(salida, variantes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for variant in variantes:
        curve = json.loads((salida / variant / 'curva_aprendizaje.json').read_text(encoding='utf-8'))
        epochs = np.arange(1, len(curve['perdida_train']) + 1)
        left.plot(epochs, curve['perdida_train'], marker='o', label=variant)
        right.plot(epochs, curve['accuracy_validacion_interna'], marker='o', label=variant)
    left.set(title='Pérdida en training', xlabel='Época', ylabel='Entropía cruzada')
    right.set(title='Accuracy en la reserva interna', xlabel='Época', ylabel='Accuracy')
    left.legend()
    right.legend()
    fig.tight_layout()
    fig.savefig(salida / 'curvas_aprendizaje.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
