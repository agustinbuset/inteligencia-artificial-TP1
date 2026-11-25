"""Busca hiperparámetros y compara los cuatro modelos del TP."""
import argparse
import json
import time
import warnings
from copy import copy

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from scripts.rutas import DATOS_PREPARADOS, PROYECTO


GRILLAS = {
    'regresion_logistica': [
        {'C': c} for c in (0.01, 0.1, 1.0, 10.0)
    ],
    'arbol_decision': [
        {'max_depth': depth, 'min_samples_leaf': leaf}
        for depth in (14, 19, 24) for leaf in (5, 20)
    ],
    'random_forest': [
        {'max_depth': 20, 'max_samples': 0.25, 'max_features': 'sqrt'},
        {'max_depth': 28, 'max_samples': 0.5, 'max_features': 'sqrt'},
        {'max_depth': 32, 'max_samples': 0.75, 'max_features': 'sqrt'},
        {'max_depth': 28, 'max_samples': 0.5, 'max_features': 0.2},
    ],
    'red_neuronal': [
        {'hidden_layer_sizes': (128, 64), 'alpha': 1e-4},
        {'hidden_layer_sizes': (128, 64), 'alpha': 1e-3},
        {'hidden_layer_sizes': (192, 96), 'alpha': 1e-4},
        {'hidden_layer_sizes': (128, 64, 32), 'alpha': 1e-4},
    ],
}


def estimador(familia, params, trabajos, exploracion=False):
    if familia == 'regresion_logistica':
        return LogisticRegression(max_iter=1000, random_state=42, **params)
    if familia == 'arbol_decision':
        return DecisionTreeClassifier(random_state=42, **params)
    if familia == 'random_forest':
        return RandomForestClassifier(n_estimators=40 if exploracion else 100,
                                      min_samples_leaf=5, n_jobs=trabajos,
                                      random_state=42, **params)
    return MLPClassifier(max_iter=65 if exploracion else 100, early_stopping=True,
                         n_iter_no_change=5, batch_size=1024,
                         random_state=42, **params)


def metricas(modelo, X, y):
    informe = classification_report(y, modelo.predict(X), labels=[1, 2, 3, 4],
                                    output_dict=True, zero_division=0)
    return dict(f1_macro=informe['macro avg']['f1-score'], accuracy=informe['accuracy'],
                balanced_accuracy=informe['macro avg']['recall'],
                **{f'recall_{c}': informe[str(c)]['recall'] for c in range(1, 5)})


def evaluar_ajuste(modelo, X, y, V, v, carpeta, trabajos, ajustar=True):
    if (carpeta / 'validacion.json').exists():
        return json.loads((carpeta / 'validacion.json').read_text(encoding='utf-8'))
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f'Evaluando {carpeta.parent.name}/{carpeta.name}', flush=True)
    inicio = time.perf_counter()
    with threadpool_limits(limits=trabajos), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always', ConvergenceWarning)
        if ajustar:
            modelo.fit(X, y)
        resultado = metricas(modelo, V, v)
    resultado.update(parametros=modelo.get_params(), segundos=time.perf_counter() - inicio,
                     avisos=[str(aviso.message) for aviso in avisos])
    if isinstance(modelo, DecisionTreeClassifier):
        resultado.update(hojas=int(modelo.get_n_leaves()), profundidad=modelo.get_depth())
    joblib.dump(modelo, carpeta / 'modelo.joblib', compress=3)
    (carpeta / 'validacion.json').write_text(json.dumps(resultado, indent=2), encoding='utf-8')
    print(f'F1 macro de validación: {resultado["f1_macro"]:.5f}', flush=True)
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trabajos', type=int, default=4)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument('--familia', choices=GRILLAS, help='Optimiza una sola familia.')
    modo.add_argument('--comparar', action='store_true', help='Compara las familias ya optimizadas.')
    args = parser.parse_args()
    salida = PROYECTO / 'outputs' / 'optimizacion_modelos'
    if args.comparar:
        comparar(salida, args.trabajos)
        return
    anteriores = PROYECTO / 'outputs' / 'modelos'
    X = load_npz(DATOS_PREPARADOS / 'X_train.npz').astype('float32')
    y = pd.read_csv(DATOS_PREPARADOS / 'y_train.csv')['Severity'].to_numpy()
    V = load_npz(DATOS_PREPARADOS / 'X_validation.npz').astype('float32')
    v = pd.read_csv(DATOS_PREPARADOS / 'y_validation.csv')['Severity'].to_numpy()
    indices, _ = train_test_split(range(len(y)), train_size=250_000, stratify=y, random_state=42)
    familias = [args.familia] if args.familia else GRILLAS
    for familia in familias:
        grilla = GRILLAS[familia]
        raiz = salida / familia
        es_logistica = familia == 'regresion_logistica'
        train, validacion = (X, V) if es_logistica else (X.toarray(), V.toarray())
        muestra = train[indices]
        exploracion = [
            evaluar_ajuste(estimador(familia, params, args.trabajos, exploracion=True),
                           muestra, y[indices], validacion, v, raiz / f'exploracion_{i}',
                           args.trabajos)
            for i, params in enumerate(grilla)
        ]
        mejor = max(range(len(grilla)), key=lambda i: exploracion[i]['f1_macro'])
        referencia = anteriores / familia / 'sin_pesos' / 'modelo.joblib'
        if familia == 'red_neuronal':
            referencia = anteriores / familia / 'dos_capas' / 'modelo.joblib'
        existe_referencia = referencia.exists()
        modelo = (joblib.load(referencia) if existe_referencia
                  else estimador(familia, grilla[0], args.trabajos))
        opciones = {
            'original': evaluar_ajuste(modelo, train, y, validacion, v, raiz / 'original',
                                       args.trabajos, ajustar=not existe_referencia),
            'reentrenado': evaluar_ajuste(estimador(familia, grilla[mejor], args.trabajos),
                                          train, y, validacion, v, raiz / 'reentrenado',
                                          args.trabajos),
        }
        if familia == 'arbol_decision':
            # Valores de ccp_alpha fijos (no se recalculan aquí). TODO: documentar su
            # origen (trayectoria de poda del árbol de referencia). `_prune_tree` es un
            # método interno de scikit-learn 1.7.x y puede cambiar en otras versiones.
            for etiqueta, alpha in [('poda_mejor_f1', 1.4660229824342447e-06),
                                    ('poda_simple', 3.1901475622532575e-06)]:
                podado = copy(modelo)
                podado.ccp_alpha = alpha
                podado._prune_tree()
                opciones[etiqueta] = evaluar_ajuste(podado, train, y, validacion, v,
                                                      raiz / etiqueta, args.trabajos, ajustar=False)
        final = max(opciones, key=lambda opcion: opciones[opcion]['f1_macro'])
        resumen = dict(seleccion=final, opciones=opciones, grilla=grilla,
                       indice_finalista=mejor, referencia_recargada=existe_referencia)
        (raiz / 'seleccion.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
        del train, validacion, muestra
    if args.familia is None:
        comparar(salida, args.trabajos)


def comparar(salida, trabajos):
    finales = {
        familia: json.loads((salida / familia / 'seleccion.json').read_text(encoding='utf-8'))
        for familia in GRILLAS
    }
    seleccion = salida / 'seleccion_previa_test.json'
    seleccion.write_text(json.dumps(finales, indent=2), encoding='utf-8')
    T = load_npz(DATOS_PREPARADOS / 'X_test.npz').astype('float32')
    t = pd.read_csv(DATOS_PREPARADOS / 'y_test.csv')['Severity'].to_numpy()
    densa = T.toarray()
    filas = []
    for familia, datos in finales.items():
        for variante in dict.fromkeys(['original', datos['seleccion']]):
            modelo = joblib.load(salida / familia / variante / 'modelo.joblib')
            with threadpool_limits(limits=trabajos):
                prueba = metricas(modelo, T if familia == 'regresion_logistica' else densa, t)
            filas.append(dict(modelo=familia, variante=variante,
                              f1_macro_validacion=datos['opciones'][variante]['f1_macro'], **prueba))
    pd.DataFrame(filas).to_csv(salida / 'comparacion_test.csv', index=False)


if __name__ == '__main__':
    main()
