"""Compara el árbol de decisión con 172 y 102 columnas en validación."""

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.tree import DecisionTreeClassifier

from scripts.evaluacion_modelos import comprobar_datos, evaluar, guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO


EXCLUIDAS = ('City', 'County', 'Airport_Code', 'Wind_Direction', 'Traffic_Calming')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'outputs/comparacion_columnas_arbol')
    args = parser.parse_args()
    reference = PROYECTO / 'outputs/modelos/arbol_decision/sin_pesos/ajuste.json'
    params = json.loads(reference.read_text(encoding='utf-8'))['parametros']
    names = pd.read_csv(DATOS_PREPARADOS / 'atributos_finales.csv')['atributo']
    keep = ~names.str.startswith(tuple(f'categoricas__{column}_' for column in EXCLUIDAS))
    if (len(names), int(keep.sum())) != (172, 102):
        raise ValueError('Se esperaban 172 columnas completas y 102 reducidas.')

    args.salida.mkdir(parents=True, exist_ok=False)
    y = pd.read_csv(DATOS_PREPARADOS / 'y_train.csv')['Severity'].to_numpy()
    v = pd.read_csv(DATOS_PREPARADOS / 'y_validation.csv')['Severity'].to_numpy()
    rows = []
    for label, columns in [('102_columnas', keep), ('172_columnas', names.notna())]:
        selected = names[columns]
        indices = selected.index.to_numpy()
        folder = args.salida / label
        folder.mkdir()
        train = load_npz(DATOS_PREPARADOS / 'X_train.npz').tocsc()[:, indices]
        X = train.astype('float32', copy=False).toarray(order='F')
        del train
        comprobar_datos(X, y, selected)
        model = DecisionTreeClassifier(**params)
        print(f'Entrenando {label} con {len(y):,} filas', flush=True)
        start = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - start
        train_metrics = evaluar(model, X, y, folder, 'train')
        del X
        validation = load_npz(DATOS_PREPARADOS / 'X_validation.npz').tocsc()[:, indices]
        V = validation.astype('float32', copy=False).toarray(order='C')
        del validation
        comprobar_datos(V, v, selected)
        metrics = evaluar(model, V, v, folder, 'validation')
        del V
        row = dict(variante=label, columnas=len(selected), segundos_ajuste=elapsed,
                   f1_macro_train=train_metrics['f1_macro'], **metrics)
        rows.append(row)
        selected.to_csv(folder / 'atributos.csv', index=False)
        joblib.dump(model, folder / 'modelo.joblib', compress=3)
        guardar_json(folder / 'ajuste.json', dict(parametros=params, resultado=row))
        print(row, flush=True)

    comparison = pd.DataFrame(rows).sort_values('f1_macro', ascending=False)
    comparison.to_csv(args.salida / 'comparacion_validation.csv', index=False)
    guardar_json(args.salida / 'experimento.json', dict(
        filas_train=len(y), filas_validation=len(v), excluidas=EXCLUIDAS,
        criterio='F1 macro de validación', test_utilizado=False,
        representacion_ajuste='densa_float32', seleccion=comparison.iloc[0]['variante']))


if __name__ == '__main__':
    main()
