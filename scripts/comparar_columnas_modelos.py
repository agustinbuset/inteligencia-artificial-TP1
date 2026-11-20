"""Evalúa los modelos sin pesos sobre las mismas particiones."""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from scripts.comparar_columnas_arbol import EXCLUIDAS
from scripts.evaluacion_modelos import CLASES, comprobar_datos, evaluar, guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path)
    parser.add_argument('--columnas-51', action='store_true',
                        help='Evalúa el recorte de 51 columnas y agrega los resultados anteriores.')
    args = parser.parse_args()
    previous_output = PROYECTO / 'outputs/comparacion_columnas_modelos'
    if args.salida is None:
        args.salida = (PROYECTO / 'outputs/comparacion_columnas_51'
                       if args.columnas_51 else previous_output)
    names = pd.read_csv(DATOS_PREPARADOS / 'atributos_finales.csv')['atributo']
    keep = ~names.str.startswith(tuple(f'categoricas__{column}_' for column in EXCLUIDAS))
    if (len(names), int(keep.sum())) != (172, 102):
        raise ValueError('Se esperaban 172 columnas completas y 102 reducidas.')
    candidates = {102: names[keep], 172: names}
    reduced_tree = PROYECTO / 'outputs/analisis_columnas_arbol'
    if args.columnas_51:
        recorded = pd.read_csv(reduced_tree / '51_columnas/atributos.csv')['atributo']
        selected = names[names.isin(recorded)]
        if len(selected) != 51 or selected.tolist() != recorded.tolist():
            raise ValueError('El recorte de 51 columnas no coincide con la matriz preparada.')
        candidates = {51: selected}
    y = pd.read_csv(DATOS_PREPARADOS / 'y_train.csv')['Severity'].to_numpy()
    v = pd.read_csv(DATOS_PREPARADOS / 'y_validation.csv')['Severity'].to_numpy()
    args.salida.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in ('regresion_logistica', 'random_forest', 'arbol_decision'):
        reference = PROYECTO / 'outputs/modelos' / family / 'sin_pesos'
        if family == 'arbol_decision':
            reference = PROYECTO / 'outputs/comparacion_columnas_arbol/172_columnas'
        original = joblib.load(reference / 'modelo.joblib')
        X_full = load_npz(DATOS_PREPARADOS / 'X_train.npz').tocsr()
        V_full = load_npz(DATOS_PREPARADOS / 'X_validation.npz').tocsr()
        for count, selected in candidates.items():
            folder = args.salida / family / f'{count}_columnas'
            folder.mkdir(parents=True)
            X = X_full[:, selected.index] if count != 172 else X_full
            V = V_full[:, selected.index] if count != 172 else V_full
            if family != 'regresion_logistica':
                X = X.astype('float32', copy=False).toarray()
                V = V.astype('float32', copy=False).toarray()
            comprobar_datos(X, y, selected)
            comprobar_datos(V, v, selected)
            source = reference
            if family == 'arbol_decision':
                source = reference.parent / f'{count}_columnas'
            reused = count == 172 or family == 'arbol_decision'
            if family == 'arbol_decision' and count == 51:
                source = reduced_tree / '51_columnas'
                model = joblib.load(source / 'modelo.joblib')
                if model.get_params() != original.get_params():
                    raise ValueError('El árbol de 51 columnas tiene otros hiperparámetros.')
                tree_results = pd.read_csv(reduced_tree / 'comparacion_validation.csv')
                elapsed = float(tree_results.loc[tree_results['columnas'].eq(51), 'segundos_ajuste'].item())
                converged = True
            elif reused:
                saved = json.loads((source / 'ajuste.json').read_text(encoding='utf-8'))
                model = original if count == 172 else joblib.load(source / 'modelo.joblib')
                elapsed = saved['resultado']['segundos_ajuste']
                converged = saved['resultado'].get('convergio', True)
            else:
                model = clone(original)
                print(f'Entrenando {family} con {count} columnas y {len(y):,} filas', flush=True)
                start = time.perf_counter()
                with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(X, y)
                elapsed = time.perf_counter() - start
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                joblib.dump(model, folder / 'modelo.joblib', compress=3)
            with threadpool_limits(limits=4):
                train = evaluar(model, X, y, folder, 'train')
                validation = evaluar(model, V, v, folder, 'validation')
            del X, V
            report = json.loads((folder / 'metricas_validation.json').read_text(encoding='utf-8'))
            if reused:
                previous = json.loads((source / 'metricas_validation.json').read_text(encoding='utf-8'))
                if report != previous:
                    raise ValueError(f'Las métricas de {family}/{count} difieren del ajuste original.')
            row = dict(modelo=family, columnas=count, ajuste_reutilizado=reused,
                       convergio=converged, segundos_ajuste=elapsed, f1_macro_train=train['f1_macro'],
                       brecha_f1_macro=train['f1_macro'] - validation['f1_macro'], **validation,
                       **{f'recall_clase_{c}': report[str(c)]['recall'] for c in CLASES})
            rows.append(row)
            selected.to_csv(folder / 'atributos.csv', index=False)
            guardar_json(folder / 'ajuste.json', dict(
                parametros=model.get_params(), resultado=row, origen=str(source) if reused else None))
            print(row, flush=True)
        del X_full, V_full

    comparison = pd.DataFrame(rows)
    if args.columnas_51:
        prior = pd.read_csv(previous_output / 'comparacion_validation.csv')
        comparison = pd.concat([comparison, prior], ignore_index=True)
    comparison = comparison.sort_values('f1_macro', ascending=False)
    comparison.to_csv(args.salida / 'comparacion_validation.csv', index=False)
    guardar_json(args.salida / 'experimento.json', dict(
        filas_train=len(y), filas_validation=len(v), excluidas_en_102=EXCLUIDAS,
        subconjuntos=list(candidates),
        variante='sin pesos', criterio='F1 macro de validación', test_utilizado=False))


if __name__ == '__main__':
    main()
