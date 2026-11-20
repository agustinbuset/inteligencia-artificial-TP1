"""Mide importancia por permutación y reentrena árboles con menos columnas."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from scripts.comparar_columnas_arbol import EXCLUIDAS
from scripts.evaluacion_modelos import CLASES, comprobar_datos, evaluar, guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO


def variable(name):
    kind, value = name.split('__', 1)
    if kind == 'numericas':
        return value
    if value.endswith('_infrequent_sklearn'):
        return value.removesuffix('_infrequent_sklearn')
    return value.rsplit('_', 1)[0]


def importancia(model, X, y, groups, permutations, baseline):
    rows = []
    for name, columns in groups.items():
        original = X[:, columns].copy()
        drops = []
        for permutation in permutations:
            X[:, columns] = original[permutation]
            score = f1_score(y, model.predict(X), labels=CLASES, average='macro')
            drops.append(baseline - score)
        X[:, columns] = original
        rows.append(dict(atributo=name, columnas=len(columns),
                         divisiones=int(np.isin(model.tree_.feature, columns).sum()),
                         caida_f1_macro=np.mean(drops), desvio_caida=np.std(drops),
                         importancia_impureza=model.feature_importances_[columns].sum()))
    return pd.DataFrame(rows).sort_values('caida_f1_macro', ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'outputs/analisis_columnas_arbol')
    args = parser.parse_args()
    names = pd.read_csv(DATOS_PREPARADOS / 'atributos_finales.csv')['atributo']
    variables = names.map(variable)
    groups = variables.groupby(variables, sort=False).indices
    if (len(names), len(groups)) != (172, 38):
        raise ValueError('Se esperaban 172 columnas codificadas y 38 variables preparadas.')
    reference = PROYECTO / 'outputs/comparacion_columnas_arbol/172_columnas/modelo.joblib'
    model = joblib.load(reference)
    validation = load_npz(DATOS_PREPARADOS / 'X_validation.npz').tocsr()
    v = pd.read_csv(DATOS_PREPARADOS / 'y_validation.csv')['Severity'].to_numpy()
    sample, _ = train_test_split(np.arange(len(v)), train_size=100_000,
                                 stratify=v, random_state=42)
    P = validation[sample].astype('float32').toarray()
    comprobar_datos(P, v[sample], names)
    baseline = f1_score(v[sample], model.predict(P), labels=CLASES, average='macro')
    rng = np.random.default_rng(42)
    permutations = [rng.permutation(len(sample)) for _ in range(5)]
    args.salida.mkdir(parents=True, exist_ok=False)
    per_group = importancia(model, P, v[sample], groups, permutations, baseline)
    per_column = importancia(model, P, v[sample], {
        name: [i] for i, name in names.items()}, permutations, baseline)
    per_column.insert(1, 'variable', per_column['atributo'].map(variable))
    per_column.to_csv(args.salida / 'importancia_172_columnas.csv', index=False)
    per_group.to_csv(args.salida / 'importancia_38_variables.csv', index=False)
    del P
    print('Importancias completas guardadas', flush=True)

    base_exclusions = (*EXCLUIDAS, 'Weather_Condition')
    eligible = ~variables.isin(base_exclusions)
    if int(eligible.sum()) != 89:
        raise ValueError('Se esperaban 89 columnas en el recorte de partida.')
    train = load_npz(DATOS_PREPARADOS / 'X_train.npz').tocsc()
    y = pd.read_csv(DATOS_PREPARADOS / 'y_train.csv')['Severity'].to_numpy()
    rows = []
    for minimum in (0.001, 0.005):
        relevant = per_group.loc[per_group['caida_f1_macro'] >= minimum, 'atributo']
        selected = names[eligible & variables.isin(relevant)]
        folder = args.salida / f'{len(selected)}_columnas'
        folder.mkdir()
        X = train[:, selected.index].astype('float32').toarray(order='F')
        comprobar_datos(X, y, selected)
        reduced = clone(model)
        print(f'Entrenando {len(selected)} columnas con {len(y):,} filas', flush=True)
        start = time.perf_counter()
        reduced.fit(X, y)
        elapsed = time.perf_counter() - start
        train_metrics = evaluar(reduced, X, y, folder, 'train')
        del X
        V = validation[:, selected.index].astype('float32').toarray()
        comprobar_datos(V, v, selected)
        metrics = evaluar(reduced, V, v, folder, 'validation')
        del V
        selected.to_csv(folder / 'atributos.csv', index=False)
        joblib.dump(reduced, folder / 'modelo.joblib', compress=3)
        row = dict(columnas=len(selected), variables=selected.map(variable).nunique(),
                   caida_minima=minimum, segundos_ajuste=elapsed,
                   f1_macro_train=train_metrics['f1_macro'], **metrics)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.salida / 'comparacion_validation.csv', index=False)
        print(row, flush=True)
    guardar_json(args.salida / 'experimento.json', dict(
        referencia=str(reference), filas_importancia=len(sample), f1_macro_base=baseline,
        repeticiones=len(permutations), filas_train=len(y), filas_validation=len(v),
        exclusiones_base=base_exclusions, test_utilizado=False))


if __name__ == '__main__':
    main()
