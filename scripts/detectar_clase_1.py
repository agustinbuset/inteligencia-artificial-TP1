"""Compara detectores binarios con Random Forest para Severity=1."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from scripts.evaluacion_modelos import guardar_json
from scripts.rutas import DATOS_PREPARADOS, PROYECTO


def cargar(particion):
    labels = pd.read_csv(DATOS_PREPARADOS / f'y_{particion}.csv')
    raw = pd.read_csv(DATOS_PREPARADOS / f'{particion}_raw.csv', usecols=[
        'fila_origen', 'Source', 'Start_Time', 'Street', 'Crossing', 'Junction'])
    if not raw['fila_origen'].equals(labels['fila_origen']):
        raise ValueError(f'Filas desalineadas en {particion}')
    matrix = load_npz(DATOS_PREPARADOS / f'X_{particion}.npz').tocsr()
    if matrix.shape[0] != len(labels):
        raise ValueError(f'Matriz desalineada en {particion}')
    return matrix, labels['Severity'].eq(1).to_numpy(), raw


def contexto(raw):
    street = raw['Street'].str.strip().str.upper()
    kinds = {
        'interestatal': r'^(?:[NSEW] )?I[- ]?\d',
        'ruta': r'^(?:US|SR|SH|CR|CA|TX|FL|NY|PA|NC|OH|IL|WA|WA SR)[- ]?\d|\b(?:HWY|HIGHWAY|ROUTE|RTE)\b',
        'calle': r'\b(?:ST|STREET)\b',
        'avenida': r'\b(?:AVE|AVENUE|BLVD|BOULEVARD)\b',
        'camino': r'\b(?:RD|ROAD|DR|DRIVE|LN|LANE)\b',
        'rampa_salida': r'\b(?:RAMP|EXIT)\b',
    }
    return np.column_stack([
        street.str.contains(pattern, regex=True, na=False).to_numpy(dtype=np.float32)
        for pattern in kinds.values()
    ] + [street.isna().to_numpy(dtype=np.float32)])


def evaluar_umbral(modelo, labels, scores, threshold):
    flagged = scores >= threshold
    positives = int(labels.sum())
    true_positives = int((labels & flagged).sum())
    alerts = int(flagged.sum())
    return dict(modelo=modelo, filas=len(labels), positivos=positives, alertas=alerts,
                verdaderos_positivos=true_positives, falsos_positivos=alerts - true_positives,
                recall=true_positives / positives if positives else None,
                precision=true_positives / alerts if alerts else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'outputs/detector_clase_1')
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=False)
    X, y, train_raw = cargar('train')
    V, v, valid_raw = cargar('validation')
    calibration, evaluation = train_test_split(
        np.arange(len(v)), test_size=0.5, stratify=v, random_state=43)
    rng = np.random.default_rng(42)
    train = np.sort(np.r_[np.flatnonzero(y), rng.choice(np.flatnonzero(~y), 200_000, replace=False)])
    train_dense = X[train].astype(np.float32).toarray()
    del X
    val_dense = V.astype(np.float32).toarray()
    baseline = joblib.load(PROYECTO / 'outputs/modelos/random_forest/sin_pesos/modelo.joblib')
    column = list(baseline.classes_).index(1)
    scores = {'random_forest_actual': baseline.predict_proba(V)[:, column]}
    del V, baseline
    weights = np.where(y[train], 4.0, 1.0)
    for name, extra in [('binario_base', False), ('binario_contexto_vial', True)]:
        train_matrix = np.column_stack([train_dense, contexto(train_raw.iloc[train])]) if extra else train_dense
        val_matrix = np.column_stack([val_dense, contexto(valid_raw)]) if extra else val_dense
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                               min_samples_leaf=40, l2_regularization=2,
                                               early_stopping=False, random_state=42)
        print(f'Entrenando {name} con {len(train):,} filas', flush=True)
        with threadpool_limits(limits=4):
            model.fit(train_matrix, y[train], sample_weight=weights)
            scores[name] = model.predict_proba(val_matrix)[:, 1]
        joblib.dump(model, args.salida / f'{name}.joblib', compress=3)
        del train_matrix, val_matrix

    groups = valid_raw.iloc[evaluation].assign(Anio=lambda frame: frame['Start_Time'].str[:4])
    summaries, cohorts, curves = [], [], []
    for name, score in scores.items():
        # El presupuesto de alertas (1 %) se fija sobre calibración únicamente.
        threshold = float(np.quantile(score[calibration], 0.99, method='higher'))
        result = evaluar_umbral(name, v[evaluation], score[evaluation], threshold)
        result.update(umbral=threshold, alertas_calibracion=int((score[calibration] >= threshold).sum()),
                      average_precision=average_precision_score(v[evaluation], score[evaluation]))
        summaries.append(result)
        precision, recall, thresholds = precision_recall_curve(v[evaluation], score[evaluation])
        curves.append(pd.DataFrame(dict(modelo=name, precision=precision[:-1],
                                        recall=recall[:-1], umbral=thresholds)))
        for field in ('Source', 'Anio'):
            for value, frame in groups.groupby(field, dropna=False):
                ix = frame.index.to_numpy()
                cohort = evaluar_umbral(name, v[ix], score[ix], threshold)
                cohort.update(variable=field, valor=value)
                cohorts.append(cohort)
        print(result, flush=True)
    pd.DataFrame(summaries).to_csv(args.salida / 'comparacion.csv', index=False)
    pd.DataFrame(cohorts).to_csv(args.salida / 'resultados_por_fuente_y_anio.csv', index=False)
    pd.concat(curves, ignore_index=True).to_csv(args.salida / 'curva_precision_recall.csv', index=False)
    guardar_json(args.salida / 'experimento.json', dict(
        train_positivos=int(y.sum()), train_negativos=200_000, peso_positivo=4,
        filas_calibracion=len(calibration), positivos_calibracion=int(v[calibration].sum()),
        filas_evaluacion=len(evaluation), positivos_evaluacion=int(v[evaluation].sum()),
        presupuesto_alertas=0.01, test_utilizado=False))


if __name__ == '__main__':
    main()
