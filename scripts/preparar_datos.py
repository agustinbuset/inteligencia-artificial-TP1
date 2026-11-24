"""Preparación reproducible de Accidente. Ejecutar con --help para ver opciones.

Siglas usadas en este módulo:
- EDA: Exploratory Data Analysis, análisis exploratorio de datos.
- IQR: Interquartile Range, rango intercuartílico para detectar extremos.
- PCA: Principal Component Analysis, análisis de componentes principales.
- CV: Cross-Validation, validación cruzada.

Training es entrenamiento, validation es validación y test es prueba.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import gc

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler, Normalizer,
                                   OneHotEncoder, StandardScaler)
from scripts.rutas import DATOS_PREPARADOS


class IQRClipper(TransformerMixin, BaseEstimator):
    """Aprende límites en training; conserva filas y recorta extremos."""
    def fit(self, X, y=None):
        q1, q3 = np.quantile(X, [.25, .75], axis=0)
        width = q3 - q1
        self.lower_ = np.where(width > 0, q1 - 1.5 * width, -np.inf)
        self.upper_ = np.where(width > 0, q3 + 1.5 * width, np.inf)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class SparseToDense(TransformerMixin, BaseEstimator):
    """PCA requiere datos centrados; se limita memoria antes de densificar."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.shape[0] * X.shape[1] * 8 > 512 * 1024**2:
            raise MemoryError('PCA excede 512 MiB. Reducir --max-rows o categorías.')
        return X.toarray() if sparse.issparse(X) else X


class SafeL1Selector(SelectFromModel):
    """Conserva un atributo si L1 anula todo; registra ese caso explícitamente."""
    def _get_support_mask(self):
        mask = super()._get_support_mask()
        self.fallback_ = not mask.any()
        if self.fallback_:
            mask[np.abs(self.estimator_.coef_).mean(axis=0).argmax()] = True
        return mask


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(frame):
    """Reglas fijas de dominio; no estima parámetros sobre el dataset."""
    frame = frame.copy().replace(r'^\s*$', np.nan, regex=True)
    for col in ['TMC', 'Zipcode', 'Number']:
        if col in frame:
            frame[col] = frame[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    bounds = {'Humidity(%)': (0, 100), 'Start_Lat': (-90, 90),
              'Start_Lng': (-180, 180), 'Wind_Speed(mph)': (0, np.inf),
              'Visibility(mi)': (0, np.inf), 'Precipitation(in)': (0, np.inf),
              'Pressure(in)': (0, np.inf)}
    for col, (low, high) in bounds.items():
        if col in frame:
            values = pd.to_numeric(frame[col], errors='coerce')
            frame[col] = values.where(values.between(low, high))
    if 'Start_Time' in frame:
        dates = pd.to_datetime(frame['Start_Time'], errors='coerce')
        frame['Start_Hour'] = dates.dt.hour
        frame['Start_Weekday'] = dates.dt.dayofweek
        frame['Start_Month'] = dates.dt.month
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


EXCLUDED = ['Severity', 'ID', 'End_Time', 'End_Lat', 'End_Lng', 'Distance(mi)',
            'Description', 'Number', 'Street', 'Zipcode', 'Weather_Timestamp', 'Start_Time']


def features(part):
    # Descartar texto y columnas inutilizadas antes de copiar/limpiar millones de filas.
    result = clean(part.drop(columns=[c for c in EXCLUDED if c != 'Start_Time'],
                             errors='ignore')).drop(columns='Start_Time', errors='ignore')
    for col in result.select_dtypes(exclude='number'):
        result[col] = result[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    return result


def stratified_sample(frame, limit, seed):
    if limit and len(frame) > limit:
        return train_test_split(frame, train_size=limit, stratify=frame['Severity'],
                                random_state=seed)[0]
    return frame


def has_converged(pipeline):
    estimators = [pipeline.named_steps['modelo']]
    if 'seleccion' in pipeline.named_steps:
        estimators.append(pipeline.named_steps['seleccion'].estimator_)
    return all(int(np.max(model.n_iter_)) < model.max_iter for model in estimators)


def explore(X, y, output):
    """EDA de entrenamiento: test y validation quedan reservados."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    X.describe(include='all').to_csv(output / 'estadisticas_train.csv')
    X.isna().mean().sort_values(ascending=False).to_csv(output / 'nulos_train.csv')
    y.value_counts().sort_index().to_csv(output / 'clases_train.csv')
    numeric = X.select_dtypes(include='number')
    numeric.corr().to_csv(output / 'correlaciones_train.csv')
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    width = q3 - q1
    ((numeric.lt(q1 - 1.5 * width) | numeric.gt(q3 + 1.5 * width))
     & numeric.notna()).sum().to_csv(output / 'outliers_iqr_train.csv')
    y.value_counts().sort_index().plot.bar(title='Severidad: entrenamiento')
    plt.tight_layout()
    plt.savefig(output / 'clases_train.png')
    plt.close()
    if numeric.shape[1]:
        numeric.hist(figsize=(16, 12), bins=30)
        plt.tight_layout()
        plt.savefig(output / 'histogramas_train.png')
        plt.close('all')


def make_pipeline(numeric, categorical, variant, seed):
    scaler = MinMaxScaler() if variant == 'minmax' else StandardScaler()
    if variant == 'discretizacion':
        scaler = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile',
                                  subsample=200_000, random_state=seed)
    num = Pipeline([('imputar', SimpleImputer(strategy='median', keep_empty_features=True)),
                    ('outliers', IQRClipper()), ('escalar', scaler)])
    if variant == 'pca':
        # PCA centrado sobre magnitudes numéricas; categorías conservan one-hot.
        num.steps.extend([('constantes', VarianceThreshold()),
                          ('pca', PCA(n_components=.95, svd_solver='full'))])
    cat = Pipeline([('imputar', SimpleImputer(strategy='constant', fill_value='Desconocido',
                                             keep_empty_features=True)),
                    ('codificar', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                               min_frequency=.01, max_categories=30))])
    steps = [('atributos', ColumnTransformer([('numericas', num, numeric),
                                               ('categoricas', cat, categorical)],
                                              sparse_threshold=1.0)),
             ('varianza', VarianceThreshold())]
    if variant == 'normalizacion':
        steps.append(('normalizar', Normalizer(norm='l2')))
    if variant == 'seleccion_l1':
        steps.append(('seleccion', SafeL1Selector(LogisticRegression(
            penalty='l1', solver='saga', C=.1, max_iter=2000, random_state=seed),
            threshold=1e-8)))
    steps.append(('modelo', LogisticRegression(C=1, penalty='l2', solver='lbfgs',
                                               max_iter=1000, random_state=seed)))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='CSV real de Accidente')
    parser.add_argument('--output', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--sep', default=',')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Muestra aleatoria opcional DESPUÉS de cargar el CSV completo')
    parser.add_argument('--min-rows-per-class', type=int, default=None,
                        help='Reserva este mínimo por severidad antes de completar la muestra estratificada')
    parser.add_argument('--comparison-max-rows', type=int, default=None,
                        help='Limita SOLO la comparación/CV; ajuste final usa todo training')
    parser.add_argument('--batch-size', type=int, default=25000)
    parser.add_argument('--resume', action='store_true',
                        help='Reutilizar copia y comparaciones existentes tras una interrupción')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'No existe el CSV: {args.input}')
    if args.output.exists() and not args.resume:
        parser.error('La carpeta de salida ya existe. Elegir otra para preservar resultados.')
    if args.folds < 2 or (args.max_rows is not None and args.max_rows < 40):
        parser.error('--folds debe ser >= 2 y --max-rows >= 40')
    if args.batch_size < 1 or (args.comparison_max_rows is not None and args.comparison_max_rows < 40):
        parser.error('--batch-size debe ser positivo y --comparison-max-rows >= 40')
    minimum = args.min_rows_per_class
    if minimum is not None and (minimum < 1 or args.max_rows is None or args.max_rows <= 4 * minimum):
        parser.error('--min-rows-per-class debe ser positivo y requiere --max-rows mayor a cuatro veces el mínimo')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    out = args.output
    copy = out / 'dataset_original_copia.csv'
    print('Copiando y verificando original...', flush=True)
    source_hash = sha256(args.input)
    if not (args.resume and copy.exists()):
        shutil.copy2(args.input, copy)
    if sha256(copy) != source_hash:
        raise RuntimeError('La copia no coincide con el original.')
    (out / 'integridad_original.json').write_text(json.dumps({
        'origen': str(args.input.resolve()), 'sha256_original': source_hash,
        'sha256_copia': source_hash, 'bytes': copy.stat().st_size}, indent=2), encoding='utf-8')
    print('Leyendo copia completa...', flush=True)
    frame = pd.read_csv(copy, sep=args.sep, dtype={'Zipcode': str, 'TMC': str, 'Number': str})
    if 'Severity' not in frame:
        raise ValueError('El CSV debe contener Severity.')
    initial = len(frame)
    print(f'{initial:,} filas leídas. Curación y particionado...', flush=True)
    frame = frame.drop_duplicates()
    duplicates = initial - len(frame)
    # IDs repetidos no pueden quedar repartidos entre conjuntos.
    if 'ID' in frame and frame['ID'].dropna().duplicated().any():
        raise ValueError('Hay IDs repetidos con registros distintos; resolver antes de particionar.')
    labels = pd.to_numeric(frame['Severity'], errors='coerce')
    valid = labels.isin([1, 2, 3, 4])
    invalid_labels = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame['Severity'] = labels.loc[valid].astype(int)
    if minimum is not None:
        counts = frame['Severity'].value_counts().reindex([1, 2, 3, 4], fill_value=0)
        if (counts < minimum).any():
            raise ValueError(f'No hay {minimum} filas disponibles por severidad: {counts.to_dict()}')
    if args.max_rows and args.max_rows < len(frame) and minimum is not None:
        severity = frame['Severity']
        reserved = severity.groupby(severity).sample(n=minimum, random_state=args.seed)
        remaining = severity.drop(index=reserved.index)
        size = args.max_rows - len(reserved)
        counts_remaining = remaining.value_counts()
        if counts_remaining.min() >= 2 and min(size, len(remaining) - size) >= len(counts_remaining):
            sample, _ = train_test_split(remaining.index, train_size=size,
                                         stratify=remaining, random_state=args.seed)
        else:
            sample = remaining.sample(n=size, random_state=args.seed).index
        frame = frame.loc[reserved.index.append(sample)]
    elif args.max_rows and args.max_rows < len(frame):
        frame, _ = train_test_split(frame, train_size=args.max_rows,
                                    stratify=frame['Severity'], random_state=args.seed)
    train, reserve = train_test_split(frame, test_size=.30, stratify=frame['Severity'],
                                      random_state=args.seed)
    validation, test = train_test_split(reserve, test_size=.50,
                                       stratify=reserve['Severity'], random_state=args.seed)
    del frame, reserve, labels, valid
    gc.collect()
    for name, part in [('train', train), ('validation', validation), ('test', test)]:
        raw_path = out / f'{name}_raw.csv'
        if args.resume and raw_path.exists():
            previous_index = pd.read_csv(raw_path, usecols=['fila_origen'])['fila_origen'].to_numpy()
            if not np.array_equal(previous_index, part.index.to_numpy()):
                raise ValueError('Las particiones existentes no corresponden a esta configuración.')
        else:
            part.to_csv(raw_path, index_label='fila_origen')
    print('EDA sobre entrenamiento completo...', flush=True)
    X, y = features(train), train['Severity']
    if not (args.resume and (out / 'histogramas_train.png').exists()):
        explore(X, y, out)
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [col for col in X if col not in numeric]
    if y.value_counts().min() < args.folds:
        raise ValueError('Cada clase de training necesita al menos --folds registros.')
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    comparison_train = stratified_sample(train, args.comparison_max_rows, args.seed)
    comparison_validation = stratified_sample(validation, args.comparison_max_rows, args.seed)
    X_comparison = X.loc[comparison_train.index]
    y_comparison = comparison_train['Severity']
    X_validation = features(comparison_validation)
    if y_comparison.value_counts().min() < args.folds:
        raise ValueError('La muestra de comparación no tiene suficientes registros por clase.')
    results, candidates = [], {}
    previous_results = {}
    if args.resume and (out / 'comparacion.csv').exists():
        previous_results = {row['variante']: row for row in
                            pd.read_csv(out / 'comparacion.csv').to_dict('records')}
    for variant in ['estandarizacion', 'minmax', 'normalizacion', 'discretizacion',
                    'seleccion_l1', 'pca']:
        saved = out / f'pipeline_comparacion_{variant}.joblib'
        if variant in previous_results and saved.exists():
            row = previous_results[variant]
            if row['filas_cv'] != len(X_comparison) or row['filas_validation'] != len(comparison_validation):
                raise ValueError('El límite de comparación difiere del utilizado anteriormente.')
            pipeline = joblib.load(saved)
            if pipeline.named_steps['modelo'].random_state != args.seed:
                raise ValueError('La semilla difiere del ajuste guardado.')
            row['convergio_ajuste_comparacion'] = has_converged(pipeline)
            # Un ajuste de comparación no convergente no es elegible para el ajuste final.
            # Ejecuciones antiguas no guardaban el estado por fold; no inventarlo.
            row.setdefault('convergio_cv', None)
            results.append(row)
            candidates[variant] = pipeline
            print(f'Reutilizando {variant} (convergencia: {has_converged(pipeline)})', flush=True)
            continue
        print(f'Evaluando {variant}...', flush=True)
        pipeline = make_pipeline(numeric, categorical, variant, args.seed)
        scores = cross_validate(pipeline, X_comparison, y_comparison, cv=cv, scoring='f1_macro',
                                error_score='raise', n_jobs=1, return_estimator=True)
        pipeline.fit(X_comparison, y_comparison)
        feature_names = pipeline.named_steps['atributos'].get_feature_names_out()
        feature_names = feature_names[pipeline.named_steps['varianza'].get_support()]
        if variant == 'seleccion_l1':
            pd.Series(feature_names[pipeline.named_steps['seleccion'].get_support()],
                      name='atributo').to_csv(out / 'seleccion_l1.csv', index=False)
            (out / 'seleccion_l1_estado.json').write_text(json.dumps({
                'sin_atributos_relevantes_segun_umbral': pipeline.named_steps['seleccion'].fallback_,
                'politica': 'Si todos los coeficientes son cero, conservar uno para poder comparar.'
            }, indent=2), encoding='utf-8')
        if variant == 'pca':
            numeric_pipeline = pipeline.named_steps['atributos'].named_transformers_['numericas']
            decomposition = numeric_pipeline.named_steps['pca']
            pca_names = np.array(numeric)[numeric_pipeline.named_steps['constantes'].get_support()]
            pd.DataFrame(decomposition.components_, columns=pca_names).to_csv(
                out / 'cargas_pca.csv', index=False)
            np.savetxt(out / 'varianza_pca.csv', decomposition.explained_variance_ratio_,
                       delimiter=',')
        validation_score = f1_score(comparison_validation['Severity'], pipeline.predict(X_validation),
                                    average='macro', zero_division=0)
        results.append({'variante': variant, 'cv_f1_macro': scores['test_score'].mean(),
                        'cv_desvio': scores['test_score'].std(),
                        'validation_f1_macro': validation_score,
                        'filas_cv': len(X_comparison), 'filas_validation': len(comparison_validation),
                        'convergio_cv': all(has_converged(p) for p in scores['estimator']),
                        'convergio_ajuste_comparacion': has_converged(pipeline)})
        pd.DataFrame(results).to_csv(out / 'comparacion.csv', index=False)
        joblib.dump(pipeline, out / f'pipeline_comparacion_{variant}.joblib')
        candidates[variant] = pipeline
    pd.DataFrame(results).to_csv(out / 'comparacion.csv', index=False)
    eligible = [r for r in results if r['convergio_cv'] != False
                and r['convergio_ajuste_comparacion']]
    if not eligible:
        raise RuntimeError('Ninguna alternativa convergió. Revisar escalas y regularización.')
    best = max(eligible, key=lambda row: row['validation_f1_macro'])['variante']
    pipeline = candidates[best]
    print(f'Ajuste final ({best}) sobre {len(X):,} filas de training...', flush=True)
    pipeline.fit(X, y)
    if not has_converged(pipeline):
        raise RuntimeError('El ajuste final no convergió; no se generan métricas finales.')
    del X, X_comparison, X_validation, candidates, comparison_train, comparison_validation
    gc.collect()
    # Se evalúa test una sola vez, después de elegir con validation.
    test_predictions = np.concatenate([pipeline.predict(features(test.iloc[start:start + args.batch_size]))
                                      for start in range(0, len(test), args.batch_size)])
    report = classification_report(test['Severity'], test_predictions,
                                    output_dict=True, zero_division=0)
    (out / 'metricas_test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, part in [('train', train), ('validation', validation), ('test', test)]:
        print(f'Exportando {name}: {len(part):,} filas...', flush=True)
        blocks = [sparse.csr_matrix(pipeline[:-1].transform(features(part.iloc[start:start + args.batch_size])))
                  for start in range(0, len(part), args.batch_size)]
        matrix = sparse.vstack(blocks, format='csr')
        sparse.save_npz(out / f'X_{name}.npz', matrix)
        del blocks, matrix
        part['Severity'].to_csv(out / f'y_{name}.csv', index_label='fila_origen')
    names = pipeline.named_steps['atributos'].get_feature_names_out()
    names = names[pipeline.named_steps['varianza'].get_support()]
    if best == 'seleccion_l1':
        names = names[pipeline.named_steps['seleccion'].get_support()]
    if best == 'pca':
        numeric_pipeline = pipeline.named_steps['atributos'].named_transformers_['numericas']
        pca = numeric_pipeline.named_steps['pca']
        pca_names = np.array(numeric)[numeric_pipeline.named_steps['constantes'].get_support()]
        pd.DataFrame(pca.components_, columns=pca_names).to_csv(out / 'cargas_pca_final.csv', index=False)
        np.savetxt(out / 'varianza_pca.csv', pca.explained_variance_ratio_, delimiter=',')
    pd.Series(names, name='atributo').to_csv(out / 'atributos_finales.csv', index=False)
    pd.DataFrame(np.abs(pipeline.named_steps['modelo'].coef_).mean(axis=0),
                 index=names, columns=['coeficiente_absoluto_medio']).sort_values(
                     'coeficiente_absoluto_medio', ascending=False).to_csv(out / 'relevancia.csv')
    joblib.dump(pipeline, out / 'pipeline.joblib')
    final_hash = sha256(args.input)
    if final_hash != source_hash:
        raise RuntimeError('El archivo original cambió durante la ejecución.')
    manifest = {'origen': str(args.input.resolve()), 'sha256': source_hash,
                'sha256_original_al_final': final_hash, 'original_intacto': True,
                'filas_originales': initial, 'duplicados_eliminados': duplicates,
                'objetivos_invalidos_eliminados': invalid_labels, 'mejor_variante': best,
                'seed': args.seed, 'folds': args.folds,
                'limite_muestra': args.max_rows, 'minimo_por_clase_muestra': minimum,
                'filas': {'train': len(train), 'validation': len(validation), 'test': len(test)},
                'filas_comparacion_training': len(y_comparison),
                'limite_comparacion': args.comparison_max_rows,
                'pca': '95% de varianza numérica; categorías conservadas con one-hot',
                'excluidas': EXCLUDED, 'version_sklearn': __import__('sklearn').__version__}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Preparación terminada: {out.resolve()}')


if __name__ == '__main__':
    # Clases serializadas con nombre de módulo estable para reutilizar joblib.
    from scripts.preparar_datos import main as ejecutar
    ejecutar()
