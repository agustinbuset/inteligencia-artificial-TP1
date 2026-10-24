"""Genera un informe legible y trazabilidad adicional a partir de las salidas."""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from scripts.preparar_datos import stratified_sample
from scripts.rutas import DATOS_PREPARADOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--advertencia-l1-cv', action='store_true',
                        help='Registrar aviso de convergencia L1 observado en la consola de CV')
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    scores = pd.read_csv(out / 'comparacion.csv')
    metrics = json.loads((out / 'metricas_test.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(out / 'pipeline.joblib')
    raw_train = pd.read_csv(out / 'train_raw.csv', usecols=['fila_origen', 'Severity'])
    sample = stratified_sample(raw_train, manifest['limite_comparacion'], manifest['seed'])
    folds = np.zeros(len(sample), dtype=int)
    cv = StratifiedKFold(manifest['folds'], shuffle=True, random_state=manifest['seed'])
    for i, (_, valid) in enumerate(cv.split(sample, sample['Severity']), start=1):
        folds[valid] = i
    fold_table = sample.copy()
    fold_table['fold_validacion'] = folds
    fold_table.to_csv(out / 'folds_comparacion.csv', index=False)
    sample_counts = sample['Severity'].value_counts().sort_index()
    sample_counts.to_csv(out / 'clases_muestra_cv.csv')
    numeric = pipeline.named_steps['atributos'].named_transformers_['numericas']
    encoded_names = pipeline.named_steps['atributos'].get_feature_names_out()
    constant_mask = ~pipeline.named_steps['varianza'].get_support()
    pd.Series(encoded_names[constant_mask], name='atributo_constante_eliminado').to_csv(
        out / 'atributos_constantes_eliminados.csv', index=False)
    final_feature_count = len(pd.read_csv(out / 'atributos_finales.csv'))
    numeric_names = pipeline.named_steps['atributos'].transformers_[0][2]
    pd.DataFrame({'atributo': numeric_names,
                  'mediana_imputacion': numeric.named_steps['imputar'].statistics_,
                  'limite_inferior_iqr': numeric.named_steps['outliers'].lower_,
                  'limite_superior_iqr': numeric.named_steps['outliers'].upper_}).to_csv(
                      out / 'parametros_numericos_finales.csv', index=False)
    missing = pd.read_csv(out / 'nulos_train.csv', index_col=0).iloc[:, 0]
    train_counts = raw_train['Severity'].value_counts().sort_index()
    pca_pipeline = joblib.load(out / 'pipeline_comparacion_pca.joblib')
    pca = pca_pipeline.named_steps['atributos'].named_transformers_['numericas'].named_steps['pca']
    l1_pipeline = joblib.load(out / 'pipeline_comparacion_seleccion_l1.joblib')
    l1 = l1_pipeline.named_steps['seleccion']
    l1_iterations = int(np.max(l1.estimator_.n_iter_))
    l1_limit = l1.estimator_.max_iter
    final_iterations = int(np.max(pipeline.named_steps['modelo'].n_iter_))
    convergence = {'l1_comparacion_iteraciones': l1_iterations, 'l1_limite': l1_limit,
                   'advertencia_l1_cv_observada': args.advertencia_l1_cv,
                   'l1_ajuste_comparacion_alcanzo_limite': l1_iterations >= l1_limit,
                   'modelo_final_iteraciones': final_iterations,
                   'modelo_final_limite': pipeline.named_steps['modelo'].max_iter}
    (out / 'convergencia.json').write_text(json.dumps(convergence, indent=2), encoding='utf-8')
    lines = [
        '# Preparación del dataset Accidente', '',
        f"Se procesaron {manifest['filas_originales']:,} registros del CSV real.",
        'El original se conservó sin modificaciones. La copia es idéntica byte por byte,',
        'con SHA-256 comprobado antes y después del procesamiento.', '',
        f"SHA-256: `{manifest['sha256']}`", '',
        f"Duplicados exactos eliminados: {manifest['duplicados_eliminados']:,}.",
        f"Objetivos ausentes o inválidos descartados: {manifest['objetivos_invalidos_eliminados']:,}.", '',
        '## Particiones', '', '| Conjunto | Filas |', '| --- | ---: |',
    ]
    lines += [f'| {name} | {count:,} |' for name, count in manifest['filas'].items()]
    lines += ['', 'División aleatoria estratificada 70/15/15, semilla 42.',
              'La validación cruzada usa únicamente una muestra estratificada de training;',
              f"se compararon {manifest['filas_comparacion_training']:,} registros con {manifest['folds']} folds.",
              'La validación de comparación usa hasta el mismo límite de filas de validation.',
              'El ajuste final usa TODO training y la evaluación final usa TODO test.', '',
              '## EDA de entrenamiento', '', '| Severidad | Training | Muestra para CV |',
              '| --- | ---: | ---: |']
    lines += [f'| {label} | {count:,} | {int(sample_counts.get(label, 0)):,} |'
              for label, count in train_counts.items()]
    lines += ['', 'La clase 1 es muy escasa. Las estimaciones por fold para esa clase',
              'son inestables por el pequeño número de ejemplos en la muestra.', '',
              '| Atributo | Nulos en training antes de imputar |', '| --- | ---: |']
    lines += [f'| {name} | {value:.2%} |' for name, value in missing.head(10).items()]
    lines += ['', 'Los valores físicamente inválidos pasan a nulos. Se imputan números con',
              'medianas y categorías con Desconocido; se recortan extremos por IQR cuando',
              'el IQR es mayor que cero. Un extremo estadístico no implica un error de registro.',
              'Los parámetros finales están en `parametros_numericos_finales.csv`.', '',
              'Se excluyen identificadores, dirección de alta cardinalidad, texto libre y',
              'variables posteriores al inicio. Se generan hora, día semanal y mes.',
              'One-hot agrupa categorías infrecuentes y se eliminan atributos constantes.', '',
              f'La matriz final contiene {final_feature_count} columnas tras codificación y selección.',
              'Ver `atributos_constantes_eliminados.csv` y `atributos_finales.csv`.', '',
              '## Comparación de técnicas', '',
              '| Variante | F1 macro CV | Desvío entre folds | F1 macro validation |',
              '| --- | ---: | ---: | ---: |']
    lines += [f"| {r.variante} | {r.cv_f1_macro:.4f} | {r.cv_desvio:.4f} | {r.validation_f1_macro:.4f} |"
              for r in scores.itertuples()]
    lines += ['', f"Variante elegida por validation entre ajustes convergentes: **{manifest['mejor_variante']}**.",
              'Las técnicas son alternativas comparadas; no se aplican todas consecutivamente.',
              f'PCA sobre números estandarizados: {pca.n_components_} componentes,',
              f'{pca.explained_variance_ratio_.sum():.2%} de varianza numérica explicada.',
              'Las categorías codificadas se conservan fuera de PCA.',
              f'L1 conserva {int(l1.get_support().sum())} atributos en el ajuste de comparación.',
              'Los modelos de referencia usan regularización L2.', '',
              ('Se observaron avisos de convergencia L1 durante CV; esa comparación es exploratoria.'
               if args.advertencia_l1_cv else 'Ver el estado de convergencia de los ajustes guardados.'),
              'Consultar `convergencia.json` y los avisos de ejecución antes de interpretar L1.', '',
              '## Evaluación final', '',
              f"F1 macro sobre test completo: **{metrics['macro avg']['f1-score']:.4f}**.",
              f"Accuracy sobre test completo: {metrics['accuracy']:.4f}.",
              'Consultar métricas por clase en `metricas_test.json`: accuracy no resume',
              'adecuadamente el rendimiento frente al fuerte desbalance de clases.', '',
              '| Clase | Ejemplos test | Precisión | Recall | F1 |',
              '| --- | ---: | ---: | ---: | ---: |',
              *[f"| {label} | {int(value['support']):,} | {value['precision']:.4f} | {value['recall']:.4f} | {value['f1-score']:.4f} |"
                for label, value in metrics.items() if label in ['1', '2', '3', '4']], '',
              'El clasificador es una referencia para comparar preparaciones, no un modelo',
              'optimizado para el desbalance. La tabla por clase muestra sus límites reales.', '',
              '## Archivos para trabajar', '',
              '- `Accidente_limpio_v1.csv`: tabla limpia legible, sin escalado ni codificación de categorías.',
              '- `dataset_original_copia.csv`: respaldo exacto del original.',
              '- `train_raw.csv`, `validation_raw.csv`, `test_raw.csv`: particiones con fila de origen.',
              '- `X_*.npz`, `y_*.csv`: matrices preparadas y etiquetas alineadas.',
              '- `atributos_finales.csv`: nombres y orden de columnas de las matrices.',
              '- `pipeline.joblib`: preparación y modelo finales ajustados en todo training.',
              '- `pipeline_comparacion_*.joblib`: seis variantes ajustadas en la muestra.',
              '- `folds_comparacion.csv`: fila de origen y fold reservado en cada iteración.',
              '- Tablas y gráficos de EDA, cargas de PCA, atributos L1 y coeficientes finales.',
              '- `verificacion.json`: integridad, cobertura y alineación verificadas.', '',
              'Las particiones aleatorias no estiman por sí solas la generalización a años',
              'futuros o regiones nuevas; esos escenarios requieren particiones específicas.',
              'Los coeficientes indican asociación del modelo y no causalidad.', '']
    (out / 'resumen.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'resumen.md')


if __name__ == '__main__':
    main()
