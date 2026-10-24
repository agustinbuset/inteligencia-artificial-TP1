"""Exporta una tabla legible usando la limpieza YA aprendida con entrenamiento.

CSV: Comma-Separated Values, valores separados por comas.
Los números conservan sus unidades y las categorías conservan sus nombres.
Las matrices X_*.npz siguen siendo la versión escalada y codificada para modelos.
"""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from scripts.preparar_datos import features, sha256
from scripts.rutas import DATOS_PREPARADOS


def export_clean_dataset(folder, batch_size=25000):
    """No ajusta parámetros nuevos ni modifica el original o las particiones."""
    folder = Path(folder)
    destination = folder / 'Accidente_limpio_v1.csv'
    temporary = destination.with_suffix('.csv.tmp')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Ya existe la exportación o una exportación parcial: {destination}')
    if batch_size < 1:
        raise ValueError('El tamaño del bloque debe ser positivo.')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(folder / 'pipeline.joblib')
    attributes = pipeline.named_steps['atributos']
    numeric_names = list(attributes.transformers_[0][2])
    categorical_names = list(attributes.transformers_[1][2])
    numeric = attributes.named_transformers_['numericas']
    categorical = attributes.named_transformers_['categoricas']
    expected_features = list(attributes.feature_names_in_)
    seen = np.zeros(manifest['filas_originales'], dtype=bool)
    counts = {}
    columns = None
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        for name in ['train', 'validation', 'test']:
            count = 0
            print(f'Exportando tabla limpia: {name}...', flush=True)
            reader = pd.read_csv(folder / f'{name}_raw.csv', chunksize=batch_size,
                                 dtype={'TMC': str, 'Zipcode': str, 'Number': str})
            for raw in reader:
                positions = raw['fila_origen'].to_numpy()
                if len(np.unique(positions)) != len(positions) or seen[positions].any():
                    raise ValueError('Hay filas de origen repetidas entre las particiones.')
                seen[positions] = True
                cleaned = features(raw.drop(columns='fila_origen'))[expected_features]
                if numeric_names:
                    values = numeric.named_steps['imputar'].transform(cleaned[numeric_names])
                    cleaned[numeric_names] = numeric.named_steps['outliers'].transform(values)
                    if not np.isfinite(cleaned[numeric_names].to_numpy(dtype=float)).all():
                        raise ValueError('La tabla numérica todavía contiene nulos o infinitos.')
                if categorical_names:
                    cleaned[categorical_names] = categorical.named_steps['imputar'].transform(
                        cleaned[categorical_names])
                if cleaned.isna().any().any():
                    raise ValueError('La tabla limpia todavía contiene valores faltantes.')
                cleaned.insert(0, 'Severity', raw['Severity'].to_numpy())
                cleaned.insert(0, 'conjunto', name)
                if 'ID' in raw:
                    cleaned.insert(0, 'ID', raw['ID'].to_numpy())
                cleaned.insert(0, 'fila_origen', positions)
                cleaned.to_csv(stream, index=False, header=columns is None)
                columns = list(cleaned.columns)
                count += len(cleaned)
            if count != manifest['filas'][name]:
                raise ValueError(f'La cantidad de filas no coincide para {name}.')
            counts[name] = count
    expected_count = sum(manifest['filas'].values())
    if int(seen.sum()) != expected_count:
        raise ValueError('La tabla limpia no cubre las particiones.')
    # Renombrar solo cuando la exportación está completa y verificada.
    temporary.rename(destination)
    metadata = {'archivo': destination.name, 'filas': expected_count,
                'columnas': columns, 'filas_por_conjunto': counts,
                'sin_nulos': True, 'numeros_finitos': True,
                'sin_escalado_ni_codificacion': True,
                'parametros_aprendidos_solo_en_training': True,
                'columnas_de_trazabilidad_no_usar_como_predictores': ['fila_origen', 'ID', 'conjunto'],
                'objetivo': 'Severity', 'sha256': sha256(destination)}
    (folder / 'dataset_limpio_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Tabla limpia guardada: {destination} ({expected_count:,} filas)', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--batch-size', type=int, default=25000)
    args = parser.parse_args()
    export_clean_dataset(args.output, args.batch_size)


if __name__ == '__main__':
    main()
