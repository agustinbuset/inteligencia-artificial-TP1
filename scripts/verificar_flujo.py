"""Prueba de integración con datos sintéticos; no representa el dataset real."""
from pathlib import Path
import subprocess
import sys
import tempfile
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scripts.rutas import PROYECTO
from scripts.exportar_dataset_limpio import export_clean_dataset
from scripts.preparar_datos import features


def main():
    rng = np.random.default_rng(42)
    n = 400
    frame = pd.DataFrame({
        'ID': [f'sintetico-{i}' for i in range(n)],
        'Severity': np.tile([1, 2, 3, 4], n // 4),
        'Temperature(F)': rng.normal(60, 15, n),
        'Humidity(%)': rng.uniform(0, 100, n),
        'State': rng.choice(['CA', 'NY', 'TX'], n),
        'Amenity': rng.choice([True, False], n),
        'Country': ['US'] * n,
        'Start_Time': ['2020-01-01 12:00:00'] * n,
    })
    frame.loc[:10, 'Temperature(F)'] = np.nan
    frame.loc[20, 'Temperature(F)'] = 10000
    frame.loc[30, 'Humidity(%)'] = -10
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'sintetico.csv'
        output = Path(tmp) / 'resultado'
        frame.to_csv(source, index=False)
        command = [sys.executable, '-m', 'scripts.preparar_datos',
                   '--input', str(source), '--output', str(output), '--folds', '2',
                   '--comparison-max-rows', '200', '--batch-size', '37']
        subprocess.run(command, check=True, cwd=PROYECTO)
        subprocess.run(command + ['--resume'], check=True, cwd=PROYECTO)
        assert source.read_bytes() == (output / 'dataset_original_copia.csv').read_bytes()
        groups = []
        for name in ['train', 'validation', 'test']:
            raw = pd.read_csv(output / f'{name}_raw.csv')
            matrix = sparse.load_npz(output / f'X_{name}.npz')
            labels = pd.read_csv(output / f'y_{name}.csv')
            assert matrix.shape[0] == len(raw) == len(labels)
            assert np.isfinite(matrix.data).all()
            assert raw['fila_origen'].equals(labels['fila_origen'])
            groups.append(set(raw['ID']))
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert sum(map(len, groups)) == n
        assert len(pd.read_csv(output / 'comparacion.csv')) == 6
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['filas_comparacion_training'] == 200
        assert manifest['filas']['train'] == 280
        assert manifest['original_intacto']
        assert joblib.load(output / 'pipeline.joblib') is not None
        clean_path = export_clean_dataset(output, batch_size=37)
        cleaned = pd.read_csv(clean_path, dtype={'TMC': str})
        assert len(cleaned) == n and not cleaned.isna().any().any()
        assert set(cleaned['conjunto']) == {'train', 'validation', 'test'}
        pipeline = joblib.load(output / 'pipeline.joblib')
        preview = cleaned.loc[cleaned['conjunto'] == 'train'].head(20)
        transformed = pipeline[:-1].transform(features(preview.drop(
            columns=['fila_origen', 'conjunto'])))
        expected = sparse.load_npz(output / 'X_train.npz')[:20]
        assert np.allclose(sparse.csr_matrix(transformed).toarray(), expected.toarray())
    print('OK: seis variantes, copia intacta, particiones disjuntas, matrices y pipeline.')


if __name__ == '__main__':
    main()
