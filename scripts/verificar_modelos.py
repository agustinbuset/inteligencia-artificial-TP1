"""Prueba de integración pequeña de los cuatro entrenamientos y su comparación."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_classification

from scripts.rutas import PROYECTO


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / 'datos'
        data.mkdir()
        X, y = make_classification(n_samples=800, n_features=8, n_informative=6,
                                   n_redundant=0, n_classes=4, random_state=42)
        for part, selection in [('train', slice(0, 400)), ('validation', slice(400, 600)),
                                ('test', slice(600, 800))]:
            sparse.save_npz(data / f'X_{part}.npz', sparse.csr_matrix(X[selection]))
            pd.DataFrame({'Severity': y[selection] + 1}).to_csv(data / f'y_{part}.csv', index=False)
        pd.DataFrame({'atributo': [f'atributo_{i}' for i in range(8)]}).to_csv(
            data / 'atributos_finales.csv', index=False)
        models = root / 'modelos'
        commands = [
            ['scripts.entrenar_regresion_logistica', '--datos', str(data), '--salida',
             str(models / 'regresion_logistica')],
            ['scripts.entrenar_arboles', '--datos', str(data), '--salida', str(models),
             '--arboles', '3', '--trabajos', '1'],
            ['scripts.entrenar_red_neuronal', '--datos', str(data), '--salida',
             str(models / 'red_neuronal'), '--epocas', '20', '--trabajos', '1'],
            ['scripts.comparar_modelos', '--datos', str(data), '--modelos', str(models),
             '--salida', str(root / 'comparacion')],
        ]
        for command in commands:
            result = subprocess.run([sys.executable, '-m', *command], cwd=PROYECTO,
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        summary = pd.read_csv(root / 'comparacion/comparacion_test.csv')
        assert len(summary) == 4 and summary['modelo'].nunique() == 4
        assert summary[['accuracy', 'f1_macro', 'balanced_accuracy']].apply(
            lambda column: column.between(0, 1).all()).all()
        for method in summary['modelo']:
            report = json.loads((root / 'comparacion' / method / 'metricas_test.json').read_text())
            assert report['macro avg']['support'] == 200
        for command in commands:
            # Repetir no debe reemplazar ninguna salida existente.
            result = subprocess.run([sys.executable, '-m', *command], cwd=PROYECTO,
                                    capture_output=True, text=True)
            assert result.returncode != 0, 'Se permitió reemplazar resultados.'
        result = subprocess.run([sys.executable, '-m', *commands[1], '--reanudar'],
                                cwd=PROYECTO, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        print('Integración correcta: cuatro métodos, evaluación final, métricas y protección de salidas.')


if __name__ == '__main__':
    main()
