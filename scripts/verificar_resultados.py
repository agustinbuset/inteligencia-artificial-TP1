"""Comprueba integridad, filas disjuntas y matrices del procesamiento real."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.preparar_datos import sha256
from scripts.rutas import DATOS_PREPARADOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATOS_PREPARADOS)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    original = Path(manifest['origen'])
    assert sha256(original) == manifest['sha256'], 'Cambió el original'
    assert sha256(out / 'dataset_original_copia.csv') == manifest['sha256'], 'Cambió la copia'
    seen = np.zeros(manifest['filas_originales'], dtype=bool)
    counts = {}
    n_features = None
    for name in ['train', 'validation', 'test']:
        raw = pd.read_csv(out / f'{name}_raw.csv', usecols=['fila_origen', 'Severity'])
        labels = pd.read_csv(out / f'y_{name}.csv')
        matrix = sparse.load_npz(out / f'X_{name}.npz')
        positions = raw['fila_origen'].to_numpy()
        assert len(np.unique(positions)) == len(positions), f'Filas repetidas en {name}'
        assert not seen[positions].any(), f'Solapamiento en {name}'
        seen[positions] = True
        assert raw.equals(labels), f'Etiquetas desalineadas en {name}'
        assert len(raw) == matrix.shape[0] == manifest['filas'][name]
        assert np.isfinite(matrix.data).all(), f'Nulos/infinitos en {name}'
        if n_features is not None:
            assert matrix.shape[1] == n_features
        n_features = matrix.shape[1]
        counts[name] = {'filas': len(raw), 'atributos': n_features}
        del matrix
    expected = (manifest['filas_originales'] - manifest['duplicados_eliminados']
                - manifest['objetivos_invalidos_eliminados'])
    # --max-rows permite una muestra global explícita; este verificador exige cobertura completa.
    assert int(seen.sum()) == expected, 'Las particiones no cubren todos los registros válidos'
    assert len(pd.read_csv(out / 'atributos_finales.csv')) == n_features
    report = {'integridad_original_y_copia': True, 'particiones_disjuntas': True,
              'cobertura_completa': True, 'matrices_finitas_y_alineadas': True,
              'conjuntos': counts}
    (out / 'verificacion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
