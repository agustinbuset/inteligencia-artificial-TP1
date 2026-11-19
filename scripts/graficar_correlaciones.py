"""Dibuja el mapa de calor de correlaciones de las variables numéricas (solo entrenamiento).

Lee correlaciones_train.csv, que ya genera preparar_datos.py, y guarda una imagen.
Escala divergente: azul (correlación negativa) - gris (cero) - rojo (positiva).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scripts.rutas import DATOS_PREPARADOS, PROYECTO

AZUL, NEUTRO, ROJO = '#256abf', '#f0efec', '#c93a3a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=DATOS_PREPARADOS)
    parser.add_argument('--salida', type=Path, default=PROYECTO / 'docs/mapa_calor_correlaciones.png')
    args = parser.parse_args()

    corr = pd.read_csv(args.datos / 'correlaciones_train.csv', index_col=0)
    mapa = LinearSegmentedColormap.from_list('divergente', [AZUL, NEUTRO, ROJO])

    fig, ax = plt.subplots(figsize=(10, 8.5))
    imagen = ax.imshow(corr.to_numpy(), cmap=mapa, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            valor = corr.iat[i, j]
            ax.text(j, i, f'{valor:.2f}'.replace('.', ','), ha='center', va='center', fontsize=8,
                    color='white' if abs(valor) > .6 else '#2b2b28')
    ax.set_xticks([x - .5 for x in range(1, len(corr))], minor=True)
    ax.set_yticks([y - .5 for y in range(1, len(corr))], minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='both', length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title('Correlación de Pearson entre variables numéricas (entrenamiento)', pad=14)
    barra = fig.colorbar(imagen, ax=ax, fraction=.046, pad=.03)
    barra.set_label('Coeficiente de correlación')
    barra.outline.set_visible(False)
    fig.tight_layout()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=150)
    print(f'Guardado: {args.salida}')


if __name__ == '__main__':
    main()
