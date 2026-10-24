"""Rutas del proyecto, independientes de la carpeta desde la que se ejecute Python."""
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[1]
DATOS_PREPARADOS = PROYECTO / 'datasets' / 'preparados' / 'accidente'
