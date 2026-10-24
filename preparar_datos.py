"""Acceso principal y compatibilidad con los modelos guardados anteriormente.

El código de preparación está organizado en scripts/preparar_datos.py.
"""
from scripts.preparar_datos import (
    IQRClipper, SafeL1Selector, SparseToDense,
    clean, features, main, sha256, stratified_sample,
)

if __name__ == '__main__':
    main()
