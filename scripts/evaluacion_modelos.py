"""Funciones comunes para evaluar clasificadores con las mismas cuatro clases."""
import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix

CLASES = [1, 2, 3, 4]


def guardar_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda v: v.item() if isinstance(v, np.generic) else str(v)),
                    encoding='utf-8')


def evaluar(modelo, X, y, salida, conjunto):
    pred = modelo.predict(X)
    reporte = classification_report(y, pred, labels=CLASES, output_dict=True, zero_division=0)
    guardar_json(salida / f'metricas_{conjunto}.json', reporte)
    pd.DataFrame(confusion_matrix(y, pred, labels=CLASES),
                 index=pd.Index(CLASES, name='real'),
                 columns=[f'pred_{c}' for c in CLASES]).to_csv(
                     salida / f'matriz_confusion_{conjunto}.csv')
    return dict(accuracy=reporte['accuracy'], f1_macro=reporte['macro avg']['f1-score'],
                balanced_accuracy=reporte['macro avg']['recall'],
                f1_weighted=reporte['weighted avg']['f1-score'])


def comprobar_datos(X, y, nombres):
    values = X.data if sparse.issparse(X) else X
    if X.shape != (len(y), len(nombres)) or not np.isfinite(values).all():
        raise ValueError('Dimensiones incompatibles o matriz con valores no finitos.')
    if set(np.unique(y)) != set(CLASES):
        raise ValueError('Se esperaban las cuatro clases de Severity.')
