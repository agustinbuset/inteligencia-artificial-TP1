# Modelo 1: regresión logística

El objetivo es clasificar `Severity` en 1, 2, 3 o 4. Aunque las etiquetas están
ordenadas, este modelo multiclase no incorpora explícitamente ese orden.
Estima probabilidades mediante una combinación lineal de los atributos y una
función softmax; asigna la clase de mayor probabilidad.

## Datos y protocolo

Se reutilizan las matrices dispersas de `datasets/preparados/accidente/`, con
172 atributos ya imputados, codificados y estandarizados. No se vuelve a
escalar ni a dividir el dataset. Los identificadores y la variable objetivo
no forman parte de las entradas.

- Entrenamiento: 2.082.034 registros.
- Validación: 446.150 registros.
- Prueba reservada: 446.151 registros.

Se comparan dos variantes sobre todo entrenamiento: sin pesos y con
`class_weight='balanced'`. Los pesos se calculan únicamente con las etiquetas
de entrenamiento y aumentan la contribución de las clases menos frecuentes;
no crean ejemplos ni aseguran una mejora.

Ambas usan regularización L2, `C=1`, `solver='lbfgs'`, `max_iter=1000`,
`tol=1e-4` y semilla 42. Es una comparación acotada del tratamiento del
desbalance, no una búsqueda exhaustiva de hiperparámetros. Se registra la
convergencia: un ajuste con aviso de falta de convergencia no se selecciona.

La regularización L2 agrega a la función de pérdida una penalización
proporcional a la suma de los cuadrados de los coeficientes. Así, los
coeficientes de mayor magnitud reciben una penalización mayor, lo que ayuda a
limitar el sobreajuste. `lbfgs` es el algoritmo de optimización que busca los
coeficientes que minimizan esa pérdida. El desbalance entre severidades se
aborda por separado mediante los pesos de clase en la variante balanceada.

La selección usa el mayor F1 macro en validación. También se conservan
accuracy, balanced accuracy (promedio del recall de las cuatro clases),
F1 ponderado, métricas por clase y matriz de confusión. F1 macro otorga el
mismo peso a cada clase, algo relevante ante la escasez de severidad 1.

Este experimento no consulta test. El modelo de referencia del preprocesamiento
ya tiene una evaluación histórica en test; esos resultados no se utilizan aquí
para seleccionar pesos. La evaluación posterior de los cuatro modelos se
documenta en [comparacion_modelos.md](comparacion_modelos.md); la regresión
logística seleccionada obtuvo 0,4117 de F1 macro y 74,07 % de accuracy en prueba.

## Ejecución y archivos

Desde la raíz del proyecto:

```bash
python -m scripts.entrenar_regresion_logistica
```

La salida predeterminada es `outputs/modelos/regresion_logistica/`:

| Archivo | Contenido |
| --- | --- |
| `comparacion_validation.csv` | Métricas y tiempo de entrenamiento de ambas variantes |
| `experimento.json` | Datos, versiones, criterio y variante seleccionada |
| `resumen.md` | Tabla de resultados y selección del experimento |
| `sin_pesos/`, `balanceada/` | Resultados separados de cada variante |
| `*/modelo.joblib` | Clasificador ajustado sobre las matrices preparadas |
| `*/ajuste.json` | Parámetros, convergencia y avisos de cada ajuste |
| `*/metricas_validation.json` | Precision, recall, F1 y soporte por clase |
| `*/matriz_confusion_validation.csv` | Filas reales y columnas predichas |
| `*/coeficientes.csv` | Coeficientes por atributo y clase |

El script rechaza reemplazar una salida existente. Para repetir, elegir otra
carpeta mediante `--salida outputs/modelos/regresion_logistica_02`.
`--datos` permite indicar otra carpeta de matrices con el mismo formato.
Los artefactos pesados permanecen en `outputs/`, excluido de Git; el código y
este documento sí quedan versionados.

El archivo `modelo.joblib` recibe las 172 columnas transformadas en su orden
original; no recibe directamente el CSV limpio. Para nuevos registros hay
que aplicar el preprocesamiento guardado en `pipeline.joblib` (todos sus pasos
excepto `modelo`) después de `features` del módulo de preparación. Los
coeficientes describen asociaciones dentro del modelo, no relaciones causales.

Ejemplo para predecir a partir de un DataFrame con columnas del dataset original:

```python
import json
from pathlib import Path
import joblib
from scripts.preparar_datos import features

carpeta = Path('outputs/modelos/regresion_logistica')
seleccion = json.loads((carpeta / 'experimento.json').read_text(encoding='utf-8'))
variante = seleccion['seleccion']['variante']
modelo = joblib.load(carpeta / variante / 'modelo.joblib')
preparacion = joblib.load('datasets/preparados/accidente/pipeline.joblib')[:-1]
# nuevos_accidentes debe contener las entradas originales que requiere features.
# entradas = preparacion.transform(features(nuevos_accidentes))
# predicciones = modelo.predict(entradas)
```

## Limitaciones

Los pesos pueden aumentar el recall minoritario a costa de precisión y aciertos
globales. La clase 1 tiene muy pocos registros: revisar siempre su soporte y
su matriz de confusión. La división aleatoria existente no mide por sí sola
generalización a años futuros o regiones nuevas. Además, la preparación fue
seleccionada previamente con un clasificador logístico de referencia; esa
decisión debe reconocerse al comparar con modelos de árboles.

## Resultados ejecutados

Se entrenaron ambas variantes con todos los registros de entrenamiento y se
evaluaron sobre los 446.150 registros de validación, sin usar test.
Entorno: Python 3.14.0 y scikit-learn 1.7.2.

| Variante | Accuracy | F1 macro | Balanced accuracy | Iteraciones | Tiempo de ajuste |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sin pesos | 74,18 % | 0,4123 | 0,3871 | 266 | 105,2 s |
| Balanceada | 44,59 % | 0,3437 | 0,6039 | 470 | 191,9 s |

Ambas convergieron. Se selecciona **sin pesos** por su mayor F1 macro.
Los tiempos corresponden a esta máquina, con límite de cuatro hilos, y
excluyen carga de archivos y evaluación.

| Severidad | Casos reales | Recall sin pesos | Recall balanceada | Precision balanceada |
| --- | ---: | ---: | ---: | ---: |
| 1 | 146 | 0,00 % | 55,48 % | 0,0755 % |
| 2 | 299.011 | 87,36 % | 34,07 % | 88,04 % |
| 3 | 133.143 | 50,63 % | 63,67 % | 55,89 % |
| 4 | 13.850 | 16,87 % | 88,33 % | 17,11 % |

La variante balanceada recupera más accidentes de severidad 1 y 4, pero
introduce numerosos falsos positivos. Para la clase 1, solo alrededor del
0,0755 % de sus predicciones es correcto. El balanceo automático no mejoró
el F1 macro en este experimento; no significa que otras ponderaciones sean
inútiles. El modelo seleccionado sigue sin reconocer la clase 1 y tiene
bajo recall de la clase 4: es una referencia para comparar, con limitaciones
claras ante las clases minoritarias.

Se recargaron ambos modelos desde disco y se recalcularon las métricas y
matrices de confusión sobre validación: coinciden con las guardadas. También
se verificaron los coeficientes finitos, las dimensiones y la selección por
F1 macro. El registro local está en `outputs/modelos/regresion_logistica/verificacion.json`.
