# Trabajo práctico: preparación del dataset Accidente

Se prepararon **2.974.335 registros**, conservando intacto `datasets/Accidente.csv`.
Los resultados están en `datasets/preparados/accidente/`.

## ¿Dónde está el dataset limpio?

**[Accidente_limpio.csv](datasets/Accidente_limpio_v1.csv)** es la
tabla limpia más fácil de consultar. CSV significa *Comma-Separated Values*:
valores separados por comas.

Contiene los faltantes tratados, los extremos recortados según las reglas
aprendidas y las variables temporales derivadas. Los números conservan sus
unidades y las categorías conservan sus nombres: esta tabla no está escalada.

Incluye `Severity` como objetivo y tres columnas de identificación:
`fila_origen`, `ID` (identificador) y `conjunto`. **Estas tres columnas no se usan como predictores.**
`conjunto` indica `train` (entrenamiento), `validation` (validación) o `test` (prueba).
La tabla reúne las particiones en ese orden; `fila_origen` indica la posición
que tenía cada accidente en el original.

Para consultar algunas filas sin cargar toda la tabla:

```python
import pandas as pd

datos = pd.read_csv('datasets/Accidente_limpio_v1.csv', nrows=10)
print(datos.head())
```

La versión **lista para entrenar**, con escalado y categorías convertidas a
números, está en archivos separados:

| Archivo | Contenido |
| --- | --- |
| `X_train.npz` / `y_train.csv` | Atributos / severidad de entrenamiento: 2.082.034 filas |
| `X_validation.npz` / `y_validation.csv` | Atributos / severidad de validación: 446.150 filas |
| `X_test.npz` / `y_test.csv` | Atributos / severidad de prueba: 446.151 filas |
| `atributos_finales.csv` | Nombres y orden de las 172 columnas preparadas |
| `dataset_original_copia.csv` | Copia exacta del original; no es la tabla limpia |
| `*_raw.csv` | Particiones anteriores a la limpieza y las transformaciones |

Los archivos `.npz` guardan matrices comprimidas. Para leerlos:

```python
from scipy.sparse import load_npz
import pandas as pd

carpeta = 'datasets/preparados/accidente/'
X_train = load_npz(carpeta + 'X_train.npz')
y_train = pd.read_csv(carpeta + 'y_train.csv')['Severity']
nombres = pd.read_csv(carpeta + 'atributos_finales.csv')['atributo']
print(pd.DataFrame(X_train[:5].toarray(), columns=nombres))
```

## Modelos supervisados (punto 4)

Se utilizan **regresión logística, árbol de decisión y Random Forest**.
Cada modelo compara variantes sin pesos y con pesos balanceados. La selección
usa F1 macro de validación; después se realiza la evaluación final en prueba.

| Documentación | Resultados locales |
| --- | --- |
| [Regresión logística](docs/regresion_logistica.md) | `outputs/modelos/regresion_logistica/` |
| [Árbol de decisión](docs/arbol_decision.md) | `outputs/modelos/arbol_decision/` |
| [Random Forest](docs/random_forest.md) | `outputs/modelos/random_forest/` |
| [Comparación y protocolo](docs/comparacion_modelos.md) | `outputs/comparacion_modelos/` |

El ganador por F1 macro de validación fue el **árbol de decisión sin pesos**.
En prueba obtuvo **79,34 % de accuracy y 0,4942 de F1 macro**, frente a 0,4515
del bosque y 0,4120 de la regresión logística. Ningún seleccionado reconoció
la severidad 1: ver las métricas por clase y limitaciones en la comparación.

## Organización

```text
datasets/
  Accidente.csv                 Original sin modificar
  preparados/
    accidente/
      Accidente_limpio.csv      Tabla limpia en unidades comprensibles
      X_*.npz, y_*.csv          Datos preparados para entrenar y evaluar
      resumen.md                Informe con decisiones y resultados
      ...                       Modelos, gráficos y comprobaciones
scripts/
  preparar_datos.py             Limpieza, transformaciones y comparación
  exportar_dataset_limpio.py    Genera la tabla limpia sin volver a entrenar
  resumir_resultados.py         Genera el informe y documenta los parámetros
  verificar_resultados.py       Comprueba los datos reales preparados
  verificar_flujo.py            Prueba pequeña con datos sintéticos
  entrenar_regresion_logistica.py  Primer modelo y variantes de pesos
  entrenar_arboles.py           Árbol de decisión y Random Forest
  evaluacion_modelos.py         Métricas comunes de clasificación
  comparar_modelos.py           Selección y evaluación final de los tres modelos
  verificar_modelos.py          Prueba del entrenamiento y comparación con datos sintéticos
  rutas.py                     Ubicaciones del proyecto
docs/                          Consigna y descripción del dataset
outputs/                       Modelos y comparación (artefactos locales, fuera de Git)
preparar_datos.py               Acceso principal y compatibilidad con modelos anteriores
requirements.txt               Dependencias de Python
```

## Ejecutar en PyCharm

Seleccionar `.venv/Scripts/python.exe` como intérprete. Ejecutar los comandos
desde la raíz del proyecto, donde está este archivo:

```powershell
python -m pip install -r requirements.txt
```

Los datos actuales ya están preparados. Para volver a ejecutar todo, elegir
otra carpeta de salida y evitar reemplazar los resultados actuales:

```powershell
python -m scripts.preparar_datos --input datasets/Accidente.csv --output datasets/preparados/nueva_ejecucion --comparison-max-rows 20000 --folds 5
python -m scripts.exportar_dataset_limpio --output datasets/preparados/nueva_ejecucion
```

La comparación utiliza una muestra estratificada de 20.000 registros de
entrenamiento; el ajuste final usa todo entrenamiento. Las particiones finales
conservan todos los registros válidos. El procesamiento completo carga el archivo
en memoria y puede tardar varios minutos. Para probar con una muestra global se
puede agregar `--max-rows 20000`; esa opción sí reduce las particiones finales.

Para reanudar una preparación interrumpida, repetir exactamente su comando con
`--resume`. Las rutas predeterminadas apuntan a `datasets/preparados/accidente/`.
En una configuración de PyCharm se puede usar el nombre de módulo
`scripts.preparar_datos` o el acceso principal `preparar_datos.py` de la raíz.

## Proceso, explicado brevemente

1. **Explorar y limpiar:** el EDA (*Exploratory Data Analysis*, análisis exploratorio
   de datos) revisa faltantes, distribuciones y extremos. Los números faltantes se
   completan con la mediana y las categorías con `Desconocido`.
2. **Tratar extremos:** se usan límites IQR (*Interquartile Range*, rango
   intercuartílico). Cuando el rango es cero no se recorta esa variable. Un valor
   extremo estadístico no es necesariamente un error de registro.
3. **Transformar:** se comparan estandarización, escalado MinMax, normalización,
   codificación de categorías y discretización. Son alternativas para comparar.
4. **Seleccionar o reducir:** se eliminan identificadores como entradas y columnas
   constantes. Se prueba PCA (*Principal Component Analysis*, análisis de
   componentes principales) sobre los números y regularización L1 y L2, que
   penalizan los coeficientes del modelo. L1 puede anular algunos coeficientes.
5. **Separar y evaluar:** 70 % entrenamiento, 15 % validación y 15 % prueba.
   La validación cruzada usa cinco partes de la muestra de entrenamiento.

Los parámetros se aprenden solamente en entrenamiento; dentro de la validación
cruzada se vuelven a aprender en cada parte de entrenamiento. Así se evita que
la información reservada para evaluar influya en las transformaciones.

La alternativa final fue estandarización. L1 no convergió y quedó como comparación
exploratoria. El clasificador de referencia tiene dificultades con las clases
minoritarias. Ver los resultados por clase y las limitaciones en el
[informe completo](datasets/preparados/accidente/resumen.md).

## Comprobaciones e informe

```powershell
python -m scripts.verificar_flujo
python -m scripts.verificar_resultados
python -m scripts.resumir_resultados --advertencia-l1-cv
```

La primera comprobación usa datos sintéticos temporales. La segunda comprueba
el original y su copia, que las particiones sean disjuntas y que matrices y
etiquetas coincidan. La huella digital SHA-256 (*Secure Hash Algorithm de 256 bits*,
algoritmo de hash seguro de 256 bits) permite detectar cambios en los archivos.

Para generar únicamente la tabla limpia, sin reentrenar, ejecutar
`python -m scripts.exportar_dataset_limpio`; se rechaza reemplazar una tabla ya existente.

Para reutilizar el modelo con nuevos accidentes:

```python
import joblib
from scripts.preparar_datos import features

modelo = joblib.load('datasets/preparados/accidente/pipeline.joblib')
# predicciones = modelo.predict(features(nuevos_accidentes))
```
