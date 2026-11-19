# TP1 – Machine Learning: Accidente

**UADE – 3.4.096 Inteligencia Artificial**
Docente: Parkinson, Christian

## Integrantes

- Buset, Agustín
- Castro, Martina Morena
- Gantus, Elías
- Toffoletto, Martina Ornella

## El problema

Dataset asignado por número de grupo: **Accidente** (2.974.335 accidentes viales, 49 atributos). Es un problema de **clasificación supervisada multiclase**: predecir la severidad del accidente (`Severity`, clases 1 a 4, donde 1 es la más leve). Hay que identificar el problema y las variables, preprocesar los datos, aplicar al menos tres modelos de Machine Learning (con técnica de selección de datos, evaluación y ventajas/limitaciones) y presentar los resultados (ver `consigna/`).

## Requisitos

- Python 3.9 o superior (resultados obtenidos con Python 3.14.0 y scikit-learn 1.7.2).
- Dependencias: `pip install -r requirements.txt` (pandas, numpy, scipy, scikit-learn, matplotlib, joblib). Se recomienda un entorno virtual (`python3 -m venv .venv`).
- El dataset original no está en el repositorio por su tamaño (1,1 GB). Viene de la cátedra en `Accidente.7z`: descomprimirlo y dejar el archivo como `datasets/Accidente.csv`.

## Cómo correrlo

Desde la carpeta del proyecto (con el dataset completo, todo el proceso tarda unos 20 minutos y carga el CSV en memoria):

```bash
# 1. Preprocesamiento (punto 3): limpia, transforma y separa 70/15/15
python -m scripts.preparar_datos --input datasets/Accidente.csv --comparison-max-rows 20000 --folds 5

# 2. Modelos (punto 4)
python -m scripts.entrenar_regresion_logistica
python -m scripts.entrenar_arboles
python -m scripts.entrenar_red_neuronal
python -m scripts.comparar_modelos
```

Los datos preparados se guardan en `datasets/preparados/accidente/` y los modelos y la comparación en `outputs/`. Para probar rápido con una muestra se puede agregar `--max-rows 20000` al paso 1. Para reanudar una preparación interrumpida, repetir el mismo comando con `--resume`.

## Estructura del proyecto

| Archivo | Contenido |
|---|---|
| `scripts/preparar_datos.py` | Limpieza, transformaciones, selección y partición de datos |
| `scripts/entrenar_regresion_logistica.py` | Regresión logística y variantes de pesos |
| `scripts/entrenar_arboles.py` | Árbol de decisión y Random Forest |
| `scripts/entrenar_red_neuronal.py` | Red neuronal (perceptrón multicapa) entrenada con retro-propagación |
| `scripts/comparar_modelos.py`, `evaluacion_modelos.py` | Selección y evaluación final de los cuatro modelos |
| `scripts/exportar_dataset_limpio.py`, `resumir_resultados.py` | Tabla limpia en CSV e informe de preparación |
| `scripts/verificar_*.py` | Comprobaciones de los datos y de los flujos |
| `preparar_datos.py` | Acceso de compatibilidad con los modelos guardados |
| `docs/` | Análisis, reporte de preprocesamiento y documentación de cada modelo |
| `consigna/` | Enunciado y descripción del dataset |

## Resumen de la solución

| Tarea | Resolución | Dónde |
|---|---|---|
| 1. Tipo de problema | Clasificación supervisada multiclase (ordinal) de `Severity` | `docs/analisis.md` |
| 2. Variables | 49 variables: identificadores, objetivo, ubicación, tiempo, clima y elementos de la vía | `docs/analisis.md` |
| 3. Preprocesamiento | Faltantes (mediana / `Desconocido`), extremos por IQR, variables temporales, codificación, binarización, estandarización, PCA y L1/L2 comparados | `docs/reporte_preprocesamiento_punto_3.md` |
| 4. Modelos | Regresión logística, árbol de decisión, Random Forest y red neuronal (perceptrón multicapa) | `docs/regresion_logistica.md`, `arbol_decision.md`, `random_forest.md`, `red_neuronal.md` |
| 4a. Selección de datos | Partición 70 % entrenamiento, 15 % validación y 15 % prueba; validación cruzada de 5 partes; los parámetros se aprenden solo con entrenamiento | `docs/reporte_preprocesamiento_punto_3.md` |
| 4b. Evaluación | Selección por F1 macro en validación y evaluación final en prueba, con métricas por clase | `docs/comparacion_modelos.md` |
| 4c. Ventajas y limitaciones | Por modelo y en la comparación | `docs/` |
| 5. Resultados | Comparación final y conclusiones | `docs/comparacion_modelos.md` |

## Resultados

Datos preparados: 2.082.034 filas de entrenamiento, 446.150 de validación y 446.151 de prueba, con 172 atributos.

| Modelo | F1 macro validación | Accuracy prueba | F1 macro prueba |
|---|---|---|---|
| Regresión logística | 0,4123 | 74,07 % | 0,4117 |
| **Árbol de decisión** | **0,4934** | **79,34 %** | **0,4942** |
| Random Forest | 0,4503 | 77,29 % | 0,4520 |
| Red neuronal (128, 64) | 0,4797 | 77,92 % | 0,4805 |
| Referencia: siempre severidad 2 | – | 67,02 % | 0,2006 |

El ganador por F1 macro de validación fue el **árbol de decisión sin pesos**; la red neuronal quedó segunda. Ningún modelo seleccionado reconoció la severidad 1: las métricas por clase y las limitaciones están en `docs/comparacion_modelos.md`.
