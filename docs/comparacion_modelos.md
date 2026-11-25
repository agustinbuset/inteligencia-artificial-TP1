# Comparación de los cuatro modelos supervisados

Se comparan regresión logística, árbol de decisión, Random Forest y una red
neuronal (perceptrón multicapa) para
clasificar `Severity` en 1, 2, 3 y 4. Las clases tienen orden, pero estos modelos
multiclase no lo incorporan explícitamente en su función objetivo.

## Protocolo común

Se reutilizan las particiones y las 172 columnas existentes: 2.082.034 filas
de entrenamiento, 446.150 de validación y 446.151 de prueba. No se emplean como
entradas los identificadores, las etiquetas ni el indicador de conjunto.
Cada método compara dos variantes elegidas por F1 macro de validación: sin pesos
y con pesos balanceados en los tres primeros, y dos arquitecturas en la red
neuronal (que no tiene `class_weight`; `fit` acepta `sample_weight`, no probado). Después se fija el ganador entre métodos con ese
mismo criterio, antes de leer prueba. No se reentrena con validación ni se
cambian parámetros a partir de prueba.

F1 macro promedia el F1 de cada severidad por igual. Balanced accuracy es el
promedio de recall de las cuatro clases. También se conservan accuracy, F1
ponderado, precision y recall por clase y matrices de confusión. La clase 1 es
extremadamente escasa; la accuracy por sí sola no resume su reconocimiento.
Se añade una referencia que siempre predice la clase mayoritaria de entrenamiento.

## Ejecución y organización

```bash
python -m scripts.entrenar_regresion_logistica
python -m scripts.entrenar_arboles
python -m scripts.entrenar_red_neuronal
python -m scripts.comparar_modelos
```

Los tres primeros comandos son para una ejecución nueva; si sus resultados ya
existen, no deben repetirse en la misma carpeta. Los scripts rechazan reemplazos.
Para recuperar un entrenamiento de árboles interrumpido, usar
`python -m scripts.entrenar_arboles --reanudar`: se reutilizan solo modelos
guardados cuyos parámetros y métricas recalculadas de entrenamiento y validación
coinciden. Los modelos pendientes se entrenan y los resúmenes se regeneran.
No ejecutar la recuperación mientras otro entrenamiento usa la misma salida.

- `outputs/modelos/<metodo>/`: ajustes y evaluación de validación por variante.
- `outputs/comparacion_modelos/seleccion_previa_test.json`: selección fijada antes de prueba.
- `outputs/comparacion_modelos/comparacion_test.csv`: métricas finales de los cuatro seleccionados.
- `outputs/comparacion_modelos/<metodo>/`: métricas y matriz de confusión de prueba.
- `outputs/comparacion_modelos/resumen.md`: comparación y gráfico de F1 macro.
- `outputs/comparacion_modelos/verificacion.json`: comprobaciones sobre modelos recargados.

El comparador vuelve a predecir validación con los modelos cargados desde disco
y exige coincidencia de métricas y matrices guardadas. También comprueba la
procedencia y consistencia de las selecciones. Los artefactos pesados permanecen
fuera de Git; los scripts y estos informes quedan versionados.

La prueba `python -m scripts.verificar_modelos` ejecuta el flujo completo con
datos sintéticos temporales, incluyendo la protección de salidas y recuperación
de modelos. No modifica los datos ni los resultados reales.

## Alcance de la comparación

La división aleatoria evalúa una mezcla temporal y geográfica similar; no prueba
generalización a regiones nuevas o al futuro. El preprocesamiento se eligió antes
con una referencia logística. Además, el proyecto ya tenía métricas históricas
de prueba durante el preprocesamiento: prueba no es un conjunto nunca observado
en todo el proyecto, aunque esta selección no utiliza esos resultados.

En esta comparación inicial los parámetros de cada familia no se optimizan
exhaustivamente; la optimización posterior se describe al final. El bosque usa
bootstrap del 25 % por árbol y todos los modelos comparten las mismas entradas;
la comparación describe estas configuraciones concretas. Los tiempos dependen
de la máquina y no constituyen un benchmark bajo condiciones idénticas de carga.
Los árboles, los bosques y la red neuronal usan matrices densas float32; la
regresión logística, dispersas. La representación se registra por ajuste.

## Resultados ejecutados

Entorno: Python 3.14.0 y scikit-learn 1.7.2. Las variantes seleccionadas fueron
**sin pesos** en los tres primeros métodos y **dos capas** en la red neuronal.
El ganador se fijó con validación antes de
consultar prueba: **árbol de decisión** (comparación inicial; tras la
optimización de hiperparámetros el primero pasa a ser Random Forest, ver abajo).

| Modelo | F1 macro validación | Accuracy prueba | F1 macro prueba | Balanced accuracy prueba |
| --- | ---: | ---: | ---: | ---: |
| Regresión logística | 0,4123 | 74,07 % | 0,4117 | 0,3863 |
| Árbol de decisión | **0,4934** | **79,34 %** | **0,4942** | **0,4744** |
| Random Forest | 0,4503 | 77,29 % | 0,4520 | 0,4211 |
| Red neuronal (128, 64) | 0,4797 | 77,92 % | 0,4805 | 0,4576 |
| Referencia: siempre severidad 2 | — | 67,02 % | 0,2006 | 0,2500 |

El árbol supera la referencia logística en aproximadamente 5,27 puntos
porcentuales de accuracy y 0,0825 de F1 macro en prueba. La red neuronal queda
segunda: mejora a la logística y al bosque, pero no alcanza al árbol. El orden observado en
validación se mantiene en prueba, con resultados próximos para cada modelo.
Esto respalda la elección para estas particiones y configuraciones, sin probar
superioridad universal ni estabilidad ante cambios temporales o geográficos.

### Reconocimiento de cada severidad en prueba

| Severidad | Casos reales | Recall logística | Recall árbol | Recall bosque | Recall red |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 145 | 0,00 % | 0,00 % | 0,00 % | 0,00 % |
| 2 | 299.012 | 87,27 % | 87,37 % | 89,75 % | 87,04 % |
| 3 | 133.143 | 50,45 % | 65,86 % | 54,96 % | 62,12 % |
| 4 | 13.851 | 16,78 % | 36,52 % | 23,74 % | 33,88 % |

El árbol mejora el recall de las clases 3 y 4 frente a los otros tres modelos,
pero sigue omitiendo una proporción importante de la severidad 4. Ninguno de
los seleccionados reconoce los 145 casos de severidad 1. Por lo tanto, el
79,34 % de aciertos no debe describirse como buen reconocimiento de todas las
severidades. Las versiones balanceadas aumentaron el recall promedio en
validación, pero bajaron F1 macro; no se eligieron ni se evaluaron en prueba.

Las importancias del árbol destacan ubicación, lado de la vía y fuente del
reporte. Son asociaciones del modelo; conviene considerar efectos de cobertura
y procedencia antes de interpretarlas como características propias de la gravedad.

### Verificación y entrega

Se recargaron los cuatro modelos seleccionados y se reprodujeron exactamente
las métricas y matrices de confusión de validación. La evaluación final usa
los mismos 446.151 registros de prueba para los cuatro. La prueba de integración
con datos sintéticos pasó, incluida la recuperación de modelos y el rechazo de
salidas existentes. La sintaxis de los scripts y `git diff --check` también pasaron.

La tabla con valores completos está en
`outputs/comparacion_modelos/comparacion_test.csv`. El gráfico y el resumen
local están en esa misma carpeta; los informes por método explican parámetros,
tratamiento del desbalance y archivos de salida.

## Optimización de hiperparámetros (`scripts/optimizar_modelos.py`)

Se conservaron las 172 columnas y las particiones. Cada familia se exploró con
una grilla sobre una muestra fija de 250.000 filas de entrenamiento, puntuando
con las 446.150 filas de validación; la mejor configuración se reentrenó con todo
el entrenamiento y la elección final se fijó por F1 macro de validación antes de
calcular prueba. Para el árbol se añadieron dos variantes podadas por
costo-complejidad (`ccp_alpha` fijo en el script). La prueba ya se había consultado
en etapas anteriores del proyecto.

| Modelo | Configuración elegida | F1 macro validación (inicial → final) | F1 macro prueba (inicial → final) |
| --- | --- | ---: | ---: |
| Regresión logística | C = 10 | 0,4130 → 0,4134 | 0,4120 → 0,4125 |
| Árbol de decisión | profundidad 24, mínimo 20 por hoja | 0,4935 → 0,5057 | 0,4942 → 0,5069 |
| **Random Forest** | profundidad 28, bootstrap 50 %, 20 % de atributos, 100 árboles | 0,4507 → **0,5174** | 0,4515 → **0,5197** |
| Red neuronal | capas (192, 96), alpha 1e-4 | 0,4724 → 0,4836 | 0,4726 → 0,4828 |

Procedencia de las cifras: provienen de la reevaluación de los modelos guardados
y de la optimización en la máquina de Agustín (Python 3.12.14, scikit-learn 1.7.2).
Por eso las referencias «iniciales» difieren en el cuarto decimal de la tabla de
la sección anterior (corrida con Python 3.14.0). Los valores definitivos están en
`outputs/optimizacion_modelos/comparacion_test.csv`.

**Ganador final: Random Forest.** Superó al árbol optimizado en 0,0117 de F1 macro
de validación y también quedó primero en prueba. Una explicación posible, no
comprobada con un experimento específico, es que cada árbol pasó a ver más datos
(50 % en lugar de 25 %), más atributos por división y mayor profundidad, de modo
que cada árbol es más fuerte y el promedio de 100 árboles reduce la variabilidad.
Las cuatro configuraciones finales siguen sin reconocer la severidad 1.
Con 51 columnas (permutación, exploratorio y sin evaluar en prueba) el árbol
igualó el F1 macro de validación del de 172 columnas (0,4937 contra 0,4935).
