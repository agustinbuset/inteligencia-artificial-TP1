# Comparación de los tres modelos supervisados

Se comparan regresión logística, árbol de decisión y Random Forest para
clasificar `Severity` en 1, 2, 3 y 4. Las clases tienen orden, pero estos modelos
multiclase no lo incorporan explícitamente en su función objetivo.

## Protocolo común

Se reutilizan las particiones y las 172 columnas existentes: 2.082.034 filas
de entrenamiento, 446.150 de validación y 446.151 de prueba. No se emplean como
entradas los identificadores, las etiquetas ni el indicador de conjunto.
Cada método compara dos variantes, sin pesos y con pesos balanceados, elegidas
por F1 macro de validación. Después se fija el ganador entre métodos con ese
mismo criterio, antes de leer prueba. No se reentrena con validación ni se
cambian parámetros a partir de prueba.

F1 macro promedia el F1 de cada severidad por igual. Balanced accuracy es el
promedio de recall de las cuatro clases. También se conservan accuracy, F1
ponderado, precision y recall por clase y matrices de confusión. La clase 1 es
extremadamente escasa; la accuracy por sí sola no resume su reconocimiento.
Se añade una referencia que siempre predice la clase mayoritaria de entrenamiento.

## Ejecución y organización

```powershell
.\.venv\Scripts\python.exe -m scripts.entrenar_regresion_logistica
.\.venv\Scripts\python.exe -m scripts.entrenar_arboles
.\.venv\Scripts\python.exe -m scripts.comparar_modelos
```

Los dos primeros comandos son para una ejecución nueva; si sus resultados ya
existen, no deben repetirse en la misma carpeta. Los scripts rechazan reemplazos.
Para recuperar un entrenamiento de árboles interrumpido, usar
`python -m scripts.entrenar_arboles --reanudar`: se reutilizan solo modelos
guardados cuyos parámetros y métricas recalculadas de entrenamiento y validación
coinciden. Los modelos pendientes se entrenan y los resúmenes se regeneran.
No ejecutar la recuperación mientras otro entrenamiento usa la misma salida.

- `outputs/modelos/<metodo>/`: ajustes y evaluación de validación por variante.
- `outputs/comparacion_modelos/seleccion_previa_test.json`: selección fijada antes de prueba.
- `outputs/comparacion_modelos/comparacion_test.csv`: métricas finales de los tres seleccionados.
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

Los parámetros de cada familia no se optimizan exhaustivamente. El bosque usa
bootstrap del 25 % por árbol y todos los modelos comparten las mismas entradas;
la comparación describe estas configuraciones concretas. Los tiempos dependen
de la máquina y no constituyen un benchmark bajo condiciones idénticas de carga.
El árbol sin pesos se ajustó en CSC disperso; el árbol balanceado y los bosques
usaron matrices densas con los mismos valores float32 para reducir el tiempo.
La representación se registra por ajuste. No atribuir la diferencia de tiempos
entre las variantes del árbol únicamente al uso de pesos.

## Resultados ejecutados

Entorno: Python 3.12.14 y scikit-learn 1.7.2. Las variantes seleccionadas fueron
**sin pesos** en los tres métodos. El ganador se fijó con validación antes de
consultar prueba: **árbol de decisión**.

| Modelo | F1 macro validación | Accuracy prueba | F1 macro prueba | Balanced accuracy prueba |
| --- | ---: | ---: | ---: | ---: |
| Regresión logística | 0,4130 | 74,06 % | 0,4120 | 0,3866 |
| Árbol de decisión | **0,4935** | **79,34 %** | **0,4942** | **0,4742** |
| Random Forest | 0,4507 | 77,25 % | 0,4515 | 0,4207 |
| Referencia: siempre severidad 2 | — | 67,02 % | 0,2006 | 0,2500 |

El árbol supera la referencia logística en aproximadamente 5,28 puntos
porcentuales de accuracy y 0,0822 de F1 macro en prueba. El orden observado en
validación se mantiene en prueba, con resultados próximos para cada modelo.
Esto respalda la elección para estas particiones y configuraciones, sin probar
superioridad universal ni estabilidad ante cambios temporales o geográficos.

### Reconocimiento de cada severidad en prueba

| Severidad | Casos reales | Recall logística | Recall árbol | Recall bosque |
| --- | ---: | ---: | ---: | ---: |
| 1 | 145 | 0,00 % | 0,00 % | 0,00 % |
| 2 | 299.012 | 87,24 % | 87,37 % | 89,79 % |
| 3 | 133.143 | 50,48 % | 65,85 % | 54,76 % |
| 4 | 13.851 | 16,92 % | 36,47 % | 23,72 % |

El árbol mejora el recall de las clases 3 y 4 frente a los otros dos modelos,
pero sigue omitiendo una proporción importante de la severidad 4. Ninguno de
los seleccionados reconoce los 145 casos de severidad 1. Por lo tanto, el
79,34 % de aciertos no debe describirse como buen reconocimiento de todas las
severidades. Las versiones balanceadas aumentaron el recall promedio en
validación, pero bajaron F1 macro; no se eligieron ni se evaluaron en prueba.

Las importancias del árbol destacan ubicación, lado de la vía y fuente del
reporte. Son asociaciones del modelo; conviene considerar efectos de cobertura
y procedencia antes de interpretarlas como características propias de la gravedad.

### Verificación y entrega

Se recargaron los tres modelos seleccionados y se reprodujeron exactamente
las métricas y matrices de confusión de validación. La evaluación final usa
los mismos 446.151 registros de prueba para los tres. La prueba de integración
con datos sintéticos pasó, incluida la recuperación de modelos y el rechazo de
salidas existentes. La sintaxis de los scripts y `git diff --check` también pasaron.

La tabla con valores completos está en
`outputs/comparacion_modelos/comparacion_test.csv`. El gráfico y el resumen
local están en esa misma carpeta; los informes por método explican parámetros,
tratamiento del desbalance y archivos de salida.
