# Modelo 2: árbol de decisión

Clasifica `Severity` en cuatro categorías mediante divisiones sucesivas de los
atributos. Puede capturar relaciones no lineales e interacciones y permite
inspeccionar reglas. No incorpora explícitamente el orden de las severidades.

## Protocolo y parámetros

Se reutilizan los 2.082.034 registros de entrenamiento y 446.150 de validación,
con las mismas 172 columnas preparadas que la regresión logística. El escalado
no es necesario para el árbol, pero se conserva para comparar sobre las mismas
entradas. Se convierte la matriz a float32, formato interno del árbol.
No se vuelve a imputar, codificar ni dividir el dataset.

El primer ajuste sin pesos utilizó CSC disperso. Para los pendientes se adoptó
una representación densa de los mismos valores float32, tras medir 23,14 s
en CSC frente a 4,66 s en denso sobre las primeras 100.000 filas, con los mismos
parámetros y cantidad de hojas. La matriz densa de entrenamiento ocupa unos
1,43 GB; se comprobó memoria disponible suficiente. Se conserva el árbol sin
pesos ya ajustado y se registra la representación de cada ajuste. Los tiempos
de las variantes no permiten aislar el efecto de los pesos, dado este cambio.

Se comparan `class_weight=None` y `class_weight='balanced'`, con `criterion='gini'`,
`max_depth=20`, `min_samples_leaf=5`, `max_features=None` y semilla 42.
La profundidad y el mínimo por hoja limitan complejidad y memoria, sin garantizar
ausencia de sobreajuste. La selección es por F1 macro de validación; también se
registra F1 macro de entrenamiento para observar la diferencia entre conjuntos.
Es una comparación acotada del uso de pesos, no una optimización exhaustiva.
Los árboles no tienen un proceso iterativo de convergencia como la regresión logística.

## Ejecutar y consultar resultados

```powershell
.\.venv\Scripts\python.exe -m scripts.entrenar_arboles --metodo arbol_decision
```

Los resultados se guardan en `outputs/modelos/arbol_decision/`, con una carpeta
por variante: modelo, parámetros, avisos, métricas y matrices de confusión de
entrenamiento y validación, importancias y reglas de los primeros niveles.
`experimento.json` registra la selección y `comparacion_validation.csv` permite
comparar variantes. No se accede a prueba durante el ajuste.

Las reglas exportadas usan los valores transformados, incluyendo los números
estandarizados: sus umbrales no están en las unidades originales.
Las importancias por reducción de impureza describen el modelo y pueden favorecer
atributos con muchos posibles cortes; no prueban causalidad.

Para repetir, usar `--salida outputs/modelos_nueva_ejecucion`; se rechaza reemplazar
resultados existentes. El clasificador guardado recibe matrices transformadas,
igual que el modelo logístico; ver su ejemplo de preprocesamiento en
[regresion_logistica.md](regresion_logistica.md).

La evaluación final y sus limitaciones se documentan en
[comparacion_modelos.md](comparacion_modelos.md).

## Resultados de validación

| Variante | Accuracy | F1 macro | Balanced accuracy | F1 macro train | Ajuste |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sin pesos | 79,32 % | 0,4935 | 0,4743 | 0,5293 | 976,7 s (CSC) |
| Balanceada | 65,73 % | 0,4420 | 0,5887 | 0,4675 | 128,3 s (denso) |

Se selecciona **sin pesos** por mayor F1 macro de validación. El uso de pesos
aumentó el recall promedio por clase, pero redujo F1 macro y accuracy. El F1
macro de entrenamiento es mayor que el de validación en ambas variantes;
la diferencia no sustituye un análisis específico de cada clase.
