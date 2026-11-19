# Modelo 3: Random Forest

Combina árboles de decisión entrenados con muestras bootstrap y subconjuntos
aleatorios de atributos en cada división. Promedia sus probabilidades para
asignar una de las cuatro severidades. Busca reducir la variabilidad de un árbol
individual, sin garantizar que supere a los otros modelos.

## Protocolo y parámetros

Se usan las mismas 172 columnas, 2.082.034 registros de entrenamiento y 446.150
de validación que los otros clasificadores. Se conserva el preprocesamiento
existente, convirtiendo las matrices a representaciones densas float32. La
matriz de entrenamiento ocupa aproximadamente 1,43 GB. Esta representación
aceleró el ajuste de un árbol en una medición previa; no cambia las variables
ni las particiones. Se requiere memoria suficiente para matrices y modelos.

Parámetros fijados antes de evaluar:

- 100 árboles, criterio Gini, profundidad máxima 20 y mínimo de 5 registros por hoja.
- `max_features='sqrt'`: en cada división se considera un subconjunto de atributos.
- `bootstrap=True`, `max_samples=0.25`: cada árbol recibe aproximadamente 520.508
  extracciones con reemplazo del entrenamiento completo. Puede repetir registros;
  no son 520.508 filas necesariamente distintas. El conjunto de validación no se muestrea.
- Semilla 42 y dos trabajos paralelos para limitar el consumo de recursos.
- Dos variantes: sin pesos y `class_weight='balanced'`. En esta última, los pesos
  se calculan con las frecuencias del entrenamiento completo, no por bootstrap.

La selección utiliza F1 macro de validación. Es una comparación acotada de pesos,
no una búsqueda exhaustiva de cantidad de árboles, profundidad o muestreo.
El bootstrap del 25 % controla el costo: el bosque utiliza todo entrenamiento
como población de muestreo, aunque cada árbol observa una fracción con reemplazo.
La mayor complejidad y costo computacional deben valorarse junto con las métricas.

## Ejecutar y consultar resultados

```bash
python -m scripts.entrenar_arboles --metodo random_forest
```

Las salidas están en `outputs/modelos/random_forest/`: selección y comparación
de validación, más carpetas por variante con modelo, parámetros, avisos, métricas
y matrices de entrenamiento y validación, importancias y estructura del bosque.
Se registra F1 macro de entrenamiento para examinar posibles diferencias frente
a validación. Las importancias por impureza no identifican causas de accidentes.

El comando sin `--metodo` ejecuta ambos métodos de árboles. Se puede cambiar
`--arboles` y `--trabajos`; para conservar ejecuciones previas, elegir otra raíz
con `--salida`. El modelo guardado requiere las mismas columnas transformadas
que la regresión logística, no el CSV limpio directamente.

La evaluación final se documenta en [comparacion_modelos.md](comparacion_modelos.md).

## Resultados de validación

| Variante | Accuracy | F1 macro | Balanced accuracy | F1 macro train | Ajuste |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sin pesos | 77,29 % | 0,4503 | 0,4203 | 0,4576 | 45,3 s |
| Balanceada | 66,57 % | 0,4285 | 0,5687 | 0,4490 | 42,1 s |

Se selecciona **sin pesos** por F1 macro de validación. Aunque los pesos
mejoraron el recall promedio por clase, redujeron F1 macro y accuracy.
El bosque no superó al árbol individual en estas configuraciones: esto no
demuestra que cualquier Random Forest sea inferior. La profundidad, la
selección aleatoria de atributos y el muestreo por árbol condicionan el resultado.
