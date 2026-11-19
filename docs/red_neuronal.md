# Modelo 4: red neuronal (perceptrón multicapa)

Una red neuronal *feed-forward* recibe las 172 columnas preparadas, las pasa por
una o más capas ocultas de neuronas con activación ReLU y termina en una capa de
salida *softmax* con una neurona por severidad. Cada neurona calcula una suma
ponderada de sus entradas más un sesgo y le aplica la función de activación; la
salida *softmax* convierte los valores finales en probabilidades para las clases
1, 2, 3 y 4. Se predice la clase con mayor probabilidad.

## Entrenamiento: retro-propagación del error

Los pesos empiezan con valores aleatorios (semilla 42) y se ajustan por lotes
de 1.024 registros. En cada lote:

1. **Propagación hacia adelante:** se calculan las salidas de cada capa hasta
   obtener las probabilidades de las cuatro severidades.
2. **Error:** se mide la entropía cruzada entre esas probabilidades y la
   severidad real, más una penalización L2 pequeña (`alpha=1e-4`) sobre los pesos.
3. **Retro-propagación:** con la regla de la cadena se calcula el gradiente del
   error respecto de cada peso, empezando por la capa de salida y retrocediendo
   capa por capa hasta la entrada. El término de error de una capa oculta se
   obtiene a partir del de la capa siguiente, multiplicado por la derivada de
   ReLU (1 si la neurona estaba activa, 0 si no).
4. **Actualización:** el optimizador Adam mueve cada peso en sentido contrario a
   su gradiente, con una tasa de aprendizaje inicial de 0,001 que adapta por peso.

Una época es una pasada completa por el entrenamiento. Para no sobreajustar se
usa **corte temprano**: el 10 % del entrenamiento (estratificado) se reserva
internamente y, si su accuracy no mejora durante 5 épocas seguidas, se detiene.
El conjunto de validación del proyecto no participa en esa decisión: se usa
solo para elegir entre arquitecturas, igual que con los otros modelos.

## Protocolo y parámetros

Se usan las mismas 172 columnas, 2.082.034 registros de entrenamiento y 446.150
de validación que los otros clasificadores, en matrices densas float32. La
estandarización del preprocesamiento es importante para este modelo: el descenso
por gradiente converge mal si las entradas tienen escalas muy distintas.

Dos arquitecturas fijadas antes de evaluar:

| Variante | Capas ocultas | Pesos entrenables |
| --- | --- | ---: |
| `una_capa` | 64 neuronas | 11.332 |
| `dos_capas` | 128 y 64 neuronas | 30.660 |

Parámetros comunes: ReLU, Adam, lote de 1.024, `alpha=1e-4`, máximo de 100
épocas, corte temprano con paciencia de 5 y semilla 42. `MLPClassifier` no
admite `class_weight`, así que a diferencia de los otros modelos las variantes
comparan arquitecturas y no ponderación de clases. La selección se hace por F1
macro de validación.

## Ejecutar y consultar resultados

```bash
python -m scripts.entrenar_red_neuronal
```

Los resultados se guardan en `outputs/modelos/red_neuronal/`, con una carpeta por
variante: modelo, parámetros, forma de las matrices de pesos, avisos, métricas y
matrices de confusión de entrenamiento y validación, y la curva de aprendizaje
(`curva_aprendizaje.json`). `curvas_aprendizaje.png` grafica la pérdida por época
y la accuracy en la reserva interna de ambas variantes. `--epocas` cambia el
máximo de épocas y `--salida` permite repetir sin reemplazar resultados.

El modelo recibe las matrices transformadas, igual que los demás; ver el ejemplo
de preprocesamiento en [regresion_logistica.md](regresion_logistica.md).

## Resultados de validación

Entorno: Python 3.14.0 y scikit-learn 1.7.2.

| Variante | Épocas | Accuracy | F1 macro | Balanced accuracy | F1 macro train | Ajuste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Una capa (64) | 50 | 77,12 % | 0,4627 | 0,4371 | 0,4663 | 54,1 s |
| Dos capas (128, 64) | 41 | 77,88 % | **0,4797** | 0,4574 | 0,4892 | 86,0 s |

Ambas variantes cortaron por corte temprano, antes del máximo de 100 épocas. Se
selecciona **dos capas** por mayor F1 macro de validación. La capa extra permite
combinar atributos de forma no lineal y mejora el recall de las severidades 3
(62,15 %) y 4 (33,86 %) frente a una sola capa. La diferencia entre F1 macro de
entrenamiento y de validación es chica (0,0095), sin señales de sobreajuste fuerte.

## Ventajas y limitaciones

**Ventajas**

- Aprende relaciones no lineales e interacciones entre atributos sin tener que
  definirlas a mano. Por eso supera a la regresión logística, que es lineal
  (0,4797 contra 0,4123 de F1 macro en validación).
- Escala bien a millones de registros: entrena por lotes y el costo crece en
  forma lineal con la cantidad de filas.
- La salida *softmax* da probabilidades por clase, útiles para fijar umbrales.

**Limitaciones**

- Es una caja negra: no da reglas legibles como el árbol ni coeficientes por
  atributo como la regresión logística.
- Tiene muchos hiperparámetros (capas, neuronas, tasa de aprendizaje, lote,
  regularización). Acá se probaron solo dos arquitecturas; una búsqueda más
  amplia podría cambiar el resultado.
- Depende de la inicialización aleatoria y del orden de los lotes: otra semilla
  da métricas algo distintas.
- Igual que los demás modelos, no reconoce la severidad 1 (146 casos en
  validación) y tiene bajo recall en la 4. Sin ponderación de clases, el
  gradiente lo dominan las severidades 2 y 3.

La evaluación final en prueba se documenta en [comparacion_modelos.md](comparacion_modelos.md).
