# Reporte de preprocesamiento del dataset Accidente

**Guía para responder el punto 3 del trabajo práctico de Inteligencia Artificial.**

Este documento describe lo que efectivamente se ejecutó, explica las decisiones
y distingue las técnicas incorporadas al resultado final de las que solamente
se compararon. Las cifras provienen de los archivos guardados por los scripts.
No se volvieron a entrenar modelos para escribir este reporte.

## 1. Qué pide la consigna y cuál es el alcance de esta guía

El punto 3 de la [consigna](<../consigna/consigna - TP 1 - MachineLearning.pdf>) solicita
realizar preprocesamiento cuando sea necesario y menciona limpieza,
normalización, categorización, remuestreo y binarización, entre otras posibilidades.
La expresión «en caso de que fuese necesario» permite fundamentar cuáles son
adecuadas para el problema; no significa aplicar todas sin evaluar su efecto.

En este proyecto el problema es **clasificar la severidad de un accidente ya
registrado**. La columna objetivo es `Severity`, con clases 1, 2, 3 y 4. Los
números representan categorías de severidad; no se estandarizaron como si fueran
una magnitud física ni se incorporaron a las entradas del modelo.

La regresión logística utilizada fue un modelo de referencia para comparar
preparaciones. **Seis variantes de preprocesamiento no equivalen a seis modelos
distintos:** este reporte no resuelve por sí solo el requisito del punto 4 de
aplicar al menos tres modelos diferentes.

## 2. Datos de entrada y archivos de salida

El archivo original tiene **2.974.335 filas y 49 columnas**. Cada fila representa
un accidente. Se conservó el original y se procesó una copia; los controles de
integridad confirmaron que ambos archivos originales tienen el mismo contenido.

CSV significa *Comma-Separated Values*, valores separados por comas.

| Archivo | Qué contiene y para qué sirve |
| --- | --- |
| [Accidente.csv](../datasets/Accidente.csv) | Dataset original, conservado sin modificar. |
| [Accidente_limpio_v1.csv](../datasets/Accidente_limpio_v1.csv) | Tabla limpia, legible, sin escalado ni codificación numérica de categorías. Es la versión que se renombró y movió a `datasets/`. |
| `datasets/preparados/accidente/dataset_original_copia.csv` | Copia exacta del original, anterior a las transformaciones. No es el dataset limpio. |
| `datasets/preparados/accidente/train_raw.csv`, `validation_raw.csv`, `test_raw.csv` | Particiones anteriores a la limpieza estadística. |
| `datasets/preparados/accidente/X_*.npz` | Matrices numéricas finales: atributos imputados, escalados y codificados. |
| `datasets/preparados/accidente/y_*.csv` | Etiquetas `Severity`, alineadas con las filas de cada matriz. |

La huella digital de `Accidente_limpio_v1.csv` coincide con la registrada al
exportar la tabla: el cambio de nombre y ubicación no cambió su contenido.

### Por qué la tabla limpia y las matrices tienen diferentes columnas

La tabla limpia contiene **44 columnas**:

- **40 atributos** de entrada, todavía en unidades originales y con categorías legibles.
- **1 objetivo:** `Severity`.
- **3 columnas de trazabilidad:** `ID` (identificador), `fila_origen` y `conjunto`.

Las tres columnas de trazabilidad **no deben usarse como predictores**.
`conjunto` identifica entrenamiento (`train`), validación (`validation`) o prueba
(`test`). La tabla reúne las particiones en ese orden, sin ordenar nuevamente
por la posición original.

Las matrices finales tienen **172 columnas predictoras**. La cantidad crece
porque una variable categórica puede convertirse en varias columnas binarias.
Por lo tanto, **172 columnas no es una reducción neta respecto de las 49 columnas
originales**: es el resultado de seleccionar entradas, generar atributos y
codificar categorías. Los nombres y el orden están en
[atributos_finales.csv](../datasets/preparados/accidente/atributos_finales.csv).

## 3. Orden de trabajo y prevención de fuga de información

El flujo seguido fue:

1. Crear y verificar la copia del original.
2. Revisar duplicados y etiquetas objetivo inválidas.
3. Separar entrenamiento, validación y prueba.
4. Explorar entrenamiento y aplicar reglas fijas de limpieza.
5. Comparar distintas preparaciones usando solamente los conjuntos permitidos.
6. Elegir una alternativa y volver a ajustarla con todo entrenamiento.
7. Evaluar una vez sobre prueba y exportar las matrices.
8. Exportar la tabla limpia legible reutilizando los parámetros ya aprendidos.

**Fuga de información** significa usar durante el aprendizaje información que
debería estar reservada para evaluar el modelo, o información que no estará
disponible cuando se haga una predicción real.

Por ejemplo, sería incorrecto calcular la mediana de temperatura con todos los
datos y después separar prueba: los registros de prueba ya habrían influido en
la preparación. Aquí las medianas, límites de recorte, escalas y categorías se
aprendieron con entrenamiento. Validación y prueba recibieron esos mismos
parámetros sin volver a calcularlos.

## 4. Exploración y limpieza de datos

El **EDA (*Exploratory Data Analysis*, análisis exploratorio de datos)** incluyó
tipos y frecuencias, estadísticas descriptivas, faltantes, distribución de
severidad, histogramas, correlaciones y detección de extremos.

Se hizo sobre **todo el entrenamiento: 2.082.034 registros**. Las estadísticas y
los gráficos exploratorios no utilizaron el conjunto de prueba.

### 4.1. Duplicados y objetivo

Se verificaron duplicados exactos y valores de `Severity` ausentes o fuera de
las clases permitidas. El resultado registrado fue:

| Control | Resultado |
| --- | ---: |
| Duplicados exactos eliminados | 0 |
| Objetivos ausentes o inválidos eliminados | 0 |

El código también comprueba que no haya identificadores repetidos con registros
distintos, para evitar que un mismo evento termine en conjuntos diferentes.

**No se borraron filas por tener datos meteorológicos faltantes ni por ser
extremas.** Se conservaron los 2.974.335 registros.

### 4.2. Reglas básicas de validez

Antes de imputar, se trataron cadenas vacías como faltantes, se quitaron espacios
en las categorías y se reemplazaron valores infinitos por faltantes.

Se aplicaron estas reglas fijas:

| Atributos | Rango admitido por el código |
| --- | --- |
| `Start_Lat` | De −90 a 90. |
| `Start_Lng` | De −180 a 180. |
| `Humidity(%)` | De 0 a 100. |
| `Wind_Speed(mph)`, `Visibility(mi)`, `Precipitation(in)`, `Pressure(in)` | Valores no negativos. |

Los valores que no cumplen estos rangos pasan a faltantes. Son reglas básicas,
no una validación meteorológica exhaustiva. Por ejemplo, admitir presión igual
a cero no alcanza para garantizar validez física; ese valor también puede ser
afectado posteriormente por el recorte estadístico.

Las unidades de los nombres son: `F`, grados Fahrenheit; `mi`, millas; `mph`,
millas por hora; `in`, pulgadas; y `%`, porcentaje. `Wind_Chill(F)` es sensación
térmica asociada al viento, no velocidad del viento.

### 4.3. Tratamiento de valores faltantes

Los principales porcentajes de faltantes en entrenamiento, después de las reglas
fijas y antes de imputar, fueron los siguientes. `TMC` significa *Traffic Message
Channel*, código de mensaje de tránsito, y se trata como categoría.

| Atributo | Faltantes |
| --- | ---: |
| `Precipitation(in)` | 67,15 % |
| `Wind_Chill(F)` | 62,26 % |
| `TMC` | 24,48 % |
| `Wind_Speed(mph)` | 14,83 % |
| `Weather_Condition` | 2,22 % |
| `Visibility(mi)` | 2,20 % |
| `Humidity(%)` | 1,99 % |
| `Temperature(F)` | 1,88 % |
| `Pressure(in)` | 1,62 % |
| `Wind_Direction` | 1,52 % |

**Atributos numéricos:** se imputaron con la mediana. La mediana es el valor
central de los datos ordenados y es menos sensible a extremos que la media.
Algunos valores aprendidos con todo entrenamiento fueron:

| Atributo | Mediana usada |
| --- | ---: |
| `Temperature(F)` | 64,4 |
| `Wind_Chill(F)` | 54,0 |
| `Humidity(%)` | 68,0 |
| `Pressure(in)` | 29,98 |
| `Visibility(mi)` | 10,0 |
| `Wind_Speed(mph)` | 7,0 |
| `Precipitation(in)` | 0,0 |

**Atributos categóricos:** se imputaron con la categoría `Desconocido`.
Así se conservaron las filas sin inventar una categoría conocida.

La precipitación faltante terminó en cero **porque la mediana aprendida fue
cero**, no porque se haya demostrado que ausencia de medición significa ausencia
de lluvia. Esta diferencia debe explicarse al justificar la decisión.

No se eliminaron automáticamente las columnas con muchos faltantes. Se optó por
conservarlas e imputarlas como punto de partida. Esto tiene una limitación:
imputar más del 60 % de una columna puede distorsionar su distribución. No se
compararon otros imputadores ni se añadieron indicadores binarios de ausencia.

### 4.4. Tratamiento de valores extremos

Un **outlier** es un valor que se aparta mucho de la distribución habitual.
No necesariamente es un error: puede representar un fenómeno real poco frecuente.

Se utilizó el **IQR (*Interquartile Range*, rango intercuartílico)**:

```text
Q1 = percentil 25
Q3 = percentil 75
IQR = Q3 − Q1
límite inferior = Q1 − 1,5 × IQR
límite superior = Q3 + 1,5 × IQR
```

Los valores fuera de los límites se **recortaron al límite correspondiente**;
no se eliminaron los accidentes. Primero se imputó y después se aprendieron los
límites con entrenamiento. Algunos límites finales fueron:

| Atributo | Límite inferior | Límite superior |
| --- | ---: | ---: |
| `Temperature(F)` | 13,65 | 113,25 |
| `Pressure(in)` | 29,385 | 30,545 |
| `Wind_Speed(mph)` | −1,1 | 17,3 |

El límite negativo de viento no implica introducir velocidades negativas:
los valores negativos ya habían pasado a faltantes por la regla de dominio.

**Caso especial:** cuando el IQR es cero, el código no recorta esa variable.
Esto ocurrió en el ajuste final de `Wind_Chill(F)`, `Visibility(mi)` y
`Precipitation(in)`. En sensación térmica, la gran cantidad de valores imputados
en la mediana hizo que los cuartiles centrales coincidieran.

Hay que distinguir dos archivos:

- [outliers_iqr_train.csv](../datasets/preparados/accidente/outliers_iqr_train.csv)
  cuenta observaciones señaladas durante el EDA, **antes de imputar**.
- [parametros_numericos_finales.csv](../datasets/preparados/accidente/parametros_numericos_finales.csv)
  registra las medianas y límites **realmente aplicados**, aprendidos después
  de imputar. Los límites infinitos significan «sin recorte».

Por eso, los conteos del primer archivo **no equivalen a cantidades de valores
modificados**. Por ejemplo, con cuartiles de precipitación iguales a cero, el EDA
señala lluvias positivas como extremos, pero el tratamiento final no las recorta.

Recortar extremos puede atenuar condiciones meteorológicas reales asociadas a
accidentes graves. La regla implementada es una decisión inicial que conviene
revisar por atributo; no demuestra que todos esos valores fueran incorrectos.

## 5. Selección de entradas y generación de atributos

### 5.1. Variables excluidas como entradas

| Variables | Decisión y motivo |
| --- | --- |
| `ID` | Identifica el evento; se conservó para trazabilidad, no como predictor. |
| `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)` | Se excluyeron por su posible relación con información posterior al inicio o con consecuencias del accidente. |
| `Description` | Texto libre que puede revelar gravedad o consecuencias; no se implementó procesamiento de texto. |
| `Number`, `Street`, `Zipcode` | Se excluyeron para limitar la complejidad de las direcciones y los códigos de alta cardinalidad. No se demostró que carezcan de toda información predictiva. |
| `Weather_Timestamp` | No se usó como predictor ni se construyó un desfase temporal meteorológico. |
| `Start_Time` original | Se reemplazó por atributos temporales más simples. |

**Cardinalidad** es la cantidad de valores distintos de una variable. Por ejemplo,
se observaron 11.173 ciudades distintas en entrenamiento: asignar una columna a
cada una puede producir una representación muy grande.

### 5.2. Variables temporales derivadas

De `Start_Time` se obtuvieron:

- `Start_Hour`: hora del día, de 0 a 23.
- `Start_Weekday`: día semanal, de 0 a 6; lunes es 0 y domingo es 6.
- `Start_Month`: mes, de 1 a 12.

No se calculó duración con `End_Time`, porque sería información posterior al
inicio. Tampoco se aplicó codificación cíclica: el modelo recibe números para
hora, día y mes, aunque estos tienen una naturaleza circular. Es una posible
mejora, por ejemplo para representar la proximidad entre las horas 23 y 0.

### 5.3. Codificación de categorías y binarización

`TMC` se trató como categoría aunque aparezca como número. Un código no debe interpretarse
automáticamente como una cantidad.

Se aplicó **one-hot encoding**, una codificación que crea columnas indicadoras
con valores 0 o 1. Por ejemplo, una variable con las categorías `Day` y `Night`
puede convertirse en dos columnas, cada una indicando la presencia de una
categoría. Los indicadores de la vía, como `Amenity`, también pasaron por esta
codificación categórica.

Esto cubre la **binarización de categorías** mencionada en la consigna. No debe
confundirse con convertir las cuatro clases objetivo en solo dos clases: esa
conversión no se hizo.

Para limitar el tamaño de la representación, se agruparon categorías con
frecuencia menor al 1 % y se fijó un máximo de 30 categorías de salida por
variable, incluyendo la agrupación cuando corresponde. Se configuró el
tratamiento de categorías desconocidas al transformar validación y prueba.
Estos parámetros controlan el costo, pero pueden perder detalle geográfico;
no se realizó una búsqueda exhaustiva para demostrar que fueran óptimos.

### 5.4. Eliminación de constantes y relevancia

Se eliminaron columnas codificadas sin variación mediante `VarianceThreshold`.
En el ajuste final se retiraron:

- `categoricas__Country_US`: todos los registros de entrenamiento tenían `US`.
- `categoricas__Turning_Loop_False`: todos tenían `False`.

Una columna constante no permite distinguir un accidente de otro dentro del
entrenamiento. **La tabla legible conserva `Country` y `Turning_Loop`; la
eliminación se aplicó a las matrices para el modelo.**

También se guardó un ranking por magnitud media absoluta de los coeficientes
del clasificador en [relevancia.csv](../datasets/preparados/accidente/relevancia.csv).
Entre las primeras entradas aparecen estados, la fuente del reporte y códigos
de tránsito. Este ranking muestra asociaciones aprendidas, **no causas de la
severidad**, y no se usó para aplicar un filtro adicional en la preparación final.

## 6. Transformaciones comparadas y elección final

Las seis alternativas compartieron la misma base: limpieza, imputación, recorte
cuando correspondía, codificación de categorías y eliminación de constantes.

### 6.1. Qué hace cada alternativa

| Alternativa | Qué se hizo | ¿Quedó en las matrices finales? |
| --- | --- | --- |
| Estandarización | A cada número se le restó la media de entrenamiento y se lo dividió por su desvío estándar. | **Sí.** Fue la alternativa elegida. |
| Escalado MinMax | Se transformaron los números al rango 0–1 de entrenamiento. Datos nuevos pueden quedar fuera de ese rango. | No; se evaluó como alternativa. |
| Normalización por fila | Después de estandarizar números y codificar categorías, se dividió cada fila por su longitud euclídea, o norma L2. | No; se evaluó como alternativa. |
| Discretización | Se agruparon números en hasta cinco intervalos por cuantiles y se representaron con códigos ordinales. Los valores repetidos pueden reducir el número efectivo de intervalos. | No; se evaluó como alternativa. |
| Selección con regularización L1 | Un modelo auxiliar penalizó coeficientes y permitió descartar atributos con coeficientes prácticamente nulos. | No; el ajuste no convergió. |
| PCA (*Principal Component Analysis*, análisis de componentes principales) | Se combinaron los atributos numéricos estandarizados en componentes que conservasen al menos el 95 % de su varianza. Las categorías quedaron fuera de esta reducción. | No; se evaluó como alternativa. |

Estandarizar **no transforma necesariamente una distribución en una distribución
normal**: cambia su centro y escala. La normalización por fila, en cambio,
modifica la magnitud del vector de atributos de cada accidente. Son operaciones
diferentes y no se aplicaron todas en cadena.

### 6.2. Regularización: qué se utilizó realmente

La regularización penaliza coeficientes grandes para controlar la complejidad
del modelo:

- **L1** puede llevar coeficientes a cero, por eso se probó para seleccionar
  atributos. El ajuste sobre la muestra conservó 97 columnas codificadas, pero
  llegó a 2.000 iteraciones sin converger. También hubo avisos durante la
  validación cruzada. Ese resultado quedó como exploratorio.
- **L2** reduce la magnitud de los coeficientes, sin ser un filtro que elimine
  automáticamente columnas. Se utilizó en los clasificadores de referencia y
  en el clasificador final, incluso en la variante que tenía un selector L1 previo.

**Converger** significa que el procedimiento de optimización cumple su criterio
de estabilización. Alcanzar el límite de iteraciones sin converger impide tratar
ese ajuste como una solución suficientemente estabilizada. El modelo final sí
convergió: utilizó 258 de las 1.000 iteraciones permitidas.

### 6.3. Resultado de PCA

En el ajuste sobre la muestra, los 12 atributos numéricos se representaron con
**11 componentes que explicaron aproximadamente el 98,06 % de la varianza**.
Se superó el objetivo del 95 % porque se conserva una cantidad entera de
componentes. La reducción fue pequeña y las componentes son menos fáciles de
interpretar que las variables originales.

PCA se ejecutó en la comparación sobre la muestra y dentro de sus particiones
de validación cruzada. **No se proyectaron en PCA todos los registros del
dataset final.** Las cargas y proporciones de varianza quedaron guardadas en
`cargas_pca.csv` y `varianza_pca.csv`.

### 6.4. Resultados usados para elegir

Se usó **F1 macro** para comparar. F1 combina precisión y recuperación:

- **Precisión:** de los registros predichos como una clase, qué proporción era correcta.
- **Recuperación o recall:** de los registros que realmente eran de una clase,
  qué proporción se identificó correctamente.
- **Macro:** se promedian los F1 de las cuatro clases con el mismo peso.

F1 toma valores de 0 a 1; no es un porcentaje de aciertos.

| Alternativa | F1 macro medio de validación cruzada | Desvío entre particiones | F1 macro en validación |
| --- | ---: | ---: | ---: |
| Estandarización | 0,4224 | 0,0168 | 0,4161 |
| MinMax | 0,4216 | 0,0169 | 0,4140 |
| Normalización por fila | 0,3701 | 0,0102 | 0,3752 |
| Discretización | 0,4168 | 0,0174 | 0,4151 |
| Selección L1 | 0,4164 | 0,0156 | 0,4164 |
| PCA | 0,4211 | 0,0170 | 0,4156 |

Se eligió **estandarización por el mayor F1 macro de validación entre los
ajustes convergentes**. L1 obtuvo un número apenas mayor en validación, pero
quedó excluida por falta de convergencia.

Las diferencias entre estandarización, MinMax, discretización y PCA son pequeñas.
No se realizó una prueba de significación estadística, por lo que no corresponde
afirmar una superioridad general de la estandarización. La elección vale para
esta comparación y este modelo de referencia; otro modelo podría preferir otra
preparación.

## 7. Partición y validación cruzada

La separación fue aleatoria y **estratificada**: se procuró mantener las
proporciones de severidad. La semilla aleatoria fue 42 para poder reproducirla.

| Conjunto | Porcentaje aproximado | Registros | Uso |
| --- | ---: | ---: | --- |
| Entrenamiento | 70 % | 2.082.034 | Aprender transformaciones y ajustar el modelo. |
| Validación | 15 % | 446.150 | Reservar datos para elegir alternativas. |
| Prueba | 15 % | 446.151 | Evaluar el resultado final sin usarlo para elegir. |

La **CV (*Cross-Validation*, validación cruzada)** se ejecutó sobre una muestra
estratificada de **20.000 registros del entrenamiento**, no sobre los casi tres
millones. Se usaron cinco partes: en cada ronda se aprendió con unas 16.000 filas
y se evaluó con las 4.000 restantes, alternando la parte reservada.

En cada ronda se volvieron a aprender imputación, recorte, escalado, codificación
y selección. La comparación con el conjunto de validación también utilizó una
muestra estratificada de 20.000 filas de ese conjunto. El muestreo redujo el costo
de comparar seis alternativas.

**Después de elegir, se volvió a ajustar la alternativa con los 2.082.034
registros de entrenamiento y se evaluó sobre los 446.151 de prueba.** Los
conjuntos finales no se redujeron a las muestras.

La muestra para validación cruzada contiene solamente **6 ejemplos de severidad
1**. En cada parte reservada quedan aproximadamente uno o dos: la evaluación de
esa clase es muy inestable. Además, el archivo histórico de comparación no
registró el estado de convergencia por cada parte; ese campo está vacío. Sí se
conservan el estado de los ajustes guardados y los avisos observados de L1.

La división aleatoria evalúa registros bajo una mezcla temporal y geográfica
similar. No garantiza rendimiento sobre años futuros o regiones nuevas; para
esos objetivos habría que usar separaciones específicas.

## 8. Explicación de los gráficos generados

Hay **dos imágenes generadas**, ambas correspondientes al entrenamiento:
distribución de severidad e histogramas de atributos numéricos. No se generaron
gráficos de datos antes/después, un mapa de calor de correlaciones ni un gráfico
de componentes principales. Para esos otros análisis se guardaron tablas.

### 8.1. Barras de severidad: `clases_train.png`

![Distribución de severidad en entrenamiento](../datasets/preparados/accidente/clases_train.png)

El eje horizontal identifica las clases. La altura de cada barra es la cantidad
de accidentes de entrenamiento de esa clase. La anotación `1e6` del eje vertical
significa que los valores se expresan en millones: 1,4 equivale a 1.400.000.

| Severidad | Registros de entrenamiento | Porcentaje |
| --- | ---: | ---: |
| 1 | 677 | 0,0325 % |
| 2 | 1.395.387 | 67,0204 % |
| 3 | 621.334 | 29,8426 % |
| 4 | 64.636 | 3,1045 % |

La conclusión principal es el **fuerte desbalance**. La barra de la clase 1 casi
no se distingue por la escala lineal, pero esa clase no está ausente. Un modelo
que siempre predijera la clase 2 acertaría alrededor del 67 % en entrenamiento,
sin reconocer ninguna de las otras clases. Por eso el porcentaje global de
aciertos no basta para evaluar el problema.

### 8.2. Histogramas numéricos: `histogramas_train.png`

![Histogramas de los atributos numéricos de entrenamiento](../datasets/preparados/accidente/histogramas_train.png)

Cada panel representa una variable. El eje horizontal contiene intervalos de
valores y el eje vertical cuenta cuántos registros caen en cada intervalo. Las
escalas de los ejes cambian entre paneles: sus alturas no deben compararse sin
mirar los números. Los faltantes no se cuentan dentro de las barras.

**Momento del gráfico:** después de las reglas fijas de dominio y de generar los
atributos temporales, pero **antes de imputar, recortar y escalar**. Por eso aún
se observan extremos que luego fueron tratados. En sensación térmica o
precipitación, el gráfico representa solo la fracción con observaciones
disponibles, no todas las filas de entrenamiento.

| Panel | Qué se observa | Interpretación y relación con la preparación |
| --- | --- | --- |
| `Start_Lat` | Varios grupos y un pico importante alrededor de 34 grados. | Los registros se concentran en ciertas ubicaciones; no siguen una distribución uniforme. No demuestra mayor riesgo por latitud. |
| `Start_Lng` | Grupos separados, con concentraciones hacia longitudes cercanas a −120 y −80. | Refleja cobertura geográfica desigual. Los picos no implican que las coordenadas sean erróneas. |
| `Temperature(F)` | Mayor concentración aproximadamente entre 40 y 90 grados, con extremos que estiran el eje. | Permite detectar valores poco habituales y justifica revisar extremos. La distribución no prueba por sí sola qué valores son errores. |
| `Wind_Chill(F)` | Dos zonas de concentración entre temperaturas moderadas y frías. | Describe la sensación térmica observada. Falta más del 60 % de esta columna: no debe interpretarse como distribución completa. |
| `Humidity(%)` | Más registros en humedades medias y altas. | La variable tiene límites físicos 0–100. El código valida ese rango; el gráfico no demuestra que la humedad cause accidentes. |
| `Pressure(in)` | Gran concentración cerca de 30 pulgadas y una cola hacia valores inferiores. | Un rango pequeño concentra la mayoría de los datos. La revisión y el recorte por IQR afectan los extremos; la presión también puede variar legítimamente por ubicación y condiciones. |
| `Visibility(mi)` | Muchísimos registros alrededor de 10 millas y pocos valores que llegan hasta 140. | Puede haber concentración o límites en la forma de registrar la visibilidad; el gráfico no permite confirmar la causa. Con IQR cero, el tratamiento final no recortó esta columna. |
| `Wind_Speed(mph)` | La mayor parte se concentra en velocidades bajas; un máximo de 822,8 estira el eje. | Los valores habituales quedan visualmente comprimidos. En la preparación final se recortaron velocidades superiores a 17,3 según el límite aprendido. |
| `Precipitation(in)` | Gran concentración en cero y una cola que alcanza 25 pulgadas. | La distribución es muy asimétrica. La mediana fue cero y el IQR final también; no se aplicó recorte. Los muchos faltantes no aparecen en estas barras. |
| `Start_Hour` | Picos alrededor de las 7–8 y de las 16–17; menos registros de madrugada. | Hay patrones horarios de registro. Podrían estar relacionados con exposición al tránsito, pero sin medir esa exposición no se puede calcular riesgo por hora. |
| `Start_Weekday` | Más registros entre lunes y viernes que en sábado y domingo. | El código 0 es lunes y el 6 es domingo. El patrón describe frecuencia registrada, no peligrosidad por día. |
| `Start_Month` | Más registros entre agosto y diciembre que en varios meses anteriores. | No permite atribuir el patrón al clima o a una estación: también puede influir la cobertura temporal de la base. |

### 8.3. Qué aportó la tabla de correlaciones

Aunque no se generó una imagen de correlaciones, se guardó
[correlaciones_train.csv](../datasets/preparados/accidente/correlaciones_train.csv).
La correlación lineal entre temperatura y sensación térmica fue aproximadamente
**0,9949**, lo que muestra una asociación muy fuerte en los registros con ambas
mediciones disponibles. Esto motiva examinar redundancia y comparar PCA.

La tabla se calculó antes de imputar y utiliza pares de observaciones
disponibles. La correlación después de imputar puede ser diferente. Tampoco se
aplicó una eliminación automática de variables por superar un umbral de
correlación: ambas se conservaron en la alternativa final.

## 9. ¿Qué técnicas no se utilizaron y por qué?

Hay que diferenciar **no implementada** de **implementada pero no elegida**.

| Técnica | Situación real | Motivo o limitación que debe explicarse |
| --- | --- | --- |
| Limpieza, imputación y tratamiento de extremos | Aplicadas. | Permiten trabajar con entradas incompletas o inusuales sin perder filas. El recorte tiene las excepciones descritas. |
| Codificación de categorías y binarización de entradas | Aplicadas a las matrices. | El modelo necesita números; las categorías se representaron con indicadores 0/1. La tabla legible conserva los nombres. |
| Estandarización | Aplicada a los números de las matrices finales. | Fue la alternativa elegida según validación entre ajustes convergentes. |
| MinMax, normalización por fila y discretización | Implementadas y evaluadas, no elegidas para la salida final. | Obtuvieron menor F1 macro de validación en este experimento. Las diferencias pequeñas no prueban inferioridad general. |
| PCA | Implementada y evaluada sobre la muestra, no elegida para la salida final. | Tuvo un F1 de validación ligeramente menor y redujo solo de 12 a 11 dimensiones numéricas. |
| Regularización L1 para selección | Implementada y evaluada, no elegida para el modelo final. | Llegó al límite de iteraciones sin converger; sus 97 atributos seleccionados son un resultado exploratorio. |
| Regularización L2 | Aplicada en el clasificador final. | Controla la magnitud de los coeficientes; no equivale a balancear clases ni a borrar columnas. |
| Remuestreo para balancear clases | **No implementado.** | El flujo conservó las proporciones originales y se centró en comparar preparaciones. Es una limitación, no evidencia de que el balanceo sea innecesario. |
| Pesos de clase balanceados | **No utilizados.** | No se incorporó esa estrategia en el modelo de referencia. El desbalance sigue siendo un problema pendiente. |
| Eliminación automática por alta correlación | No implementada como filtro adicional. | Se examinó correlación y se compararon PCA y L1, pero no se fijó un umbral para borrar variables relacionadas. |

**Tomar una muestra estratificada de 20.000 registros para comparar no es
balancear clases.** Esa muestra mantiene aproximadamente las proporciones
originales y dejó apenas 6 ejemplos de la clase 1.

Si se evaluara balanceo mediante aumento de ejemplos minoritarios o reducción
de mayoritarios, tendría que hacerse solo dentro de entrenamiento y, durante
validación cruzada, dentro de cada parte de entrenamiento. No debe modificarse
la distribución de prueba para presentar una evaluación artificialmente fácil.

## 10. Resultado final y limitaciones que conviene reconocer

La verificación de resultados confirmó:

- Original y copia íntegros.
- Particiones sin filas de origen compartidas y con cobertura de todos los registros válidos.
- Matrices con 172 atributos, valores finitos y etiquetas alineadas.
- Tabla limpia con los mismos registros, sin faltantes en sus atributos preparados.

El clasificador de referencia obtuvo en prueba **0,4120 de F1 macro** y
**74,06 % de aciertos**. La recuperación de la clase 1 fue **0 sobre 145 casos
reales de esa clase en prueba**. Por lo tanto, no sería correcto presentar el
porcentaje de aciertos como prueba de buen reconocimiento de todas las
severidades.

Las principales limitaciones de esta preparación son:

1. La gran ausencia de información meteorológica se resolvió con imputación
   simple, no recuperando mediciones reales.
2. El desbalance no se corrigió; la muestra usada para comparar es especialmente
   pequeña para la clase 1.
3. Los límites IQR pueden recortar fenómenos reales; en varias columnas con IQR
   cero no hubo recorte.
4. La agrupación de categorías poco frecuentes puede perder información local.
5. La selección de la preparación depende del modelo de referencia. No se
   realizó optimización exhaustiva de parámetros ni se probaron aquí los tres
   modelos diferentes exigidos en el punto 4.
6. La división aleatoria no sustituye una evaluación temporal o geográfica.
7. No se verificó de forma exhaustiva que todas las mediciones meteorológicas
   estuvieran disponibles antes del inicio del accidente. Para un uso operativo
   al inicio habría que revisar esa condición con sus marcas de tiempo.

## 11. Texto base para responder el punto 3

El siguiente texto sirve como borrador para adaptar y defender con los archivos
de evidencia, sin afirmar que se aplicaron técnicas que quedaron fuera:

> Se realizó el preprocesamiento del dataset Accidente sobre una copia,
> conservando el archivo original. Se revisaron duplicados, etiquetas objetivo,
> valores faltantes, rangos y distribuciones. No se eliminaron filas por
> duplicados ni por etiquetas inválidas, porque esos controles no encontraron
> casos. Se separaron entrenamiento, validación y prueba en proporciones
> aproximadas de 70 %, 15 % y 15 %, manteniendo las proporciones de severidad.
>
> Las variables numéricas faltantes se imputaron con la mediana y las categorías
> faltantes con Desconocido. Se trataron extremos mediante límites del rango
> intercuartílico, sin eliminar registros y sin recortar variables cuyo rango
> intercuartílico era cero. Se excluyeron identificadores como predictores,
> campos de dirección de alta cardinalidad, texto libre y posibles datos
> posteriores al inicio. Se generaron hora, día semanal y mes, y se codificaron
> las categorías mediante columnas binarias, agrupando valores infrecuentes.
>
> Se compararon estandarización, MinMax, normalización por fila, discretización,
> selección con regularización L1 y análisis de componentes principales. Se
> eligió estandarización por su resultado de validación entre los ajustes
> convergentes. L1 quedó como exploratoria por falta de convergencia. El
> clasificador final utilizó regularización L2. Las transformaciones se
> aprendieron solo con entrenamiento; la validación cruzada de cinco partes se
> ejecutó sobre una muestra estratificada de 20.000 registros y el ajuste final
> utilizó todo el entrenamiento.
>
> Se conservaron los 2.974.335 registros y se generaron matrices finales de 172
> atributos. No se aplicó remuestreo para balancear clases: el fuerte desbalance
> sigue siendo una limitación, especialmente para la severidad 1. La tabla
> Accidente_limpio_v1.csv conserva unidades y categorías legibles; las matrices
> separadas contienen el escalado y la codificación necesarios para entrenar.

### Preguntas para practicar la explicación oral

- **¿Por qué mediana y no borrar las filas incompletas?** Para conservar
  registros ante una ausencia importante de datos y reducir la influencia de
  extremos sobre el valor de relleno. No elimina el posible sesgo de la ausencia.
- **¿Se usaron todas las técnicas juntas?** No. Hubo una base común y seis
  alternativas comparadas; la salida final usa estandarización.
- **¿Se corrigió el desbalance?** No. Estratificar conserva proporciones;
  balancear requeriría una estrategia adicional que no se aplicó.
- **¿Por qué la tabla limpia no está entre 0 y 1?** Porque conserva unidades
  legibles. Las matrices contienen la transformación final, que fue
  estandarización y no MinMax; sus valores tampoco están limitados a 0–1.
- **¿Por qué hay 172 columnas?** Porque la codificación de categorías puede
  producir varias columnas binarias por cada atributo original.
- **¿Los gráficos muestran el resultado después de limpiar?** Muestran
  entrenamiento tras reglas fijas, pero antes de imputar y recortar extremos.
- **¿Por qué no se eligió L1 si tuvo un valor de validación algo mayor?** Porque
  no convergió y la diferencia observada era mínima; no se tomó ese ajuste como
  una solución final estabilizada.

## 12. Evidencia y código para consultar

| Pregunta | Archivo |
| --- | --- |
| ¿Cuántos registros y qué particiones se generaron? | [manifest.json](../datasets/preparados/accidente/manifest.json) |
| ¿Qué valores faltantes había? | [nulos_train.csv](../datasets/preparados/accidente/nulos_train.csv) |
| ¿Qué estadísticas describen los datos explorados? | [estadisticas_train.csv](../datasets/preparados/accidente/estadisticas_train.csv) |
| ¿Qué se usó para imputar y recortar? | [parametros_numericos_finales.csv](../datasets/preparados/accidente/parametros_numericos_finales.csv) |
| ¿Qué comparación se ejecutó? | [comparacion.csv](../datasets/preparados/accidente/comparacion.csv) |
| ¿Qué ocurrió con la convergencia? | [convergencia.json](../datasets/preparados/accidente/convergencia.json) |
| ¿Qué filas entraron en la validación cruzada? | [folds_comparacion.csv](../datasets/preparados/accidente/folds_comparacion.csv) |
| ¿Qué métricas obtuvo cada clase en prueba? | [metricas_test.json](../datasets/preparados/accidente/metricas_test.json) |
| ¿Qué controles de integridad y alineación pasaron? | [verificacion.json](../datasets/preparados/accidente/verificacion.json) |
| ¿Cómo se implementó la preparación? | [scripts/preparar_datos.py](../scripts/preparar_datos.py) |
| ¿Cómo se exportó la tabla legible? | [scripts/exportar_dataset_limpio.py](../scripts/exportar_dataset_limpio.py) |
| ¿Cómo se verificaron los resultados? | [scripts/verificar_resultados.py](../scripts/verificar_resultados.py) |

Para interpretar el problema y el diccionario de variables, consultar también
[analisis.md](analisis.md). Este reporte documenta los resultados existentes y
no modifica el original, la versión limpia v1 ni los modelos guardados.
