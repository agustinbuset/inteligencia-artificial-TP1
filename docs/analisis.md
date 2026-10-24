# Análisis del TP 1 — Machine Learning

> Actualización: el dataset real ya está en `../datasets/Accidente.csv` y se procesaron
> sus 2.974.335 registros conservando el original intacto (SHA-256 verificado).
> El informe ejecutado está en [resumen de preparación](../datasets/preparados/accidente/resumen.md).
> Los apartados siguientes corresponden al análisis inicial de la documentación,
> anterior a la incorporación de los registros reales.

## Fuentes y alcance

- Consigna: [consigna - TP 1 - MachineLearning.pdf](consigna - TP 1 - MachineLearning.pdf).
- Descripción del dataset: [Dataset_no_me_choques.ipynb](Dataset_no_me_choques.ipynb).
- Versión HTML: [Dataset_Accidente.html](Dataset_Accidente.html).

El notebook y el HTML contienen la misma descripción del dataset **Accidente** y su diccionario de variables. Informan **2.974.335 eventos y 49 atributos**. Cada fila representa un accidente vial.

Estos archivos no incluyen los registros del dataset ni un análisis ejecutado. Por lo tanto, el tamaño informado, los tipos efectivos de las columnas, los valores faltantes y la distribución de las clases aún no se verificaron sobre datos reales. Los tipos indicados a continuación son conceptuales o esperados según la documentación.

## 1. Identificar el tipo/clase de problema

Se trata de un problema de **aprendizaje supervisado de clasificación multiclase**, cuyo objetivo es predecir la severidad de un accidente vial a partir de sus características, como la ubicación, el horario, las condiciones meteorológicas y los elementos de la vía.

La variable objetivo es **`Severity`**, que puede tomar cuatro categorías: **1, 2, 3 y 4**, donde 1 corresponde al nivel más leve. Como las categorías tienen un orden de gravedad, también puede caracterizarse como una **clasificación ordinal**.

Es supervisado porque cada accidente tiene una etiqueta conocida de severidad que permite entrenar y evaluar los modelos. Aunque las etiquetas sean números, representan categorías ordenadas; no se debe asumir que la distancia entre categorías sea uniforme.

El objetivo es **estimar la severidad de un accidente registrado**. Los archivos no plantean predecir si ocurrirá un accidente ni permiten identificar, por sí solos, sus causas.

## 2. Identificar las variables y sus características

Las 49 variables documentadas incluyen una variable objetivo, un identificador y atributos potencialmente predictivos. La selección final de entradas debe considerar qué información está disponible en el momento en que se desea hacer la predicción.

### Identificación, procedencia y variable objetivo

| Variable | Descripción | Tipo conceptual y características |
| --- | --- | --- |
| `ID` | Identificador único del accidente. | Categórica nominal de identificación. No tiene un orden ni una magnitud útiles como predictor directo. |
| `Source` | Fuente u origen del reporte. | Categórica nominal. Puede reflejar diferencias entre los sistemas que reportan los eventos. |
| `TMC` | Código de Traffic Message Channel. | Categórica nominal codificada numéricamente. El número identifica un tipo de mensaje; no debe interpretarse como una cantidad. |
| `Severity` | Severidad del accidente. | **Objetivo categórico ordinal**, con clases `{1, 2, 3, 4}`; 1 es la más leve. La documentación no define con precisión el criterio de cada nivel, por lo que no corresponde equipararlo automáticamente con lesiones o muertes. |

### Tiempo y ubicación

| Variable | Descripción | Tipo conceptual y características |
| --- | --- | --- |
| `Start_Time` | Fecha y hora de inicio del evento. | Temporal. Permite derivar hora, día de la semana o mes; hay que verificar el formato y la zona horaria. |
| `End_Time` | Fecha y hora de finalización del evento. | Temporal. Puede permitir calcular la duración, pero suele conocerse después del inicio. |
| `Start_Lat` | Latitud inicial del accidente. | Numérica continua, coordenada geográfica. |
| `Start_Lng` | Longitud inicial del accidente. | Numérica continua, coordenada geográfica. |
| `End_Lat` | Latitud final del tramo asociado al accidente. | Numérica continua, coordenada geográfica. Se debe comprobar su disponibilidad y significado en los datos. |
| `End_Lng` | Longitud final del tramo asociado al accidente. | Numérica continua, coordenada geográfica. Se debe comprobar su disponibilidad y significado en los datos. |
| `Distance(mi)` | Longitud del camino afectado por el accidente, en millas. | Numérica continua, conceptualmente no negativa. Su disponibilidad al momento de predecir debe verificarse. |
| `Description` | Descripción del evento en lenguaje natural. | Texto libre. Requiere tratamiento específico y puede contener información que revele la severidad. |
| `Number` | Número de la dirección de la calle. | Identificador de dirección, aunque se almacene como número. No representa una cantidad física ni un orden de gravedad. |
| `Street` | Nombre de la calle. | Categórica nominal o texto de dirección. |
| `Side` | Lado de la calle asociado al evento, documentado como Right/Left. | Categórica nominal binaria según el diccionario. No confundir con el sentido de circulación sin verificarlo. |
| `City` | Ciudad del accidente. | Categórica nominal geográfica. |
| `County` | Condado del accidente. | Categórica nominal geográfica. |
| `State` | Estado del accidente. | Categórica nominal geográfica. |
| `Zipcode` | Código postal. | Categórica nominal. Conviene conservarlo como texto para no perder ceros iniciales ni posibles extensiones. |
| `Country` | País del accidente. | Categórica nominal. Debe comprobarse si presenta variación. |
| `Timezone` | Zona horaria de la ubicación. | Categórica nominal; por ejemplo, eastern o central según el diccionario. |
| `Airport_Code` | Código del aeropuerto más cercano al accidente. | Categórica nominal. No implica que el accidente ocurra en un aeropuerto. |

### Condiciones meteorológicas

| Variable | Descripción | Tipo conceptual y características |
| --- | --- | --- |
| `Weather_Timestamp` | Fecha y hora del registro meteorológico. | Temporal. Permite evaluar la proximidad temporal entre el accidente y la medición del clima. |
| `Temperature(F)` | Temperatura en grados Fahrenheit. | Numérica continua; puede tener valores negativos. |
| `Wind_Chill(F)` | Temperatura de sensación térmica asociada al viento, en grados Fahrenheit. | Numérica continua. El diccionario la llama «viento», pero su nombre y unidad corresponden a una temperatura, no a una velocidad. Interpretación pendiente de confirmar con la fuente original. |
| `Humidity(%)` | Humedad relativa en porcentaje. | Numérica continua, con rango conceptual de 0 a 100. |
| `Pressure(in)` | Presión atmosférica, expresada en pulgadas según el diccionario. | Numérica continua. Se debe confirmar la unidad completa en la fuente del dataset. |
| `Visibility(mi)` | Distancia de visibilidad en millas. | Numérica continua, conceptualmente no negativa. |
| `Wind_Direction` | Dirección del viento. | Categórica nominal en la representación documentada. Si se convierte a ángulo, su naturaleza es circular. |
| `Wind_Speed(mph)` | Velocidad del viento en millas por hora. | Numérica continua, conceptualmente no negativa. |
| `Precipitation(in)` | Cantidad de precipitación en pulgadas. | Numérica continua, conceptualmente no negativa. Un valor faltante no debe interpretarse automáticamente como cero. |
| `Weather_Condition` | Condición meteorológica, como lluvia, nieve, tormenta o niebla. | Categórica nominal. Las categorías reales y sus frecuencias deben verificarse. |

### Elementos de la vía y puntos de interés

Estas 13 variables son **categóricas nominales binarias**, esperadas como indicadores de presencia o ausencia cerca del accidente. Hay que verificar su codificación efectiva y si admiten valores faltantes.

| Variable | Descripción e interpretación |
| --- | --- |
| `Amenity` | Presencia de un servicio o punto de interés cercano; el diccionario no especifica cuáles incluye. |
| `Bump` | Presencia de un reductor de velocidad o resalto cercano. |
| `Crossing` | Presencia de un cruce peatonal cercano. |
| `Give_Way` | Por el nombre, presencia de una señal de ceda el paso. La descripción original «signo de alejarse» es imprecisa. |
| `Junction` | Presencia de una intersección o enlace vial cercano. |
| `No_Exit` | Presencia de una calle sin salida cercana. |
| `Railway` | Presencia de infraestructura ferroviaria cercana; el diccionario la describe como un cruce de tren. |
| `Roundabout` | Por el nombre, presencia de una rotonda. La descripción original como «ruta provincial» parece un error de traducción. |
| `Station` | Presencia de una estación de transporte cercana. |
| `Stop` | Presencia de una señal de alto o pare cercana. |
| `Traffic_Calming` | Por el nombre, presencia de elementos de moderación del tránsito. La descripción original es poco precisa. |
| `Traffic_Signal` | Por el nombre, presencia de un semáforo. La descripción original como «señal de mucho tráfico» parece un error de traducción. |
| `Turning_Loop` | Por el nombre, presencia de un bucle de giro. El diccionario lo asocia con giro en U; se debe confirmar el significado exacto. |

Las aclaraciones basadas en los nombres de estas variables son interpretaciones del diccionario suministrado, no definiciones verificadas contra una fuente externa.

### Iluminación y período del día

| Variable | Descripción | Tipo conceptual y características |
| --- | --- | --- |
| `Sunrise_Sunset` | Día o noche según la salida y puesta del sol. | Categórica nominal binaria, con valores documentados `Day` y `Night`. |
| `Civil_Twilight` | Día o noche según el criterio de crepúsculo civil. | Categórica nominal binaria. El diccionario dice «crepúsculo local»; el nombre indica el criterio civil. |
| `Nautical_Twilight` | Día o noche según el criterio de crepúsculo náutico. | Categórica nominal binaria. |
| `Astronomical_Twilight` | Día o noche según el criterio de crepúsculo astronómico. | Categórica nominal binaria. |

Estos indicadores usan distintos criterios de iluminación y pueden estar relacionados entre sí; no se debe asumir que sean idénticos.

### Observaciones para la selección posterior de variables

- **Separar objetivo y entradas:** `Severity` es la etiqueta que se desea predecir y no debe formar parte de los predictores.
- **Excluir el identificador como entrada directa:** `ID` identifica registros, pero no describe una característica generalizable del accidente.
- **Definir cuándo se hará la predicción:** si se pretende estimar la severidad al inicio, `End_Time` y cualquier duración derivada no estarán disponibles. También se debe revisar la disponibilidad de `End_Lat`, `End_Lng` y `Distance(mi)`.
- **Revisar posible fuga de información:** `Description` podría mencionar la gravedad o consecuencias del evento. Antes de usarla hay que comprobar que no revele la etiqueta ni incluya información posterior al momento de predicción.
- **No confundir códigos con magnitudes:** `TMC`, `Zipcode` y `Number` no deben tratarse automáticamente como variables numéricas continuas por su apariencia o almacenamiento.
- **Verificar los datos reales antes de decidir el preprocesamiento:** faltan comprobar tipos almacenados, valores únicos, faltantes, duplicados, rangos, frecuencias de `Severity` y posibles columnas constantes. No se puede afirmar todavía que haya desbalance de clases o un porcentaje determinado de valores faltantes.

El inventario contiene **49 variables: una objetivo y 48 columnas restantes**. Estas últimas no constituyen automáticamente 48 predictores válidos; su selección se resolverá al definir el escenario de predicción y revisar el dataset.
