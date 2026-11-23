from copy import deepcopy
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Inches, Pt

base = Path(__file__).parent
doc = Document(base / 'original.docx')
paragraphs = doc.paragraphs
p39, p40 = paragraphs[39], paragraphs[40]
assert p39.text.startswith('Se revisaron los duplicados')
assert p40.text.startswith('Las categorías se representaron')

texts = [
'Se revisaron los duplicados, la validez de la etiqueta y los rangos físicos de las mediciones. Los valores numéricos faltantes se completaron con la mediana de entrenamiento, es decir, el valor central de las observaciones ordenadas. Se eligió por su simplicidad y porque los valores extremos la afectan menos que a la media. Los valores categóricos faltantes se completaron con la etiqueta Desconocido. Esto permitió conservar registros incompletos, aunque completar muchos valores con una misma mediana puede reducir la variabilidad y alterar relaciones entre variables.',
'Después de imputar, se trataron los valores extremos mediante el rango intercuartílico (IQR). Este se calcula como la diferencia entre el tercer cuartil (Q3, percentil 75) y el primero (Q1, percentil 25), y mide la amplitud del 50 % central de los datos. Se establecieron los límites Q1 − 1,5 × IQR y Q3 + 1,5 × IQR. Las mediciones fuera de esos límites se reemplazaron por el límite correspondiente, sin eliminar las filas.',
'Este procedimiento busca reducir la influencia de valores muy alejados del resto mediante un criterio poco sensible a los propios extremos. Sin embargo, una medición inusual puede ser válida y aportar información sobre la severidad del accidente. Cuando el IQR fue cero, se omitió el recorte; esto ocurrió en sensación térmica, visibilidad y precipitación. No se compararon otros métodos de imputación o tratamiento de extremos, por lo que no se comprobó que estas decisiones mejoraran el desempeño frente a otras alternativas.',
'De las 49 columnas originales, incluida la variable objetivo, se excluyeron 12 de las entradas y se generaron tres atributos temporales a partir de Start_Time. Así se obtuvieron 40 variables: 12 numéricas y 28 categóricas. La codificación de las categorías produjo 162 indicadores binarios. Junto con las 12 variables numéricas sumaron 174 columnas; después de eliminar dos indicadores constantes, las matrices finales quedaron con 172 columnas predictoras: 12 numéricas y 160 binarias.',
'Tabla 2. Correspondencia entre las columnas originales y los atributos finales.'
]

def replace_text(p, text):
    props = deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    p.clear()
    run = p.add_run(text)
    if props is not None:
        run._r.insert(0, props)

replace_text(p39, texts[0])
anchor = p39._p
for text in texts[1:]:
    element = OxmlElement('w:p')
    if p39._p.pPr is not None:
        element.append(deepcopy(p39._p.pPr))
    anchor.addnext(element)
    p = Paragraph(element, p39._parent)
    p.add_run(text)
    anchor = element
caption = p
caption.paragraph_format.keep_with_next = True

rows = [
('ID, End_Time, End_Lat, End_Lng, Distance(mi), Description, Number, Street, Zipcode, Weather_Timestamp', 'Excluidas de los predictores: 0 columnas. ID se conserva para trazabilidad.'),
('Severity', 'Separada como variable objetivo: 0 columnas predictoras.'),
('Start_Time', 'Reemplazada por hora, día semanal y mes: 3 columnas numéricas.'),
('Start_Lat, Start_Lng, Temperature(F), Wind_Chill(F), Humidity(%), Pressure(in), Visibility(mi), Wind_Speed(mph), Precipitation(in)', 'Imputadas, tratadas frente a extremos cuando correspondía y estandarizadas: 1 por variable, 9 en total.'),
('Country, Turning_Loop', 'Sus indicadores se eliminaron por ser constantes: 0 columnas.'),
('Source, Side', 'Codificación binaria: 3 por variable, 6 en total.'),
('TMC, Timezone', 'Codificación binaria: 5 por variable, 10 en total.'),
('City', 'Codificación binaria: 11 columnas.'),
('County', 'Codificación binaria: 23 columnas.'),
('State', 'Codificación binaria: 27 columnas.'),
('Airport_Code', 'Codificación binaria: 9 columnas.'),
('Wind_Direction', 'Codificación binaria: 25 columnas.'),
('Weather_Condition', 'Codificación binaria: 13 columnas.'),
('Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal', 'Codificación binaria: 2 por variable, 24 en total.'),
('Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight', 'Codificación binaria: 3 por variable, 12 en total.'),
('Total', '172 columnas: 12 numéricas y 160 binarias.')
]
table = doc.add_table(rows=1, cols=2)
table._tbl.getparent().remove(table._tbl)
anchor.addnext(table._tbl)
table.autofit = False
width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
table.columns[0].width = int(width * .57)
table.columns[1].width = int(width * .43)
table.cell(0,0).text = 'Columnas originales'
table.cell(0,1).text = 'Tratamiento y columnas finales'
for left, right in rows:
    cells = table.add_row().cells
    cells[0].text, cells[1].text = left, right
for i,row in enumerate(table.rows):
    trpr = row._tr.get_or_add_trPr()
    trpr.append(OxmlElement('w:cantSplit'))
    if i == 0:
        trpr.append(OxmlElement('w:tblHeader'))
    for j,cell in enumerate(row.cells):
        cell.width = int(width * (.57 if j == 0 else .43))
        tcpr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement('w:tcBorders')
        for side in ['top','left','bottom','right']:
            border = OxmlElement('w:' + side)
            for key,value in [('val','single'),('sz','4'),('color','D9D9D9')]:
                border.set(qn('w:'+key),value)
            borders.append(border)
        tcpr.append(borders)
        if i == 0:
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'),'E7E6E6'); tcpr.append(shade)
        for p in cell.paragraphs:
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.keep_with_next = False
            for run in p.runs:
                run.font.size = Pt(9)
                run.bold = i in (0,len(rows))

replace_text(p40, 'Cada indicador binario toma el valor 1 cuando el registro pertenece a la categoría representada y 0 en caso contrario. Las categorías con una frecuencia inferior al 1 % en entrenamiento se agruparon y se estableció un máximo de 30 indicadores por variable categórica. Las cantidades de la tabla incluyen estos grupos de categorías poco frecuentes. Los indicadores Country_US y Turning_Loop_False se eliminaron por ser constantes en entrenamiento. La tabla limpia conserva unidades y etiquetas legibles; las matrices de entrenamiento, validación y prueba contienen los atributos numéricos transformados que consumen los modelos.')
for p in paragraphs:
    if p.text.startswith('Tabla ') and p is not caption:
        for old in range(5,1,-1):
            prefix = f'Tabla {old}.'
            if p.text.startswith(prefix):
                replace_text(p, p.text.replace(prefix,f'Tabla {old+1}.',1))
                break
replace_text(paragraphs[43], 'La estratificación conserva aproximadamente la proporción de cada severidad. Las medianas, los límites de recorte, las categorías y los parámetros de transformación se aprendieron en entrenamiento y se aplicaron a validación y prueba sin volver a estimarlos. Durante la validación cruzada se ajustaron dentro de cada pliegue con sus datos de entrenamiento. La comparación de transformaciones usó una muestra estratificada de 20.000 registros de entrenamiento y cinco pliegues. El ajuste final utilizó el conjunto completo de entrenamiento.')
replace_text(paragraphs[46], 'Se eligió la estandarización por su F1 macro de validación entre los ajustes que convergieron. La selección L1 llegó al límite de iteraciones y quedó como resultado exploratorio. PCA redujo las variables numéricas de 12 a 11 componentes, pero no fue la alternativa seleccionada. Por lo tanto, las matrices finales conservaron las 172 columnas descritas. Las particiones mantuvieron las proporciones originales de clase.')
doc.save(base / 'corregido.docx')
check = Document(base / 'corregido.docx')
assert len(check.tables) == 6
assert [p.text.split('.')[0] for p in check.paragraphs if p.text.startswith('Tabla ')] == [f'Tabla {i}' for i in range(1,7)]
print('Documento actualizado y verificado:', base / 'corregido.docx')
