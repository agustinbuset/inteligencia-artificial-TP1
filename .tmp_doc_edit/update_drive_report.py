from copy import deepcopy
from pathlib import Path
import csv

from docx import Document
from docx.oxml import OxmlElement
from docx.table import Table


folder = Path('.tmp_doc_edit')
doc = Document(folder / 'drive-original.docx')
normal = next(p.style for p in doc.paragraphs if p.text.startswith('Se compararon tres'))


def find(prefix):
    return next(p for p in doc.paragraphs if p.text.startswith(prefix))


def add(before, content, style=normal):
    return before.insert_paragraph_before(content, style=style)


def fmt(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')


with Path('outputs/comparacion_columnas_51/comparacion_validation.csv').open(newline='') as source:
    rows = {(r['modelo'], int(r['columnas'])): r for r in csv.DictReader(source)}

summary = find('Se compararon tres')
summary.clear()
summary.add_run(
    'Se compararon tres modelos de clasificación de severidad sobre 2.974.335 registros. '
    'El árbol sin pesos de 172 columnas obtuvo 0,4942 de F1 macro y 79,34 % de aciertos en prueba. '
    'Los tres modelos tuvieron 0 % de recall para la severidad 1 (145 casos en prueba). '
    'En validación, el árbol de 51 columnas conservó prácticamente el mismo F1 macro '
    'y redujo el tiempo de ajuste.'
)

before = find('4 Modelos y protocolo')
add(before, 'Selección de variables por permutación', 'Heading 2')
add(before, 'Se midió cuánto caía el F1 macro del árbol de 172 columnas al permutar '
    'cada una de las 172 columnas y cada una de las 38 variables representadas. '
    'Se usaron cinco repeticiones sobre una muestra estratificada de 100.000 filas de validación. '
    'Para decidir qué retirar se permutaron juntos los indicadores de cada variable categórica. '
    'Longitud, latitud, fuente del reporte, estado y código TMC produjeron las mayores caídas.')
add(before, 'A partir del recorte ya evaluado de 89 columnas se conservaron las variables '
    'cuya permutación redujo el F1 macro en al menos 0,005. Así se obtuvieron 51 columnas '
    'codificadas que representan 13 variables, enumeradas en el anexo.')

before = find('Interpretación')
add(before, 'Comparación con menos columnas', 'Heading 2')
add(before, 'Para cada modelo sin pesos se compararon los recortes de 102 y 51 columnas '
    'con la referencia de 172. Los modelos se ajustaron con 2.082.034 filas y los mismos '
    'hiperparámetros dentro de cada método. La Tabla 7 resume los resultados sobre 446.150 '
    'filas de validación. Las variantes reducidas todavía no se evaluaron en prueba.')
caption = add(before, 'Tabla 7. F1 macro en validación según la cantidad de columnas y accuracy con 51 columnas.')
table_element = deepcopy(doc.tables[-1]._tbl)
caption._p.addprevious(table_element)
table = Table(table_element, doc._body)
values = [['Modelo', 'F1 macro\n172 columnas', 'F1 macro\n102 columnas',
           'F1 macro\n51 columnas', 'Accuracy\n51 columnas']]
for key, label in [('regresion_logistica', 'Regresión logística'),
                   ('arbol_decision', 'Árbol de decisión'), ('random_forest', 'Random Forest')]:
    values.append([label] + [fmt(float(rows[key, n]['f1_macro'])) for n in (172, 102, 51)]
                  + [fmt(100 * float(rows[key, 51]['accuracy']), 2) + ' %'])
while len(table.rows) > len(values):
    table_element.remove(table.rows[-1]._tr)
for index, (row, content) in enumerate(zip(table.rows, values)):
    row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    for cell, text in zip(row.cells, content):
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.keep_with_next = True
        runs = paragraph.runs
        if runs:
            runs[0].text = text
            for run in runs[1:]:
                run.text = ''
        else:
            paragraph.add_run(text)

add(before, 'El árbol de 51 columnas obtuvo el mayor F1 macro de validación (0,4937), '
    'prácticamente igual al del árbol de 172 columnas (0,4935). El recorte redujo un 70 % '
    'las columnas y el tiempo de ajuste pasó de 94,8 a 37,4 segundos en esta ejecución. '
    'Random Forest mejoró su F1 macro de 0,4507 a 0,4737, mientras que la regresión logística '
    'bajó de 0,4130 a 0,3812.')
diagnosis = add(before, 'Con 51 columnas, el árbol mostró la mayor brecha de F1 macro entre entrenamiento '
    'y validación (0,0390), compatible con sobreajuste. La brecha de Random Forest fue de '
    '0,0105. La regresión logística obtuvo valores bajos y similares en ambos conjuntos, '
    'un resultado compatible con subajuste. Los tres modelos reducidos conservaron un recall '
    'de 0 % para la severidad 1.')
diagnosis.paragraph_format.keep_together = True

find('El árbol sin pesos fue').add_run(
    ' El árbol de 51 columnas es el candidato reducido recomendado. La validación orientó '
    'el recorte y sirvió para comparar las variantes. Su desempeño queda por confirmar en prueba. '
    'Como la selección partió del árbol de decisión, la comparación mide cómo responde cada '
    'modelo al mismo subconjunto de variables.'
)

before = find('Evidencia producida')
for command in ('python -m scripts.comparar_columnas_arbol',
                'python -m scripts.comparar_columnas_modelos',
                'python -m scripts.analizar_columnas_arbol',
                'python -m scripts.comparar_columnas_modelos --columnas-51'):
    add(before, command)
add(before, 'Variables del modelo con 51 columnas', 'Heading 2')
add(before, 'Se conservaron Start_Lat, Start_Lng, Wind_Chill(F), Start_Hour, Start_Weekday, '
    'Start_Month, Source, TMC, Side, State, Junction, Traffic_Signal y Astronomical_Twilight. '
    'La lista exacta de columnas codificadas está en el archivo atributos.csv del árbol de 51 columnas.')
find('La preparación genera').add_run(
    ' Los rankings de permutación y los atributos seleccionados se guardan en '
    'outputs/analisis_columnas_arbol. La comparación de los tres modelos con 51 columnas '
    'se guarda en outputs/comparacion_columnas_51. El análisis por permutación usa como '
    'insumo el archivo de atributos del árbol de 89 columnas en outputs/arbol_menos_de_100.'
)
doc.save(folder / 'informe-actualizado.docx')
