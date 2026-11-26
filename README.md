# Tarjado asistido de documentos PDF

Procesa copias escaneadas de actos administrativos chilenos y genera PDF de revisión con franjas opacas incrustadas en las imágenes. Conserva los originales.

## Flujo

| Archivo | Función |
|---|---|
| `ocr_a.py` | Renderiza cada página, reconoce texto y obtiene coordenadas con PaddleOCR y PP-StructureV3. Exporta `diagnostico_ocr_layout.json`. |
| `deteccion_b.py` | Combina GLiNER, patrones y reglas para proponer qué conservar, cubrir o revisar. Exporta `sistred_gliner_detections.json`. |
| `tarjado_c.py` | Verifica el vínculo con el PDF original y exporta una copia rasterizada. Incluye revisión de escritorio y modo automático. |
| `reglas.py` | Define categorías y reglas compartidas por la detección y la revisión. |
| `tarjar_lote.py` | Ejecuta A → B → C por PDF y muestra el tiempo total y el promedio por página completada. |

La V5 corrige la inclinación en la imagen de análisis cuando hay evidencia suficiente y devuelve las coordenadas al render original. En tablas usa las celdas disponibles y la proximidad entre rótulo y valor. OpenCV localiza códigos QR. La etapa C comprueba los píxeles cubiertos en el PDF exportado. Las demás regiones visuales se proponen para revisión según su geometría y los rótulos próximos. Para repetir las regresiones, ejecuta `python -m unittest -v test_v5`.

El modo automático une las cajas de cada dato por línea y reduce el margen cerca del texto vecino. Las zonas ambiguas quedan visibles y anotadas en un registro privado. La detección de firmas, fotos, huellas y datos de menores requiere revisión. La clasificación del titular de un RUT también requiere revisión.

## Instalación en Windows

Usa Python 3.11 de 64 bits. Para la revisión de escritorio, instala Python con Tcl/Tk.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_pc.txt
```

Para habilitar una GPU NVIDIA compatible con la distribución CUDA 12.6:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y paddlepaddle
.\.venv\Scripts\python.exe -m pip install -r requirements_ocr_gpu.txt
```

GLiNER usa CPU en el lote. Los modelos descargan sus pesos en la primera ejecución. El procesamiento de los PDF ocurre localmente.

## Ejecución por lotes

Crea `pdfs` junto a los scripts y coloca allí los documentos. Para OCR en CPU:

```powershell
.\.venv\Scripts\python.exe tarjar_lote.py --device cpu
```

Para OCR en GPU, usa `--device gpu:0`. El lote guarda copias en `tarjados` y diagnósticos en `resultados`. Si ya existen la copia y el registro, omite ese documento. Usa `--salida otra_carpeta` para repetir la exportación.

Las copias automáticas están marcadas como dudosas para revisión. El archivo `.revision_privada.json` y los diagnósticos contienen texto detectado y son de uso interno.

## Revisión manual

```powershell
.\.venv\Scripts\python.exe tarjado_c.py "pdfs\documento.pdf" "resultados\lote_XXX\diagnostico_ocr_layout.json" "resultados\lote_XXX\sistred_gliner_detections.json"
```

En la ventana puedes revisar propuestas, ajustar o agregar zonas y exportar una copia con leyenda después de revisar todas las páginas.
