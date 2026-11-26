"""Etapa A: OCR local. Ejecutar python ocr_a.py documento.pdf --salida resultados."""
import argparse
import hashlib
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path

PDF_DPI = 220
MAX_PAGES = None
LAYOUT_MIN_OVERLAP = 0.15

def validar_dispositivo(device):
    if device == "cpu":
        return
    if not re.fullmatch(r"gpu(?::\d+)?", device):
        raise ValueError("Dispositivo inválido: use cpu o gpu:0")
    import paddle
    if not paddle.is_compiled_with_cuda():
        raise ValueError("Se solicitó GPU, pero PaddlePaddle es la versión CPU. Instale paddlepaddle-gpu; no se cambiará a CPU silenciosamente.")
    index = int(device.split(":")[1]) if ":" in device else 0
    if paddle.device.cuda.device_count() <= index:
        raise ValueError(f"La GPU {index} no está disponible para PaddlePaddle; compruebe controlador y CUDA")
    paddle.set_device(f"gpu:{index}")
    # Comprobar que el runtime realmente puede ejecutar una operación CUDA.
    (paddle.to_tensor([1.0]) + 1).numpy()

def ejecutar_etapa(nombre, funcion, progreso=None):
    inicio = time.monotonic()
    terminar = threading.Event()
    def informar(estado):
        segundos = round(time.monotonic() - inicio, 1)
        print(f"  {nombre}: {estado} ({segundos}s)", flush=True)
        if progreso:
            ruta = Path(progreso)
            temporal = ruta.with_suffix(".tmp")
            temporal.write_text(json.dumps({"etapa": nombre, "estado": estado, "segundos": segundos}, ensure_ascii=False), encoding="utf-8")
            temporal.replace(ruta)
    def latido():
        while not terminar.wait(15):
            informar("en ejecución")
    informar("iniciando")
    hilo = threading.Thread(target=latido, daemon=True)
    hilo.start()
    try:
        resultado = funcion()
    except BaseException:
        terminar.set(); hilo.join()
        informar("interrumpida o con error")
        raise
    terminar.set(); hilo.join()
    informar("terminada")
    return resultado

def load_pages(path, dpi=PDF_DPI):
    path = Path(path)
    suffix = path.suffix.lower()
    pages = []
    if suffix == ".pdf":
        doc = fitz.open(path)
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        page_count = len(doc) if MAX_PAGES is None else min(len(doc), MAX_PAGES)
        for page_index in range(page_count):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append({
                "page_number": page_index + 1,
                "image": image,
                "width": pix.width,
                "height": pix.height,
                "pdf_page_rotation": int(page.rotation),
                "pdf_page_rect_points": [float(v) for v in page.rect],
                "render_dpi": dpi,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "render_sha256": hashlib.sha256(pix.samples).hexdigest(),
            })
        doc.close()
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        image = Image.open(path).convert("RGB")
        pages.append({
            "page_number": 1, "image": image,
            "width": image.width, "height": image.height,
            "pdf_page_rotation": None, "pdf_page_rect_points": None,
            "render_dpi": dpi,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "render_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        })
    else:
        raise ValueError(f"Formato no admitido: {suffix}")
    return pages

def to_plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value

def get_result_dict(result_obj):
    raw = getattr(result_obj, "json", None)
    if callable(raw):
        raw = raw()
    if raw is None and isinstance(result_obj, dict):
        raw = result_obj
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"No se pudo leer resultado: {type(result_obj)}")
    return raw.get("res", raw)

def polygon_from_any(box):
    arr = np.asarray(to_plain(box), dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 4:
        return arr[:4, :2].tolist()
    if arr.ndim == 1 and arr.size >= 4:
        x0, y0, x1, y1 = arr[:4]
        return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    return None

def poly_bbox(poly):
    pts = np.asarray(poly, dtype=float)
    return [float(pts[:,0].min()), float(pts[:,1].min()),
            float(pts[:,0].max()), float(pts[:,1].max())]

def rect_polygon(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]

def poly_center(poly):
    pts = np.asarray(poly, dtype=float)
    return [float(pts[:,0].mean()), float(pts[:,1].mean())]

def build_word_map(text, char_boxes):
    """
    Mapa de offsets de caracteres a palabras.
    Si PaddleOCR entrega cajas por carácter alineadas, construye un polígono
    rectangular por palabra a partir de esas cajas. Si no, conserva la palabra
    y sus offsets, pero deja la geometría de palabra como no disponible.
    """
    words = []
    for idx, match in enumerate(re.finditer(r"\S+", str(text))):
        start, end = match.span()
        polygon = None
        bbox = None
        geometry_source = "no_word_geometry"
        if char_boxes and len(char_boxes) == len(str(text)):
            selected = [b for b in char_boxes[start:end] if b is not None]
            if selected:
                bbox = [
                    float(min(b[0] for b in selected)),
                    float(min(b[1] for b in selected)),
                    float(max(b[2] for b in selected)),
                    float(max(b[3] for b in selected)),
                ]
                polygon = rect_polygon(bbox)
                geometry_source = "paddle_character_boxes_union"
        words.append({
            "word_id": int(idx),
            "text": match.group(),
            "start": int(start),
            "end": int(end),
            "polygon": polygon,
            "bbox_xyxy": bbox,
            "geometry_source": geometry_source,
        })
    return words

def normalise_ocr_result(result_obj):
    data = get_result_dict(result_obj)
    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    polys = data.get("rec_polys")
    if polys is None or len(polys) != len(texts):
        polys = data.get("dt_polys")
    boxes = data.get("rec_boxes")
    if boxes is None:
        boxes = []
    text_word = data.get("text_word")
    text_word_boxes = data.get("text_word_boxes")

    lines = []
    for i, text in enumerate(texts):
        poly = None
        geometry_source = None
        if polys is not None and i < len(polys):
            poly = polygon_from_any(polys[i])
            if poly is not None:
                geometry_source = "rec_polys_or_dt_polys"
        if poly is None and i < len(boxes):
            poly = polygon_from_any(boxes[i])
            if poly is not None:
                geometry_source = "rec_boxes_fallback_axis_aligned"
        if poly is None:
            continue

        char_boxes = None
        try:
            chars = text_word[i]
            chars = list(chars) if isinstance(chars, str) else [str(x) for x in chars]
            raw_char_boxes = list(text_word_boxes[i])
            if len(chars) == len(str(text)) == len(raw_char_boxes) and "".join(chars) == str(text):
                char_boxes = []
                for cb in raw_char_boxes:
                    cb_poly = polygon_from_any(cb)
                    char_boxes.append(poly_bbox(cb_poly) if cb_poly is not None else None)
                if any(v is None for v in char_boxes):
                    char_boxes = None
        except Exception:
            char_boxes = None

        line_text = str(text)
        words = build_word_map(line_text, char_boxes)
        # Algunas versiones devuelven tokens completos, no caracteres.
        # Solo se acepta una alineación exacta y completa contra el texto original.
        if char_boxes is None and text_word is not None and text_word_boxes is not None:
            try:
                tokens, token_boxes = text_word[i], text_word_boxes[i]
                aligned, cursor = [], 0
                if isinstance(tokens, str) or len(tokens) != len(token_boxes):
                    raise ValueError("Formato de palabras no alineado")
                for token, box in zip(tokens, token_boxes):
                    token = str(token)
                    start = line_text.find(token, cursor)
                    if not token or start < 0 or line_text[cursor:start].strip():
                        raise ValueError("Geometría parcial")
                    end = start + len(token)
                    cursor = end
                    # Paddle puede devolver espacios como tokens explícitos.
                    # Consumen offsets, pero no requieren una caja de tarjado.
                    if token.isspace():
                        continue
                    poly_word = polygon_from_any(box)
                    if poly_word is None:
                        raise ValueError("Geometría parcial")
                    aligned.append({"word_id": len(aligned), "text": token, "start": start, "end": end,
                                    "polygon": poly_word, "bbox_xyxy": poly_bbox(poly_word),
                                    "geometry_source": "paddle_word_boxes"})
                if aligned and not line_text[cursor:].strip():
                    words = aligned
            except (IndexError, TypeError, ValueError):
                pass  # Se conserva explícitamente la geometría de línea, sin inventar cajas.
        lines.append({
            "line_index": int(i),
            "text": line_text,
            "ocr_confidence": float(scores[i]) if i < len(scores) else None,
            "polygon": [[float(x), float(y)] for x,y in poly],
            "bbox_xyxy": poly_bbox(poly),
            "center_xy": poly_center(poly),
            "geometry_source": geometry_source,
            "char_boxes_xyxy": char_boxes,
            "words": words,
        })
    return lines, data

def run_pipelines(page_record):
    rgb = np.asarray(page_record["image"].convert("RGB"))

    numero = page_record["page_number"]
    progreso = page_record.get("progress_path")
    ocr_results = ejecutar_etapa(f"Página {numero} / reconocimiento de texto", lambda: list(ocr.predict(rgb)), progreso)
    if not ocr_results:
        ocr_lines, ocr_raw = [], {}
    else:
        ocr_lines, ocr_raw = normalise_ocr_result(ocr_results[0])

    page_record.update(ocr_lines=ocr_lines, ocr_raw=ocr_raw)
    if page_record.get("checkpoint_dir"):
        ruta = Path(page_record["checkpoint_dir"]) / f"pagina_{numero:04d}_ocr.json"
        ruta.write_text(json.dumps(json_safe({"schema_version": "sistred_ocr_checkpoint_v1", "page_number": numero,
                        "source_sha256": page_record["source_sha256"], "ocr_lines": ocr_lines}), ensure_ascii=False, indent=2), encoding="utf-8")
    structure_results = ejecutar_etapa(f"Página {numero} / análisis de estructura", lambda: list(structure.predict(input=rgb)), progreso)
    if not structure_results:
        structure_raw = {}
    else:
        structure_raw = get_result_dict(structure_results[0])

    return ocr_lines, ocr_raw, structure_raw

def extract_layout_blocks(structure_raw):
    blocks = []
    parsing = structure_raw.get("parsing_res_list") or []
    for item in parsing:
        bbox = item.get("block_bbox")
        if bbox is None:
            continue
        blocks.append({
            "block_id": item.get("block_id"),
            "block_order": item.get("block_order"),
            "label": item.get("block_label", "unknown"),
            "score": None,
            "bbox_xyxy": [float(v) for v in bbox],
            "source": "parsing_res_list",
        })
    if blocks:
        return blocks

    layout = structure_raw.get("layout_det_res") or {}
    for idx, item in enumerate(layout.get("boxes") or []):
        bbox = item.get("coordinate")
        if bbox is None:
            continue
        blocks.append({
            "block_id": idx,
            "block_order": None,
            "label": item.get("label", "unknown"),
            "score": float(item["score"]) if item.get("score") is not None else None,
            "bbox_xyxy": [float(v) for v in bbox],
            "source": "layout_det_res.boxes",
        })
    return blocks

def intersection_over_line_bbox(line_bbox, block_bbox):
    x0 = max(line_bbox[0], block_bbox[0])
    y0 = max(line_bbox[1], block_bbox[1])
    x1 = min(line_bbox[2], block_bbox[2])
    y1 = min(line_bbox[3], block_bbox[3])
    inter = max(0.0, x1-x0) * max(0.0, y1-y0)
    area = max(1.0, (line_bbox[2]-line_bbox[0]) * (line_bbox[3]-line_bbox[1]))
    return inter / area

def assign_lines_to_regions(lines, blocks):
    assigned = []
    for line in lines:
        best_block, best_overlap = None, 0.0
        for block in blocks:
            overlap = intersection_over_line_bbox(line["bbox_xyxy"], block["bbox_xyxy"])
            if overlap > best_overlap:
                best_block, best_overlap = block, overlap
        item = dict(line)
        if best_block is not None and best_overlap >= LAYOUT_MIN_OVERLAP:
            item["layout_block_id"] = best_block.get("block_id")
            item["layout_block_order"] = best_block.get("block_order")
            item["layout_label"] = best_block.get("label")
            item["layout_overlap_ratio"] = float(best_overlap)
        else:
            item["layout_block_id"] = None
            item["layout_block_order"] = None
            item["layout_label"] = "unassigned"
            item["layout_overlap_ratio"] = float(best_overlap)
        assigned.append(item)
    return assigned

def region_sort_key(block):
    order = block.get("block_order")
    if order is not None:
        return (0, int(order), block["bbox_xyxy"][1], block["bbox_xyxy"][0])
    return (1, block["bbox_xyxy"][1], block["bbox_xyxy"][0], 0)

def reconstruct_page_text(page):
    blocks = sorted(page.get("layout_blocks", []), key=region_sort_key)
    lines = page.get("assigned_lines", [])
    chunks = []
    used_indices = set()
    for block in blocks:
        selected = [ln for ln in lines if ln.get("layout_block_id") == block.get("block_id")]
        selected.sort(key=lambda ln: (ln["center_xy"][1], ln["center_xy"][0]))
        if not selected:
            continue
        chunks.append({
            "block_id": block.get("block_id"),
            "block_order": block.get("block_order"),
            "label": block.get("label"),
            "text": "\n".join(ln["text"] for ln in selected),
            "line_indices": [ln["line_index"] for ln in selected],
        })
        used_indices.update(ln["line_index"] for ln in selected)

    unassigned = [ln for ln in lines if ln["line_index"] not in used_indices]
    unassigned.sort(key=lambda ln: (ln["center_xy"][1], ln["center_xy"][0]))
    if unassigned:
        chunks.append({
            "block_id": None, "block_order": None, "label": "unassigned",
            "text": "\n".join(ln["text"] for ln in unassigned),
            "line_indices": [ln["line_index"] for ln in unassigned],
        })
    return chunks

def make_overlay(page):
    image = np.asarray(page["image"].convert("RGB")).copy()
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Regiones de layout: rectángulos y etiqueta.
    for block in page.get("layout_blocks", []):
        x0,y0,x1,y1 = [int(round(v)) for v in block["bbox_xyxy"]]
        cv2.rectangle(canvas, (x0,y0), (x1,y1), (255,140,0), 2)
        label = f"B{block.get('block_id')} {block.get('label')} o={block.get('block_order')}"
        cv2.putText(canvas, label[:55], (max(0,x0), max(15,y0-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,140,0), 1, cv2.LINE_AA)

    # Polígonos OCR: contorno verde, número de línea.
    for line in page.get("assigned_lines", []):
        pts = np.asarray(line["polygon"], dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(canvas, [pts], True, (0,200,0), 2)
        cx,cy = [int(round(v)) for v in line["center_xy"]]
        cv2.putText(canvas, str(line["line_index"]), (cx,cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,220), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items() if k not in {"image"}}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def procesar(entrada, salida, dpi=220, device="cpu"):
    global np, Image, fitz, cv2, ocr, structure, DEVICE
    import importlib
    np = ejecutar_etapa("Importación de NumPy", lambda: importlib.import_module("numpy"))
    Image = ejecutar_etapa("Importación de Pillow", lambda: importlib.import_module("PIL.Image"))
    fitz = ejecutar_etapa("Importación de PyMuPDF", lambda: importlib.import_module("pymupdf"))
    cv2 = ejecutar_etapa("Importación de OpenCV", lambda: importlib.import_module("cv2"))
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    paddleocr_module = ejecutar_etapa("Importación de PaddleOCR", lambda: importlib.import_module("paddleocr"))
    PaddleOCR, PPStructureV3 = paddleocr_module.PaddleOCR, paddleocr_module.PPStructureV3
    entrada, salida = Path(entrada), Path(salida)
    if not entrada.is_file() or entrada.suffix.lower() != ".pdf":
        raise ValueError("La entrada debe ser un PDF existente")
    if not 72 <= dpi <= 600:
        raise ValueError("DPI fuera de rango (72 a 600)")
    ejecutar_etapa("Comprobación del dispositivo y carga de PaddlePaddle", lambda: validar_dispositivo(device))
    json_path = salida / "diagnostico_ocr_layout.json"
    if json_path.exists():
        raise ValueError("La salida ya contiene un OCR; use otra carpeta")
    salida.mkdir(parents=True, exist_ok=True)
    overlays = salida / "overlays"
    overlays.mkdir(exist_ok=True)
    checkpoints = salida / "paginas"
    checkpoints.mkdir(exist_ok=True)
    DEVICE = device
    print("Cargando modelos OCR y layout. La primera ejecución puede descargar pesos.")
    ocr = ejecutar_etapa("Carga del modelo OCR", lambda: PaddleOCR(
        lang="es",
        ocr_version="PP-OCRv6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        enable_mkldnn=False,
        device=DEVICE,
    ), salida / "progreso.json")

    # Se intenta desactivar submódulos no necesarios para el diagnóstico de layout.
    # Si una compilación concreta no acepta alguno de estos argumentos, se usa
    # la inicialización mínima documentada y se informa la excepción.
    try:
        structure = ejecutar_etapa("Carga de PP-StructureV3", lambda: PPStructureV3(
            device=DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_seal_recognition=False,
            enable_mkldnn=False,
        ), salida / "progreso.json")
        print("PP-StructureV3 inicializado con módulos opcionales desactivados.")
    except TypeError as exc:
        print("Constructor extendido no aceptado; intento constructor básico:", exc)
        structure = ejecutar_etapa("Carga básica de PP-StructureV3", lambda: PPStructureV3(
            device=DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        ), salida / "progreso.json")
        print("PP-StructureV3 inicializado con configuración básica.")


    all_pages = ejecutar_etapa("Renderizado del PDF", lambda: load_pages(entrada, dpi=dpi), salida / "progreso.json")
    for page in all_pages:
        page["source_file"] = entrada.name
        page["progress_path"] = str(salida / "progreso.json")
        page["checkpoint_dir"] = str(checkpoints)
        print(f"OCR: página {page['page_number']} de {len(all_pages)}", flush=True)
        try:
            lines, raw, structure_raw = run_pipelines(page)
            page.update(ocr_lines=lines, ocr_raw=raw, structure_raw=structure_raw)
        except Exception as exc:
            page.update(processing_error=repr(exc), processing_traceback=traceback.format_exc())
            page.setdefault("ocr_lines", [])
            print("Página pendiente de revisión manual:", exc)
        page["layout_blocks"] = extract_layout_blocks(page.get("structure_raw", {}))
        page["assigned_lines"] = assign_lines_to_regions(page["ocr_lines"], page["layout_blocks"])
        page["ordered_regions"] = reconstruct_page_text(page)
        if not page["ocr_lines"] and not page.get("processing_error"):
            page["processing_error"] = "OCR sin texto: confirmar visualmente la página"
        overlay_path = overlays / f"pagina_{page['page_number']:04d}.png"
        Image.fromarray(make_overlay(page)).save(overlay_path)
        page["overlay_path"] = str(overlay_path.resolve())
        (checkpoints / f"pagina_{page['page_number']:04d}.json").write_text(
            json.dumps(json_safe({"schema_version": "sistred_page_checkpoint_v1", "page": page}), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Página {page['page_number']} guardada: {len(page['ocr_lines'])} líneas", flush=True)
    export_pages = []
    for page in all_pages:
        # Mantiene el índice original de PaddleOCR; no renumera las líneas.
        lines_for_export = []
        for line in page.get("assigned_lines", []):
            lines_for_export.append({
                **line,
                "page_number": int(page["page_number"]),
                "image_width_px": int(page["width"]),
                "image_height_px": int(page["height"]),
            })

        export_pages.append({
            "source_file": page["source_file"],
            "page_number": int(page["page_number"]),
            "image_width_px": int(page["width"]),
            "image_height_px": int(page["height"]),
            "render_dpi": int(page["render_dpi"]),
            "pdf_page_rotation": int(page["pdf_page_rotation"] or 0),
            "source_sha256": page["source_sha256"],
            "render_sha256": page["render_sha256"],
            "pdf_page_rect_points": page["pdf_page_rect_points"],
            "coordinate_system": "rendered_page_pixels_top_left_origin",
            "ocr_lines": lines_for_export,
            "layout_blocks": page.get("layout_blocks", []),
            "ordered_regions": page.get("ordered_regions", []),
            "processing_error": page.get("processing_error"),
            "overlay_path": page.get("overlay_path"),
        })

    integration_payload = {
        "schema_version": "sistred_ocr_layout_v2",
        "project": "SISTRED OCR + layout",
        "ocr": "PaddleOCR 3.7.0 PP-OCRv6 Spanish",
        "layout": "PP-StructureV3",
        "coordinate_system": {
            "units": "pixels",
            "origin": "top_left",
            "page_image_is_unmodified": True,
            "note": "Las coordenadas corresponden a la imagen renderizada del PDF. No se aplicó deskew, rotación ni deswarping."
        },
        "pages": export_pages,
    }


    import importlib.metadata as metadata
    integration_payload["runtime"] = {name: metadata.version(name) for name in ("paddleocr", "paddlex", "pymupdf")}
    json_path.write_text(json.dumps(json_safe(integration_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    texto = "\n\n".join(f"PÁGINA {p['page_number']}\n" + "\n".join(r["text"] for r in p["ordered_regions"]) for p in all_pages)
    (salida / "texto_reconstruido.txt").write_text(texto, encoding="utf-8")
    print("OCR guardado:", json_path.resolve())
    (salida / "progreso.json").write_text(json.dumps({"estado": "completo", "paginas": len(all_pages), "json": str(json_path.resolve())}), encoding="utf-8")
    return json_path

def main(argv=None):
    parser = argparse.ArgumentParser(description="A: extraer texto y geometría de un PDF local")
    parser.add_argument("entrada", type=Path, help="PDF original")
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--device", default="cpu", help="cpu o gpu:0 según la instalación")
    args = parser.parse_args(argv)
    try:
        procesar(args.entrada, args.salida, args.dpi, args.device)
    except (ImportError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\nRevise las dependencias en requirements_pc.txt.\n")

if __name__ == "__main__":
    main()
