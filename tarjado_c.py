"""Revisión en Jupyter/Colab y exportación de un PDF nuevo de imágenes saneadas.

Uso: from tarjado_c import Sesion, revisar
     s = Sesion('original.pdf', 'diagnostico_ocr_layout.json', 'sistred_gliner_detections.json')
     revisar(s)
Dependencias: pymupdf pillow ipywidgets. Mantener reglas.py junto a este archivo.
"""
import copy
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw
from reglas import VERSION, TIPOS, aplicar

def sha(data):
    return hashlib.sha256(data).hexdigest()

class Sesion:
    def __init__(self, original, ocr_json, detecciones_json):
        self.original = Path(original)
        self.original_hash = sha(self.original.read_bytes())
        self.ocr = json.loads(Path(ocr_json).read_text(encoding="utf-8"))
        self.datos = json.loads(Path(detecciones_json).read_text(encoding="utf-8"))
        if self.ocr.get("schema_version") != "sistred_ocr_layout_v2" or self.datos.get("schema_version") != "sistred_gliner_detections_v1":
            raise ValueError("Esquema incompatible")
        if self.datos.get("source_ocr_sha256") != sha(Path(ocr_json).read_bytes()):
            raise ValueError("Detecciones sin vínculo al OCR. Vuelva a ejecutar GLiNER actualizado.")
        self.paginas = [p for p in self.ocr["pages"] if p["source_file"] == self.original.name]
        if not self.paginas or any(p.get("source_sha256") != self.original_hash for p in self.paginas):
            raise ValueError("El OCR no corresponde al original. Vuelva a ejecutar el OCR actualizado.")
        self.tipo = "sin_clasificar"
        self.tipo_propuesto = self.datos.get("tipos_documento", {}).get(self.original.name, {}).get("tipo", "sin_clasificar")
        self.eventos, self.revisadas = [], set()
        self.detecciones = []
        self.imagenes, self.tamanos = [], []
        self.errores = [p["page_number"] for p in self.paginas if p.get("processing_error")]
        self.errores += [p["page_number"] for p in self.datos.get("page_errors", []) if p["source_file"] == self.original.name]
        with fitz.open(self.original) as doc:
            if len(self.paginas) != len(doc) or [p["page_number"] for p in self.paginas] != list(range(1, len(doc)+1)):
                raise ValueError("Faltan páginas o el orden no coincide")
            for page, meta in zip(doc, self.paginas):
                pix = page.get_pixmap(matrix=fitz.Matrix(meta["render_dpi"]/72, meta["render_dpi"]/72), alpha=False)
                if (pix.width, pix.height) != (meta["image_width_px"], meta["image_height_px"]):
                    raise ValueError("Dimensiones incompatibles")
                if sha(pix.samples) != meta.get("render_sha256"):
                    raise ValueError("El render difiere del OCR; usar el mismo entorno de renderizado")
                self.imagenes.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                self.tamanos.append((page.rect.width, page.rect.height))
        for d in self.datos["detections"]:
            if d["source_file"] == self.original.name:
                item = aplicar(d)
                item["review_id"] = len(self.detecciones)
                self.detecciones.append(item)

    def clasificar(self, tipo):
        if tipo not in TIPOS or tipo == "sin_clasificar":
            raise ValueError("Seleccione el tipo de acto")
        self.tipo = tipo
        self.revisadas.clear()
        self.detecciones = [dict(aplicar(d, tipo), review_id=i) for i, d in enumerate(self.detecciones)]
        self.eventos.append({"accion": "clasificar", "tipo": tipo})

    def _poligonos(self, pagina, poligonos):
        if not 1 <= pagina <= len(self.imagenes) or not poligonos:
            raise ValueError("Página o geometría inválida")
        w, h = self.imagenes[pagina-1].size
        for poly in poligonos:
            if len(poly) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) or not (0 <= p[0] < w and 0 <= p[1] < h) for p in poly):
                raise ValueError("Coordenadas fuera de la imagen")
            area = abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly))))/2
            if area < 1:
                raise ValueError("Zona vacía")

    def decidir(self, indice, decision, motivo, poligonos=None, *, precision="manual"):
        if decision not in {"aceptar", "rechazar"} or not motivo.strip():
            raise ValueError("Indique decisión y motivo")
        d = self.detecciones[indice]
        antes = copy.deepcopy(d)
        if decision == "aceptar":
            zonas = poligonos if poligonos is not None else d.get("polygons")
            if d.get("requires_geometry_review") and poligonos is None:
                raise ValueError("Ajuste manualmente la zona: la geometría no es suficientemente precisa")
            self._poligonos(d["page"], zonas)
            d["polygons"] = zonas
            if poligonos is not None:
                d["location_precision"] = precision
                d["requires_geometry_review"] = False
        d.update(decision=decision, motivo=motivo)
        self.revisadas.discard(d["page"])
        self.eventos.append({"accion": "decision", "antes": antes, "despues": copy.deepcopy(d), "fecha": datetime.now(timezone.utc).isoformat()})

    def agregar(self, pagina, poligonos, categoria, regla, motivo):
        self._poligonos(pagina, poligonos)
        if not categoria.strip() or not regla.strip() or not motivo.strip():
            raise ValueError("Indique categoría, regla y motivo")
        d = {"page": pagina, "category": categoria, "rule_ids": [regla], "polygons": poligonos,
             "location_precision": "manual", "decision": "aceptar", "motivo": motivo,
             "review_id": len(self.detecciones), "tarjar": "revisar"}
        self.detecciones.append(d)
        self.revisadas.discard(pagina)
        self.eventos.append({"accion": "agregar", "dato": copy.deepcopy(d)})

    def marcar_pagina(self, pagina):
        if not 1 <= pagina <= len(self.imagenes):
            raise ValueError("Página inválida")
        if any(d["page"] == pagina and d["decision"] == "pendiente" for d in self.detecciones):
            raise ValueError("Quedan detecciones pendientes en esta página")
        self.revisadas.add(pagina)
        self.eventos.append({"accion": "revision_visual_completa", "pagina": pagina,
                             "fecha": datetime.now(timezone.utc).isoformat()})

    def _recuadro_pendiente(self, d):
        """Caja visual de diagnóstico; no autoriza una franja de tarjado."""
        pagina = d.get("page")
        if not isinstance(pagina, int) or not 1 <= pagina <= len(self.imagenes):
            return None
        w, h = self.imagenes[pagina - 1].size

        def validar(caja):
            if caja is None or len(caja) != 4 or not all(math.isfinite(v) for v in caja):
                return None
            x0, y0, x1, y1 = caja
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                return None
            return (max(0, math.floor(x0) - 3), max(0, math.floor(y0) - 3),
                    min(w - 1, math.ceil(x1) + 3), min(h - 1, math.ceil(y1) + 3))

        # Priorizar polígonos del dato; usar la caja OCR si no son utilizables.
        try:
            puntos = [p for poly in (d.get("polygons") or []) for p in poly]
            if puntos and all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in puntos):
                xs, ys = zip(*puntos)
                caja = validar((min(xs), min(ys), max(xs), max(ys)))
                if caja is not None:
                    return caja
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            return validar(d.get("bbox_xyxy"))
        except (TypeError, ValueError, OverflowError):
            return None

    def imagen(self, pagina, vista_previa=False, recuadros_pendientes=False):
        im = self.imagenes[pagina-1].copy()
        draw = ImageDraw.Draw(im)
        zonas = [d for d in self.detecciones if d["page"] == pagina]
        for d in zonas:
            if d["decision"] != "pendiente":
                continue
            if recuadros_pendientes:
                caja = self._recuadro_pendiente(d)
                if caja is not None:
                    color = "#0078D4" if d.get("motivo_pendiente_tipo") == "geometria" else "#E87500"
                    draw.rectangle(caja, outline=color, width=4)
            elif vista_previa:
                for poly in d.get("polygons") or []:
                    pts = [tuple(p) for p in poly]
                    if pts:
                        draw.line(pts+[pts[0]], fill="orange", width=3)
        # Las franjas negras van al final para conservar la opacidad.
        for d in zonas:
            if d["decision"] == "aceptar":
                for poly in d.get("polygons") or []:
                    draw.polygon([tuple(p) for p in poly], fill="black")
        return im

    def exportar(self, destino, revisor, leyenda, comprobaciones):
        """Las comprobaciones son aceptación explícita de cobertura, excepciones y leyenda."""
        if self.tipo == "sin_clasificar" or not revisor.strip() or not leyenda.strip() or comprobaciones is not True:
            raise ValueError("Falta clasificación, revisor, leyenda o comprobación final")
        if self.revisadas != set(range(1, len(self.imagenes)+1)):
            raise ValueError("Revise todas las páginas, incluidas las que fallaron en OCR")
        if any(d["decision"] == "pendiente" for d in self.detecciones):
            raise ValueError("Hay decisiones pendientes")
        return self._guardar_pdf(destino, leyenda, {
            "revisor": revisor, "errores_ocr_revisados": self.errores,
        })

    def _franja_automatica(self, deteccion, lineas):
        pagina = deteccion["page"]
        poligonos = deteccion.get("polygons")
        self._poligonos(pagina, poligonos)

        def limites(puntos):
            xs, ys = zip(*puntos)
            return (math.floor(min(xs)), math.floor(min(ys)),
                    math.ceil(max(xs)), math.ceil(max(ys)))

        def cruza(a, b):
            return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]

        indice = deteccion.get("line_index")
        linea = lineas.get(indice)
        inicio, fin = deteccion.get("start"), deteccion.get("end")
        if (linea is None or not isinstance(inicio, int) or not isinstance(fin, int)
                or not 0 <= inicio < fin <= len(linea["text"])):
            raise ValueError("Falta la ubicación del dato dentro de la línea OCR")

        # B entrega un fragmento por línea. Una franja cubre sus cajas y los espacios entre ellas.
        x0, y0, x1, y1 = limites(p for poligono in poligonos for p in poligono)
        ancho, alto = self.imagenes[pagina - 1].size
        base = (x0, y0, min(ancho - 1, x1), min(alto - 1, y1))
        margen = max(1, round((y1 - y0) * 0.12))
        franja = [max(0, x0 - margen), max(0, y0 - margen),
                  min(ancho - 1, x1 + margen), min(alto - 1, y1 + margen)]

        # Proteger los caracteres no seleccionados, usando cajas de palabras cuando
        # Paddle no entrega cajas individuales.
        texto = linea["text"]
        cajas = linea.get("char_boxes_xyxy")
        if cajas and len(cajas) == len(texto):
            vecinos = [(i, i + 1, tuple(caja)) for i, caja in enumerate(cajas) if caja]
        else:
            vecinos = [(p["start"], p["end"], limites(p["polygon"]))
                       for p in linea.get("words", []) if p.get("polygon")]
        fuera = {i for i, caracter in enumerate(texto)
                 if not caracter.isspace() and not inicio <= i < fin}
        protegidos = []
        for desde, hasta, caja in vecinos:
            posiciones = fuera.intersection(range(desde, hasta))
            if posiciones:
                protegidos.append(caja)
                fuera.difference_update(posiciones)
        if fuera:
            raise ValueError("Faltan coordenadas del texto vecino en la misma línea")

        protegidos.extend(limites(otra["polygon"]) for otro, otra in lineas.items()
                          if otro != indice and otra.get("polygon"))
        for caja in protegidos:
            if not cruza(franja, caja):
                continue
            if cruza(base, caja):
                raise ValueError("La franja invadiría texto vecino")
            # Recortar el margen donde la caja vecina se acerque a la franja.
            if caja[2] < base[0]:
                franja[0] = max(franja[0], math.floor(caja[2]) + 1)
            if caja[0] > base[2]:
                franja[2] = min(franja[2], math.ceil(caja[0]) - 1)
            if caja[3] < base[1]:
                franja[1] = max(franja[1], math.floor(caja[3]) + 1)
            if caja[1] > base[3]:
                franja[3] = min(franja[3], math.ceil(caja[1]) - 1)

        x0, y0, x1, y1 = franja
        return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]

    def exportar_automatico(self, destino, *, recuadros_pendientes=True):
        """Genera una copia de prueba con recuadros de diagnóstico opcionales."""
        dudosas = []
        # Solo la copia automática usa la clasificación propuesta sin confirmación humana.
        if self.tipo == "sin_clasificar" and self.tipo_propuesto in TIPOS:
            self.tipo = self.tipo_propuesto
        lineas_por_pagina = {p["page_number"]: {l["line_index"]: l for l in p["ocr_lines"]}
                             for p in self.paginas}
        for d in self.detecciones:
            propuesta = d.get("tarjar")
            if propuesta == "no":
                self.decidir(d["review_id"], "rechazar", "Reglas: conservar visible")
                continue
            if propuesta == "si" and not d.get("requires_geometry_review"):
                try:
                    franja = self._franja_automatica(d, lineas_por_pagina.get(d["page"], {}))
                    self.decidir(d["review_id"], "aceptar", "Reglas: tarjar automáticamente",
                                 franja, precision="franja_automatica")
                    continue
                except ValueError as exc:
                    motivo = str(exc)
            else:
                motivo = "Geometría por revisar" if propuesta == "si" else "Clasificación por revisar"
            tipo_motivo = "geometria" if propuesta == "si" else "clasificacion"
            d["motivo_pendiente_tipo"] = tipo_motivo
            d["motivo_pendiente"] = motivo
            encuadrada = recuadros_pendientes and self._recuadro_pendiente(d) is not None
            dudosas.append({"id": d["review_id"], "pagina": d["page"],
                            "categoria": d["category"], "motivo": motivo,
                            "recuadro_diagnostico": encuadrada,
                            "tipo_motivo": tipo_motivo,
                            "color_recuadro": ("azul" if tipo_motivo == "geometria" else "naranja") if encuadrada else None})
        aplicadas = sum(d["decision"] == "aceptar" for d in self.detecciones)
        encuadradas = sum(d["recuadro_diagnostico"] for d in dudosas)
        naranjas = sum(d["color_recuadro"] == "naranja" for d in dudosas)
        azules = sum(d["color_recuadro"] == "azul" for d in dudosas)
        aviso_recuadros = (
            f"Recuadros naranjas: {naranjas} (clasificación por revisar).\n"
            f"Recuadros azules: {azules} (geometría por revisar).\n"
            f"Pendientes sin recuadro localizable: {len(dudosas) - encuadradas}.\n"
            "Los recuadros pueden ser aproximados; no equivalen a un tarjado.\n"
        ) if recuadros_pendientes else "Recuadros de diagnóstico desactivados.\n"
        leyenda = (
            "COPIA DUDOSA - PENDIENTE DE REVISIÓN\n\n"
            "Tarjado automático de prueba. Uso interno. Los casos pendientes pueden seguir visibles.\n"
            f"Tipo de acto propuesto: {self.tipo_propuesto}.\n"
            f"Zonas tarjadas automáticamente: {aplicadas}.\n"
            f"Zonas pendientes de decisión: {len(dudosas)}.\n"
            f"{aviso_recuadros}"
            "NO PUBLICAR. Revisar visualmente todas las páginas, incluidas firmas, imágenes y códigos QR."
        )
        return self._guardar_pdf(destino, leyenda, {
            "estado": "dudoso", "modo": "automatico", "dudosas": dudosas,
            "recuadros_pendientes": recuadros_pendientes,
            "cantidad_recuadros": encuadradas,
            "recuadros_naranjas": naranjas,
            "recuadros_azules": azules,
            "tipo_acto_propuesto": self.tipo_propuesto,
            "paginas_con_error": sorted(set(self.errores)),
            "paginas_por_revisar": list(range(1, len(self.paginas) + 1)),
        }, recuadros_pendientes=recuadros_pendientes)

    def _guardar_pdf(self, destino, leyenda, datos_revision, *, recuadros_pendientes=False):
        if sha(self.original.read_bytes()) != self.original_hash:
            raise ValueError("El original cambió")
        destino = Path(destino)
        registro = destino.with_suffix(".revision_privada.json")
        if destino.resolve() == self.original.resolve() or destino.exists() or registro.exists():
            raise ValueError("Use una salida nueva; no se sobrescriben archivos")
        doc = fitz.open()
        for i, (w, h) in enumerate(self.tamanos, 1):
            image = self.imagen(i, recuadros_pendientes=recuadros_pendientes)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            p = doc.new_page(width=w, height=h)
            p.insert_image(p.rect, stream=buf.getvalue())
        # La leyenda también se rasteriza; no se importa ningún objeto del original.
        aviso = fitz.open()
        p = aviso.new_page()
        if p.insert_textbox(fitz.Rect(40, 40, 555, 802), leyenda, fontsize=11) < 0:
            doc.close(); aviso.close()
            raise ValueError("La leyenda no cabe; reduzca su extensión")
        pix = p.get_pixmap(dpi=180, alpha=False)
        target = doc.new_page(width=p.rect.width, height=p.rect.height)
        target.insert_image(target.rect, stream=pix.tobytes("png"))
        aviso.close()
        doc.set_metadata({})
        data = doc.tobytes(garbage=4, deflate=True)
        doc.close()
        with fitz.open(stream=data, filetype="pdf") as check:
            if check.embfile_count() or any(p.get_text().strip() or p.get_links() or list(p.annots() or []) for p in check):
                raise ValueError("El PDF no superó la comprobación estructural")
        audit = {"source_sha256": self.original_hash, "output_sha256": sha(data), "rules_version": VERSION,
                 "tipo": self.tipo, "fecha": datetime.now(timezone.utc).isoformat(),
                 "leyenda": leyenda, "decisiones": self.detecciones, "eventos": self.eventos,
                 **datos_revision}
        registro_data = json.dumps(audit, ensure_ascii=False, indent=2)
        with registro.open("x", encoding="utf-8") as archivo:
            archivo.write(registro_data)
        try:
            with destino.open("xb") as archivo:
                archivo.write(data)
        except OSError:
            registro.unlink()
            raise
        return destino

def revisar(sesion):
    """Interfaz de notebook sin servicios externos; coordenadas en píxeles originales."""
    import ipywidgets as W
    from IPython.display import display, clear_output
    s = sesion
    tipo = W.Dropdown(options=TIPOS, description="Tipo")
    pagina = W.IntSlider(min=1, max=len(s.imagenes), description="Página")
    indice = W.Dropdown(description="Detección")
    motivo = W.Textarea(description="Motivo")
    zona = W.Text(description="Rectángulo", placeholder="x0,y0,x1,y1; vacío conserva polígonos")
    categoria = W.Text(value="qr", description="Categoría")
    regla = W.Text(value="RNA-11", description="Regla")
    revisor = W.Text(description="Revisor")
    leyenda = W.Textarea(description="Leyenda", placeholder="Texto y fundamentos acordados; no se generan causales")
    destino = W.Text(value="documento_tarjado.pdf", description="Salida")
    confirmar = W.Checkbox(description="Verifiqué cobertura, excepciones, QR, códigos y leyenda", indent=False)
    pantalla, mensajes = W.Output(), W.Output()

    def zonas():
        if not zona.value.strip():
            return None
        x0, y0, x1, y1 = map(float, zona.value.split(","))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Rectángulo inválido")
        return [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]

    def refrescar(*_):
        confirmar.value = False
        indice.options = [(f'{d["review_id"]}: {d["category"]} / {d["decision"]} / {d.get("fragment_text", "manual")}', d["review_id"])
                          for d in s.detecciones if d["page"] == pagina.value]
        with pantalla:
            clear_output(wait=True)
            print(f"Imagen original: {s.imagenes[pagina.value-1].size}. Naranja: pendiente. Negro: aceptado.")
            print("Revisar también firmas, fotos, huellas, menores, referencias al recinto y omisiones del OCR.")
            for d in s.detecciones:
                if d["page"] == pagina.value:
                    print(d["review_id"], d["category"], "propuesta:", d.get("tarjar"),
                          "reglas:", d.get("rule_ids"), "confianza:", d.get("score"),
                          "geometría:", d.get("location_precision"), "caja:", d.get("bbox_xyxy"))
            if pagina.value in s.errores:
                print("ATENCIÓN: esta página falló en OCR/detección. Revisión manual completa obligatoria.")
            display(s.imagen(pagina.value, True))

    def accion(fn):
        def ejecutar(_):
            with mensajes:
                clear_output()
                try:
                    result = fn()
                    print(result or "Registrado")
                    refrescar()
                except Exception as exc:
                    print(str(exc))
        return ejecutar

    botones = []
    for texto, fn in [
        ("Confirmar tipo", lambda: s.clasificar(tipo.value)),
        ("Aceptar / ajustar", lambda: s.decidir(indice.value, "aceptar", motivo.value, zonas())),
        ("Rechazar", lambda: s.decidir(indice.value, "rechazar", motivo.value)),
        ("Agregar zona", lambda: s.agregar(pagina.value, zonas(), categoria.value, regla.value, motivo.value)),
        ("Página revisada", lambda: s.marcar_pagina(pagina.value)),
        ("Exportar aprobado", lambda: s.exportar(destino.value, revisor.value, leyenda.value, confirmar.value)),
    ]:
        b = W.Button(description=texto)
        b.on_click(accion(fn)); botones.append(b)
    pagina.observe(refrescar, names="value")
    display(W.VBox([tipo, pagina, indice, zona, motivo, categoria, regla, W.HBox(botones[:3]), W.HBox(botones[3:]),
                    revisor, leyenda, destino, confirmar, mensajes, pantalla]))
    refrescar()


def revisar_pc(sesion):
    """Ventana local de revisión; arrastrar sobre la imagen dibuja una zona manual."""
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import ImageTk

    s = sesion
    root = tk.Tk()
    root.title("Tarjado C — revisión local")
    root.geometry("1400x900")
    pagina = tk.IntVar(value=1)
    tipo = tk.StringVar(value=s.tipo)
    motivo = tk.StringVar()
    categoria = tk.StringVar(value="qr")
    regla = tk.StringVar(value="RNA-11")
    revisor = tk.StringVar()
    confirmar = tk.BooleanVar(value=False)
    rectangulo = tk.StringVar()
    estado = tk.StringVar()
    escala = tk.DoubleVar(value=0.4)
    original_visible = tk.BooleanVar(value=False)
    panel = ttk.Frame(root)
    panel.pack(side="left", fill="y")
    panel_canvas = tk.Canvas(panel, width=430, highlightthickness=0)
    panel_scroll = ttk.Scrollbar(panel, orient="vertical", command=panel_canvas.yview)
    panel_canvas.configure(yscrollcommand=panel_scroll.set)
    panel_scroll.pack(side="right", fill="y")
    panel_canvas.pack(side="left", fill="y", expand=True)
    controles = ttk.Frame(panel_canvas, padding=8)
    panel_canvas.create_window(0, 0, window=controles, anchor="nw", width=430)
    controles.bind("<Configure>", lambda _: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")))
    visor = ttk.Frame(root)
    visor.pack(side="right", fill="both", expand=True)
    canvas = tk.Canvas(visor, background="#666666")
    vertical = ttk.Scrollbar(visor, orient="vertical", command=canvas.yview)
    horizontal = ttk.Scrollbar(visor, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    vertical.pack(side="right", fill="y")
    horizontal.pack(side="bottom", fill="x")
    canvas.pack(fill="both", expand=True)
    indices = []
    foto = None
    inicio = None
    dibujo = None
    detalles = tk.StringVar()

    def seleccionado():
        seleccion = lista.curselection()
        if not seleccion:
            raise ValueError("Seleccione una detección")
        return indices[seleccion[0]]

    def zonas():
        if not rectangulo.get().strip():
            return None
        x0, y0, x1, y1 = map(float, rectangulo.get().split(","))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Rectángulo inválido")
        return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]

    def refrescar():
        nonlocal foto, dibujo
        confirmar.set(False)
        indices.clear()
        lista.delete(0, "end")
        rectangulo.set("")
        detalles.set("")
        for i, d in enumerate(s.detecciones):
            if d["page"] == pagina.get():
                indices.append(i)
                lista.insert("end", f'{i}: {d["category"]} | {d["decision"]} | {d.get("fragment_text", "manual")}')
        im = s.imagenes[pagina.get()-1] if original_visible.get() else s.imagen(pagina.get(), True)
        factor = escala.get()
        foto = ImageTk.PhotoImage(im.resize((max(1, round(im.width*factor)), max(1, round(im.height*factor)))))
        canvas.delete("all")
        canvas.create_image(0, 0, image=foto, anchor="nw")
        canvas.configure(scrollregion=(0, 0, foto.width(), foto.height()))
        dibujo = None
        error = " — OCR con error: revisión manual completa" if pagina.get() in s.errores else ""
        estado.set(f"Página {pagina.get()}/{len(s.imagenes)}. Revisadas: {len(s.revisadas)}{error}")

    def accion(fn):
        def ejecutar():
            try:
                fn()
                refrescar()
            except Exception as exc:
                messagebox.showerror("Revisión pendiente", str(exc), parent=root)
        return ejecutar

    def mostrar_dato(_=None):
        rectangulo.set("")
        canvas.delete("seleccion")
        try:
            d = s.detecciones[seleccionado()]
            detalles.set(f'Propuesta: {d.get("tarjar")}\nReglas: {d.get("rule_ids")}\nConfianza: {d.get("score")}\nGeometría: {d.get("location_precision")}\n{d.get("text", "Zona manual")}')
            for poly in d.get("polygons") or []:
                canvas.create_polygon(*[v*escala.get() for pt in poly for v in pt], outline="#00bfff", fill="", width=3, tags="seleccion")
        except ValueError:
            pass

    def coordenada(event):
        w, h = s.imagenes[pagina.get()-1].size
        return (min(w-1, max(0, canvas.canvasx(event.x)/escala.get())),
                min(h-1, max(0, canvas.canvasy(event.y)/escala.get())))

    def comenzar(event):
        nonlocal inicio, dibujo
        inicio = coordenada(event)
        if dibujo is not None:
            canvas.delete(dibujo)
        x, y = inicio
        dibujo = canvas.create_rectangle(x*escala.get(), y*escala.get(), x*escala.get(), y*escala.get(), outline="red", width=2)

    def arrastrar(event):
        if inicio is None:
            return
        x, y = coordenada(event)
        a, b = inicio
        canvas.coords(dibujo, a*escala.get(), b*escala.get(), x*escala.get(), y*escala.get())
        rectangulo.set(f"{min(a,x):.1f},{min(b,y):.1f},{max(a,x):.1f},{max(b,y):.1f}")

    def terminar(event):
        nonlocal inicio
        arrastrar(event)
        inicio = None

    def exportar():
        if not confirmar.get():
            raise ValueError("Complete la comprobación final")
        destino = filedialog.asksaveasfilename(parent=root, title="Guardar PDF aprobado", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="documento_tarjado.pdf")
        if destino:
            result = s.exportar(destino, revisor.get(), leyenda.get("1.0", "end").strip(), True)
            messagebox.showinfo("Exportado", f"PDF: {result}\nRegistro privado: {result.with_suffix('.revision_privada.json')}", parent=root)

    def cambiar_pagina(delta):
        pagina.set(max(1, min(len(s.imagenes), pagina.get()+delta)))

    ttk.Label(controles, text="Tipo de acto").pack(anchor="w")
    ttk.Combobox(controles, textvariable=tipo, values=TIPOS, state="readonly", width=42).pack(fill="x")
    ttk.Button(controles, text="Confirmar tipo (reinicia decisiones)", command=accion(lambda: s.clasificar(tipo.get()))).pack(fill="x")
    nav = ttk.Frame(controles); nav.pack(fill="x")
    ttk.Button(nav, text="Anterior", command=accion(lambda: cambiar_pagina(-1))).pack(side="left")
    ttk.Button(nav, text="Siguiente", command=accion(lambda: cambiar_pagina(1))).pack(side="right")
    ttk.Label(controles, textvariable=estado, wraplength=390).pack(fill="x")
    lista = tk.Listbox(controles, width=54, height=9, exportselection=False)
    lista.pack(fill="x"); lista.bind("<<ListboxSelect>>", mostrar_dato)
    ttk.Label(controles, textvariable=detalles, wraplength=390).pack(fill="x")
    for titulo, variable in [("Motivo", motivo), ("Rectángulo (o dibuja con el ratón)", rectangulo),
                              ("Categoría de zona manual", categoria), ("Regla de zona manual", regla)]:
        ttk.Label(controles, text=titulo).pack(anchor="w")
        ttk.Entry(controles, textvariable=variable).pack(fill="x")
    for titulo, fn in [
        ("Aceptar / ajustar tarjado", lambda: s.decidir(seleccionado(), "aceptar", motivo.get(), zonas())),
        ("Rechazar: conservar visible", lambda: s.decidir(seleccionado(), "rechazar", motivo.get())),
        ("Agregar zona manual", lambda: s.agregar(pagina.get(), zonas(), categoria.get(), regla.get(), motivo.get())),
        ("Marcar página revisada", lambda: s.marcar_pagina(pagina.get())),
    ]:
        ttk.Button(controles, text=titulo, command=accion(fn)).pack(fill="x")
    ttk.Checkbutton(controles, text="Ver original (sin marcas)", variable=original_visible, command=refrescar).pack(anchor="w")
    zoom = ttk.Combobox(controles, textvariable=escala, values=(0.25, 0.4, 0.6, 1.0), state="readonly")
    zoom.pack(fill="x"); zoom.bind("<<ComboboxSelected>>", lambda _: refrescar())
    ttk.Label(controles, text="Revisor").pack(anchor="w")
    ttk.Entry(controles, textvariable=revisor).pack(fill="x")
    ttk.Label(controles, text="Leyenda y fundamento acordados").pack(anchor="w")
    leyenda = tk.Text(controles, width=48, height=3); leyenda.pack(fill="x")
    ttk.Checkbutton(controles, text="Verifiqué cobertura, excepciones, QR y leyenda", variable=confirmar).pack(anchor="w")
    ttk.Button(controles, text="Exportar PDF aprobado", command=accion(exportar)).pack(fill="x")
    canvas.bind("<ButtonPress-1>", comenzar)
    canvas.bind("<B1-Motion>", arrastrar)
    canvas.bind("<ButtonRelease-1>", terminar)
    refrescar()
    root.mainloop()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="C: revisar localmente y exportar un PDF tarjado")
    parser.add_argument("original", type=Path)
    parser.add_argument("ocr", type=Path)
    parser.add_argument("detecciones", type=Path)
    parser.add_argument("--automatico", type=Path, metavar="PDF_DE_SALIDA",
                        help="Exportar una copia dudosa sin abrir la ventana")
    parser.add_argument("--sin-recuadros", action="store_true",
                        help="Omitir recuadros naranjas de pendientes en la copia automática")
    args = parser.parse_args(argv)
    try:
        sesion = Sesion(args.original, args.ocr, args.detecciones)
        if args.automatico is None:
            revisar_pc(sesion)
        else:
            args.automatico.parent.mkdir(parents=True, exist_ok=True)
            print(sesion.exportar_automatico(args.automatico, recuadros_pendientes=not args.sin_recuadros))
    except (ImportError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
