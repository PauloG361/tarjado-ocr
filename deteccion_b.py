"""Genera versiones nuevas de deteccion_b.py y reglas.py a partir del repositorio.
No modifica los originales. Ejecutar: python actualizar_deteccion.py RUTA_REPO RUTA_SALIDA
"""
import argparse
import ast
from pathlib import Path


RUT_HELPER = r'''
# RNA-05/08/14: la validez del DV no identifica al titular.
RUT_TITULAR_RE = re.compile(
    r"(?P<label>\brun\b|\brut\b|\bcedula\s+de\s+identidad\b|\bc\.i\.)"
    r"\s*(?P<owner>(?:d(?:e|el)\s+)?(?:la\s+|el\s+)?"
    r"(?:persona\s+natural|persona\s+juridica|representante(?:\s+legal)?|"
    r"empresa|sociedad|corporacion|fundacion|personal|particular))?"
    r"\s*(?:n[°ºo.]?\s*)?[:=]?\s*$"
)
RUT_OWNER_PUBLIC = {"empresa", "sociedad", "persona juridica", "corporacion", "fundacion"}
RUT_OWNER_PRIVATE = {"representante", "representante legal", "persona natural", "personal", "particular"}


def clasificar_rut(text, s, e, valido):
    izquierda = fold(text[max(0, s - 100):s])
    match = RUT_TITULAR_RE.search(izquierda)
    derecha = fold(text[e:e + 75])
    posterior_publico = bool(re.match(r"\s*\(\s*(?:empresa|persona juridica|sociedad)\s*\)", derecha))
    posterior_personal = bool(re.match(r"\s*\(\s*(?:representante legal|persona natural|personal)\s*\)", derecha))
    k = clave_rut(text[s:e])
    cuerpo = int(k[:-1]) if k[:-1].isdigit() else 0
    rango = "natural_compatible" if 0 < cuerpo < 30000000 else "juridico_compatible" if cuerpo >= 50000000 else "indeterminado"
    natural, juridico = posterior_personal, posterior_publico
    if match:
        owner = re.sub(r"^(?:de(?:l)?\s+)?(?:la\s+|el\s+)?", "", (match.group("owner") or "").strip())
        juridico |= owner in RUT_OWNER_PUBLIC
        natural |= owner in RUT_OWNER_PRIVATE or match.group("label") in {"run", "cedula de identidad", "c.i."}
        # Contexto adicional acotado: no cruzar otro identificador o fin de cláusula.
        previo = izquierda[:match.start()]
        previo = re.split(r"[;\n]|(?<=[.!?])\s+|\b(?:rut|run)\b", previo)[-1]
        if not owner and not natural and not juridico:
            natural = rango == "natural_compatible" and bool(re.search(r"\b(?:representad[oa]\s+(?:legalmente\s+)?por|representante\s+legal|persona\s+natural|don|dona)\b", previo))
            juridico = rango == "juridico_compatible" and bool(re.search(r"\b(?:spa|ltda|sociedad|empresa|fundacion|corporacion|s\.a\.)\b", previo)) and not re.search(r"\b(?:representad[oa]|representante)\b", previo)
    if natural and juridico:
        return "run_rut_revisar", "titularidad_contradictoria"
    if natural:
        return "run_rut_personal", ("titular_natural_explicito" if valido else "titular_natural_dv_invalido") + ":" + rango
    if juridico and valido:
        return "run_rut_juridico", "titular_juridico_explicito:" + rango
    return "run_rut_revisar", ("dv_valido_titular_no_confirmado" if valido else "dv_invalido_o_error_ocr") + ":" + rango


def clave_rut(valor):
    return re.sub(r"[^0-9K]", "", valor.upper())

'''

REPETICIONES = r'''
def consolidar_repeticiones_seguras(detecciones):
    """RNA-12: heredar solo identidad confirmada sin evidencia opuesta.
    Cada aparición conserva sus coordenadas, precisión y confianza originales.
    No convierte datos genéricos ni números con DV inválido en datos confirmados.
    """
    categorias = {
        "run_rut_personal": ("rut", "si"), "run_rut_juridico": ("rut", "no"),
        "run_rut_revisar": ("rut", None), "run_rut": ("rut", None),
        "correo_personal": ("correo", "si"), "correo_institucional": ("correo", "no"),
        "correo_revisar": ("correo", None),
        "telefono_personal": ("telefono", "si"), "telefono_institucional": ("telefono", "no"),
        "telefono_revisar": ("telefono", None),
        "direccion_personal": ("direccion", "si"), "direccion_inmueble": ("direccion", "no"),
        "direccion_institucional": ("direccion", "no"), "direccion_revisar": ("direccion", None),
    }
    grupos = {}
    for d in detecciones:
        cat = d.get("category")
        if cat not in categorias:
            continue
        familia, _ = categorias[cat]
        valor = d.get("text", "")
        if familia == "rut":
            k = clave_rut(valor)
            if len(k) not in (8, 9) or not k[:-1].isdigit() or _rut_dv(k[:-1]) != k[-1]:
                continue
        elif familia == "telefono":
            k = re.sub(r"\D", "", valor)
            if len(k) < 8:
                continue
        elif familia == "correo":
            k = re.sub(r"\s+", "", valor).lower()
            if "@" not in k:
                continue
        else:
            k = " ".join(fold(valor).split())
            if len(k) < 8 or not any(c.isdigit() for c in k):
                continue
        grupos.setdefault((d["source_file"], familia, k), []).append(d)
    for grupo in grupos.values():
        # Las repeticiones sintéticas y las coincidencias contextuales débiles no son evidencia independiente.
        semillas = [d for d in grupo if categorias[d["category"]][1] == "si"
                    and d.get("tarjar") == "si"
                    and "consistencia_documental" not in d.get("sources", [])]
        opuestas = [d for d in grupo if categorias[d["category"]][1] == "no"
                   and "consistencia_documental" not in d.get("sources", [])]
        if opuestas or any(d.get("classification_basis") == "titularidad_contradictoria" for d in grupo):
            # Ante conflicto no reafirmar automáticamente una categoría personal.
            if semillas:
                for d in grupo:
                    if categorias[d["category"]][1] != "no" or "consistencia_documental" in d.get("sources", []):
                        d["tarjar"] = "revisar"
                        d["classification_basis"] = "conflicto_contextual_documental"
            continue
        if not semillas:
            continue
        semilla = max(semillas, key=lambda d: d.get("score", 0))
        for d in grupo:
            # Un RUN válido sin titular o un dato idéntico de contexto incierto puede heredar la evidencia.
            # No promover conflictos de titularidad ni casos ya resueltos explícitamente.
            if d.get("classification_basis") == "titularidad_contradictoria":
                continue
            if categorias[d["category"]][1] is None or d.get("tarjar") == "revisar":
                d["category"] = semilla["category"]
                d["tarjar"] = "si"
                d["classification_basis"] = "repeticion_identica_confirmada"
                d["sources"] = list(dict.fromkeys(d.get("sources", []) + ["consistencia_documental"]))
                d["rule_ids"] = list(dict.fromkeys(aplicar(d).get("rule_ids", []) + ["RNA-12"]))
                d["repetition_seed"] = semilla.get("entity_id")

'''


TIPO_HELPER = r'''
def clasificar_acto(paginas):
    """Solo encabezado de primera página. No inferir del cuerpo del acto."""
    if not paginas:
        return "sin_clasificar", "sin_paginas"
    p = min(paginas, key=lambda x: x.get("page_number", 1))
    if p.get("page_number") != 1 or p.get("processing_error"):
        return "sin_clasificar", "primera_pagina_no_disponible"
    lineas = p.get("ocr_lines", [])
    alto = p.get("image_height_px", 0)
    encabezado = []
    for l in lineas[:25]:
        caja = l.get("bbox_xyxy")
        if alto and caja and caja[1] > alto * 0.30:
            continue
        linea = fold(str(l.get("text", "")))
        if re.match(r"\s*(?:vistos|considerando[s]?|resuelvo|decreto\s*:|se\s+resuelve)\b", linea):
            break
        encabezado.append(linea)
    texto = " ".join(encabezado)
    tipos = []
    if re.search(r"\b(?:destinacion|destina)\b", texto) and re.search(r"\b(?:fuerzas\s+armadas|ff\.?\s*aa\.?|ejercito|armada|fuerza\s+aerea|carabineros|policia\s+de\s+investigaciones|orden\s+y\s+seguridad\s+publica)\b", texto):
        tipos.append("destinacion_ffaa")
    for tipo, patron in (("arriendo", r"\b(?:arriendo|arrendamiento)\b"), ("titulo_gratuito", r"\btitulo\s+gratuito\b"), ("saneamiento", r"\bsaneamiento\b")):
        if re.search(patron, texto):
            tipos.append(tipo)
    return (tipos[0], "encabezado_explicito") if len(tipos) == 1 else ("sin_clasificar", "encabezado_ambiguo" if tipos else "encabezado_sin_tipo")
'''


def reemplazar(s, antes, despues):
    if s.count(antes) != 1:
        raise ValueError(f"Versión distinta del código esperado: {antes[:100]!r}")
    return s.replace(antes, despues, 1)


def modificar_b(s):
    r = reemplazar
    s = r(s, '    "run_rut": "si",', '    "run_rut": "revisar",\n    "run_rut_personal": "si",\n    "run_rut_juridico": "no",\n    "codigo_validacion": "si",')
    s = r(s, 'def find_ruts(text):', RUT_HELPER + '\ndef find_ruts(text):')
    s = r(s, '        cat = "run_rut" if m["valid"] else "run_rut_revisar"\n        ents.append(mk(text, m["s"], m["e"], cat, m["score"], ["regex"], "dv_valido" if m["valid"] else "dv_invalido"))', '        cat, basis = clasificar_rut(text, m["s"], m["e"], m["valid"])\n        ents.append(mk(text, m["s"], m["e"], cat, m["score"], ["regex"], basis))')
    s = r(s, '                ents.append(mk(text, s, e, "run_rut" if valid else "run_rut_revisar",\n                               score * (0.9 if valid else 0.7), ["gliner"], "solo_gliner"))', '                rs, re_ = s + mm.start(), s + mm.end()\n                cat, basis = clasificar_rut(text, rs, re_, valid)\n                ents.append(mk(text, rs, re_, cat, score * (0.9 if valid else 0.7), ["gliner"], basis))')
    # A code explicitly identified by the contextual rule should have a distinct basis.
    s = r(s, '        ents.append(mk(text, s, e, cat, 0.5, ["regla_contextual"], "requiere_confirmacion"))', '        if cat == "codigo_validacion" and not (len(text[s:e]) >= 4 and any(c.isdigit() for c in text[s:e])):\n            cat = "codigo_validacion_revisar"\n        basis = "codigo_validacion_explicito" if cat == "codigo_validacion" else "requiere_confirmacion"\n        ents.append(mk(text, s, e, cat, 0.95 if cat == "codigo_validacion" else 0.5, ["regla_contextual"], basis))')
    s = r(s, 'def procesar(entrada, salida=None, device="cpu", ner_es=False, modelo_ner_es=None):', REPETICIONES + '\ndef procesar(entrada, salida=None, device="cpu", ner_es=False, modelo_ner_es=None):')
    s = r(s, 'def procesar(entrada, salida=None, device="cpu", ner_es=False, modelo_ner_es=None):', TIPO_HELPER + '\n' + 'def procesar(entrada, salida=None, device="cpu", ner_es=False, modelo_ner_es=None):')
    s = r(s, '    OCR_PAGES_INTERNAL = []', '''    tipos_documento = {}
    for nombre in {p.get("source_file") for p in OCR_DATA["pages"]}:
        tipo, evidencia = clasificar_acto([p for p in OCR_DATA["pages"] if p.get("source_file") == nombre])
        tipos_documento[nombre] = {"tipo": tipo, "evidencia": evidencia}
        print("Tipo propuesto:", nombre, tipo, evidencia, flush=True)
    OCR_PAGES_INTERNAL = []''')
    s = r(s, '            ALL_DETECTIONS.extend(detections)', '''            for d in detections:
                d["tipo_acto"] = tipos_documento[d["source_file"]]["tipo"]
                d.update(aplicar(d))
            ALL_DETECTIONS.extend(detections)''')
    s = r(s, '                    d["tarjar"] = "revisar"\n                    d["entity_id"] += "-repeticion"', '                    d["tarjar"] = "revisar"\n                    d["tipo_acto"] = tipos_documento[document_name]["tipo"]\n                    d["entity_id"] += "-repeticion"')
    s = r(s, '    DETECTIONS_OUTPUT = {', '    consolidar_repeticiones_seguras(ALL_DETECTIONS)\n    ALL_DETECTIONS = [aplicar(d) for d in ALL_DETECTIONS]\n\n    DETECTIONS_OUTPUT = {\n        "tipos_documento": tipos_documento,')
    return s


def modificar_reglas(s):
    r = reemplazar
    s = r(s, 'VERSION = "2026-11-23.1"', 'VERSION = "2026-11-26.2"')
    s = r(s, '    "run_rut": ["RNA-05", "RNA-08", "RNA-14"],', '    "run_rut": ["RNA-05", "RNA-08", "RNA-14"],\n    "run_rut_personal": ["RNA-05", "RNA-08"],\n    "run_rut_juridico": ["RNA-14"],')
    s = r(s, '    "codigo_validacion": ["RNA-11", "RNA-24"],', '    "codigo_validacion": ["RNA-11", "RNA-24"],\n    "codigo_validacion_revisar": ["RNA-11", "RNA-24"],')
    s = r(s, '    d.update(rule_ids=reglas, rules_version=VERSION, decision="pendiente")', '    if d.get("classification_basis") == "repeticion_identica_confirmada":\n        reglas = list(dict.fromkeys(reglas + ["RNA-12"]))\n    d.update(rule_ids=reglas, rules_version=VERSION, decision="pendiente")')
    s = r(s, '    if d.get("location_precision") not in {"word_polygons", "character_boxes_union", "manual"}:', '    if d.get("classification_basis") == "conflicto_contextual_documental":\n        d["tarjar"] = "revisar"\n    elif cat in {"run_rut_personal", "codigo_validacion"}:\n        # Mantener la revisión si una repetición todavía no tiene evidencia confirmada.\n        if d.get("classification_basis") != "revisar_contexto_repeticion":\n            d["tarjar"] = "si"\n    elif cat == "run_rut_juridico":\n        d["tarjar"] = "no"\n    if d.get("location_precision") not in {"word_polygons", "character_boxes_union", "manual"}:')
    s = r(s, '    cat = d["category"]', '    cat = d["category"]\n    if tipo == "sin_clasificar":\n        tipo = d.get("tipo_acto", tipo)\n    d["tipo_acto"] = tipo')
    s = r(s, '    if d.get("classification_basis") == "conflicto_contextual_documental":', '    if d.get("classification_basis") in {"conflicto_contextual_documental", "revisar_contexto_repeticion"}:')
    s = r(s, 'if cat in {"direccion_inmueble", "direccion_institucional", "plano", "deslindes"}:', 'if cat in {"direccion_inmueble", "plano", "deslindes"}:')
    s = r(s, '    if cat == "nacionalidad":', '    if cat == "direccion_institucional":\n        d["tarjar"] = "revisar" if tipo in {"destinacion_ffaa", "sin_clasificar"} else "no"\n    if cat == "nacionalidad":')
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("salida", type=Path)
    a = parser.parse_args()
    if a.repo.resolve() == a.salida.resolve():
        parser.error("La salida debe ser otra carpeta")
    archivos = {"deteccion_b.py": modificar_b((a.repo / "deteccion_b.py").read_text(encoding="utf-8")),
                "reglas.py": modificar_reglas((a.repo / "reglas.py").read_text(encoding="utf-8"))}
    for nombre, codigo in archivos.items():
        ast.parse(codigo, filename=nombre)
        if (a.salida / nombre).exists():
            raise FileExistsError(a.salida / nombre)
    a.salida.mkdir(parents=True, exist_ok=True)
    for nombre, codigo in archivos.items():
        (a.salida / nombre).write_text(codigo, encoding="utf-8")
        print("Generado:", a.salida / nombre)


if __name__ == "__main__":
    main()
