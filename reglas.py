"""Política técnica: propuestas, nunca aprobación automática."""
import re

VERSION = "2026-11-26.2"
TIPOS = ("sin_clasificar", "destinacion_ffaa", "arriendo", "titulo_gratuito", "saneamiento", "otro")
REGLAS = {
    "run_rut": ["RNA-05", "RNA-08", "RNA-14"],
    "run_rut_personal": ["RNA-05", "RNA-08"],
    "run_rut_juridico": ["RNA-14"],
    "run_rut_revisar": ["RNA-05", "RNA-08", "RNA-14"],
    "nacionalidad": ["RNA-06"], "estado_civil": ["RNA-06"],
    "fecha_nacimiento": ["RNA-06", "RNA-15"],
    "direccion_personal": ["RNA-05", "RNA-16"],
    "direccion_inmueble": ["RNA-09", "RNA-14", "RNA-16"],
    "direccion_institucional": ["RNA-09", "RNA-14"],
    "persona": ["RNA-07", "RNA-13", "RNA-14"],
    "persona_revisar": ["RNA-07", "RNA-13", "RNA-14"],
    "plano": ["RNA-09"], "deslindes": ["RNA-09"],
    "recinto": ["RNA-10"], "dato_pendiente": ["RNA-13"],
    "codigo_validacion": ["RNA-11", "RNA-24"],
    "codigo_validacion_revisar": ["RNA-11", "RNA-24"],
    "qr": ["RNA-11", "RNA-24"], "firma": ["RNA-05", "RNA-14"],
    "fotografia": ["RNA-05"], "huella": ["RNA-05"],
}

def aplicar(deteccion, tipo="sin_clasificar"):
    d = dict(deteccion)
    cat = d["category"]
    if tipo == "sin_clasificar":
        tipo = d.get("tipo_acto", tipo)
    d["tipo_acto"] = tipo
    reglas = REGLAS.get(cat, ["RNA-05"] if cat.startswith(("correo", "telefono", "dato_")) else [])
    if d.get("classification_basis") == "repeticion_identica_confirmada":
        reglas = list(dict.fromkeys(reglas + ["RNA-12"]))
    d.update(rule_ids=reglas, rules_version=VERSION, decision="pendiente")
    # La proximidad de una palabra no demuestra titularidad ni condición de menor.
    if cat in {"run_rut", "persona", "persona_revisar", "firma", "dato_pendiente", "recinto"}:
        d["tarjar"] = "revisar"
    if cat in {"direccion_inmueble", "plano", "deslindes"}:
        d["tarjar"] = "si" if tipo == "destinacion_ffaa" else "revisar" if tipo == "sin_clasificar" else "no"
    if cat == "direccion_institucional":
        d["tarjar"] = "revisar" if tipo in {"destinacion_ffaa", "sin_clasificar"} else "no"
    if cat == "nacionalidad":
        d["tarjar"] = "revisar"  # Confirmar que describe a una persona natural.
    if d.get("classification_basis") in {"conflicto_contextual_documental", "revisar_contexto_repeticion"}:
        d["tarjar"] = "revisar"
    elif cat in {"run_rut_personal", "codigo_validacion"}:
        # Mantener la revisión si una repetición todavía no tiene evidencia confirmada.
        if d.get("classification_basis") != "revisar_contexto_repeticion":
            d["tarjar"] = "si"
    elif cat == "run_rut_juridico":
        d["tarjar"] = "no"
    if d.get("location_precision") not in {"word_polygons", "character_boxes_union", "manual"}:
        d["requires_geometry_review"] = True
    return d

def candidatos_adicionales(text):
    """Reglas conservadoras con offsets originales; no infieren causales legales."""
    patrones = {
        "nacionalidad": r"\bnacionalidad\s+([\wáéíóúñ-]+)",
        "plano": r"\bplano\s+(?:n[°ºo.]?\s*)?([\w./-]+)",
        "codigo_validacion": r"\bc[oó]digo\s+(?:de\s+)?(?:validaci[oó]n|verificaci[oó]n)\s*[:：]?\s*([\w-]+)",
        "dato_pendiente": r"\b(?:profesi[oó]n|garant[ií]a|monto\s+adeudado)\s*[:：]?\s*([^\n;,]{1,70})",
    }
    for cat, patron in patrones.items():
        for m in re.finditer(patron, text, re.I):
            yield m.start(1), m.end(1), cat
