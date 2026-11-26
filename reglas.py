"""Política de propuestas de tarjado. Única fuente de decisiones, no aprobación."""
import re

VERSION = "2026-11-26.4"
TIPOS = ("sin_clasificar", "destinacion_ffaa", "arriendo", "titulo_gratuito", "saneamiento", "otro")
REGLAS = {
    "identificador_administrativo": ["RNA-14", "RNA-15"],
    "run_rut": ["RNA-05", "RNA-08", "RNA-14"],
    "run_rut_personal": ["RNA-05", "RNA-08"],
    "run_rut_juridico": ["RNA-14"],
    "run_rut_revisar": ["RNA-05", "RNA-08", "RNA-14"],
    "nacionalidad": ["RNA-06"], "nacionalidad_personal": ["RNA-06"],
    "estado_civil": ["RNA-06"], "estado_civil_revisar": ["RNA-06"],
    "fecha_nacimiento": ["RNA-06", "RNA-15"],
    "direccion_personal": ["RNA-05", "RNA-16"],
    "direccion_revisar": ["RNA-05", "RNA-09", "RNA-16"],
    "direccion_inmueble": ["RNA-09", "RNA-14", "RNA-16"],
    "direccion_institucional": ["RNA-09", "RNA-14"],
    "persona": ["RNA-07", "RNA-13", "RNA-14"],
    "persona_revisar": ["RNA-07", "RNA-13", "RNA-14"],
    "persona_funcionario": ["RNA-14"], "persona_menor": ["RNA-07"],
    "dato_menor": ["RNA-07"], "dato_menor_revisar": ["RNA-07"],
    "ubicacion_inmueble_revisar": ["RNA-09", "RNA-14"],
    "plano": ["RNA-09"], "deslindes": ["RNA-09"],
    "recinto": ["RNA-10"], "dato_pendiente": ["RNA-13"],
    "codigo_validacion": ["RNA-11", "RNA-24"],
    "codigo_validacion_revisar": ["RNA-11", "RNA-24"],
    "vinculo_validacion": ["RNA-24"], "qr": ["RNA-11", "RNA-24"],
    "firma": ["RNA-05", "RNA-14"], "fotografia": ["RNA-05"], "huella": ["RNA-05"],
    "dato_salud": ["RNA-05", "RNA-07"], "dato_salud_personal": ["RNA-05", "RNA-07"],
}
_SI = frozenset({"run_rut_personal", "direccion_personal", "correo_personal", "telefono_personal",
                 "estado_civil", "fecha_nacimiento", "nacionalidad_personal", "persona_menor", "dato_menor",
                 "dato_financiero_personal", "dato_previsional_personal", "dato_salud_personal",
                 "codigo_validacion", "vinculo_validacion", "qr", "fotografia", "huella"})
_NO = frozenset({"identificador_administrativo", "run_rut_juridico", "persona_funcionario", "correo_institucional", "telefono_institucional"})
_CONFLICTOS = {"conflicto_contextual_documental", "revisar_contexto_repeticion", "titularidad_contradictoria"}

def politica(categoria, tipo="sin_clasificar", *, base="", flags=()):
    """La categoría y su evidencia deciden; nunca el score de GLiNER por sí solo."""
    if base in _CONFLICTOS or "conflicto_solapamiento" in flags:
        return "revisar"
    if categoria in _SI:
        return "si"
    if categoria in _NO or categoria.startswith("ignorar:"):
        return "no"
    if categoria in {"direccion_inmueble", "plano", "deslindes"}:
        return "si" if tipo == "destinacion_ffaa" else "revisar" if tipo == "sin_clasificar" else "no"
    if categoria in {"direccion_institucional", "ubicacion_inmueble_revisar"}:
        return "revisar" if tipo in {"destinacion_ffaa", "sin_clasificar"} else "no"
    return "revisar"

def aplicar(deteccion, tipo="sin_clasificar"):
    d = dict(deteccion)
    cat = d["category"]
    if tipo == "sin_clasificar":
        tipo = d.get("tipo_acto", tipo)
    d["tipo_acto"] = tipo
    reglas = REGLAS.get(cat, ["RNA-05"] if cat.startswith(("correo", "telefono", "dato_")) else [])
    if d.get("classification_basis") in {"repeticion_identica_confirmada", "repeticion_menor_confirmada"}:
        reglas = list(dict.fromkeys(reglas + ["RNA-12"]))
    d.update(rule_ids=reglas, rules_version=VERSION, decision="pendiente",
             tarjar=politica(cat, tipo, base=d.get("classification_basis", ""), flags=d.get("flags") or []))
    if d.get("location_precision") not in {"word_polygons", "character_boxes_union", "manual"}:
        d["requires_geometry_review"] = True
    return d

def candidatos_adicionales(text):
    """Candidatos básicos; el contexto más amplio se analiza en B."""
    patrones = {
        "plano": r"\bplano\s+(?:(?:n[°ºo.]|n[uú]mero)\s*)?[:：]?\s*([\w./-]*\d[\w./-]*)",
        "codigo_validacion": r"\bc[oó]digo\s+(?:de\s+)?(?:validaci[oó]n|verificaci[oó]n)\s*[:：]?\s*([\w-]+)",
        "dato_pendiente": r"\b(?:profesi[oó]n|(?:n[uú]mero\s+de\s+)?garant[ií]a|monto\s+adeudado)\s*[:：]?\s*([^\n;,]{1,70})",
    }
    for cat, patron in patrones.items():
        for m in re.finditer(patron, text, re.I):
            yield m.start(1), m.end(1), cat
