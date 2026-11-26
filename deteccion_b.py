"""Etapa B: detección local sobre el JSON de OCR; no altera el PDF."""
import argparse
import re
import json
import hashlib
import math
import traceback
from time import perf_counter
from collections import Counter
from reportes_csv import COLUMNAS, fila_base, escribir_csv
import unicodedata
import html as _html
from pathlib import Path
from reglas import aplicar, politica, candidatos_adicionales, VERSION as RULES_VERSION
from ocr_a import ejecutar_etapa

ner_model = None
es_pipe = None

GAZETTEER_FILE = Path(__file__).with_name("comunas_chile.txt")

NER_MODEL_NAME = "urchade/gliner_multi_pii-v1"

GLINER_PASSES = {
    "P1_personas": {
        "person": "persona",
        "organization": "ignorar:organizacion",
        "government institution": "ignorar:institucion",
        "job title": "ignorar:cargo",
        "location": "ignorar:lugar",
    },
    "P2_direcciones": {
        "address": "direccion",
        "location": "ignorar:lugar",
    },
    "P3_identificadores": {
        "national id number": "rut",
        "identity card number": "rut",
        "tax identification number": "rut",
        "email": "correo",
    },
    "P4_sensibles": {
        "date of birth": "fecha_nacimiento",
        "bank account number": "dato_financiero",
        "health insurance number": "dato_previsional",
        "medical condition": "dato_salud",
        "medical treatment": "dato_salud",
        "disability": "dato_salud",
        "health insurance provider": "dato_salud",
        "bank balance": "dato_financiero",
        "personal income": "dato_financiero",
        "pension fund": "dato_previsional",
        "pension contribution": "dato_previsional",
        "phone number": "telefono",
        "mobile phone number": "telefono",
    },
    "P5_contexto": {
        "nationality": "nacionalidad",
        "property boundary": "deslindes",
        "military facility name": "recinto",
        "profession": "dato_pendiente",
        "age": "dato_menor_revisar",
        "school name": "dato_menor_revisar",
    },
}

GLINER_LABELS = {}

for _labels in GLINER_PASSES.values():
    GLINER_LABELS.update(_labels)

GLINER_THRESHOLDS = {   # umbral por etiqueta (no son comparables entre sí)
    "person": 0.35, "address": 0.30, "email": 0.30,
    "national id number": 0.40, "identity card number": 0.40, "tax identification number": 0.40,
    "organization": 0.30, "government institution": 0.30, "job title": 0.30, "location": 0.30,
    "date of birth": 0.35, "bank account number": 0.40, "health insurance number": 0.40,
    "medical condition": 0.40, "phone number": 0.40, "mobile phone number": 0.40,
}

for _label in ("medical treatment", "disability", "health insurance provider", "bank balance",
               "personal income", "pension fund", "pension contribution", "age", "school name"):
    GLINER_THRESHOLDS[_label] = 0.40

GLINER_MIN_THRESHOLD = min(GLINER_THRESHOLDS.values())

SPANISH_NER_MODEL = "PlanTL-GOB-ES/roberta-base-bne-capitel-ner"

SPA_VETO_SCORE = 0.60              # ORG/LOC del NER español con este puntaje veta una persona débil

PERSON_STRONG = 0.80               # una persona de GLiNER con puntaje >= esto no se veta por el NER español

SPA_PERSON_ADD_SCORE = 0.85        # PER del NER español que GLiNER no vio -> persona_revisar

ACEPTAR_PERSONA_SOLO_NER_ES = True

BIRTH_CUE_DISTANCE = 40            # caracteres a la izquierda de una fecha donde debe estar el cue de nacimiento

ID_MIN_DIGITS = 6                  # mínimo de dígitos para aceptar un nº de cuenta / previsional de GLiNER

PHONE_MIN_SCORE = 0.45

CUE_MAX_DISTANCE = 80              # caracteres entre un cue de contexto y la entidad

ADDRESS_CUE_DISTANCE = 140

PROPAGAR_NOMBRES = False  # V4: no multiplicar nombres aún no confirmados

PROPAGATE_MIN_SCORE = 0.55

PROPAGAR_VARIANTES = True          # primer+segundo, último+penúltimo y primero+último de nombres de >= 3 palabras

PROPAGAR_TOKENS_SUELTOS = False    # propagar palabras sueltas (apellidos) como persona_revisar

SHOW_NO_IN_REVIEW = True

BOX_PADDING_PX = 4

INSTITUTION_WORDS = {
    "ministerio", "municipalidad", "subsecretaria", "secretaria", "division", "departamento", "servicio",
    "gobierno", "corporacion", "direccion", "oficina", "tesoreria", "contraloria", "conservador", "juzgado",
    "republica", "intendencia", "gobernacion", "seremi", "fiscalia", "hacienda", "bienes", "nacionales",
    "vivienda", "urbanismo", "registro", "notaria",
}

CARGO_WORDS = {
    "ministro", "ministra", "subsecretario", "subsecretaria", "seremi", "alcalde", "alcaldesa", "jefe", "jefa",
    "director", "directora", "secretario", "secretaria", "fiscal", "juez", "intendente", "gobernador",
}

NON_PERSON_WORDS = {
    "arrendatario",
    "arrendataria",
    "arrendatarios",
    "arrendatarias",
    "solicitante",
    "solicitantes",
    "domicilio",
    "domiciliado",
    "domiciliada",
    "arrendador",
    "arrendadora",
    "propietario",
    "propietaria",
    "representante",
    "interesado",
    "interesada",
    "compareciente",
    "beneficiario",
    "beneficiaria",
    "inmueble",
    "terreno",
    "persona",
    "fisco",
    "estado",
    "municipalidad",
}

def fold(text):
    """Minúsculas sin acentos, conservando el largo (los índices siguen valiendo)."""
    out = []
    for c in text:
        base = unicodedata.normalize("NFD", c)[0]
        low = base.lower()
        out.append(low if len(low) == 1 else base)
    return "".join(out)

def mask_spans(text, spans):
    chars = list(text)
    for a, b in spans:
        for k in range(max(0, a), min(len(chars), b)):
            chars[k] = " "
    return "".join(chars)

def _ov(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1

def _overlaps_any(a0, a1, spans):
    return any(_ov(a0, a1, b0, b1) for b0, b1 in spans)

def _c(patterns):
    return [re.compile(p) for p in patterns]

CUES_PHONE = {
    "personal": _c([r"\b(?:celular|cel|movil|telefono|fono|contacto|numero)\s+(?:personal|particular)\b"]),
    "institucional": _c([
        r"\bmesa\s+central\b", r"\bcentral\s+telefonica\b",
        r"\b(?:telefono|fono|contacto)\s+(?:institucional|oficina|laboral|trabajo)\b",
        r"\banexo\b", r"\bsecretaria\b", r"\boficina\s+de\s+partes\b",
        r"\boficina\b(?!\s*(?:n[o°º]?\.?\s*)?\d)",
    ]),
}

CUES_EMAIL = {
    "personal": _c([r"\b(?:correo|email|e-mail|mail)\s+(?:personal|particular)\b"]),
    "institucional": _c([r"\b(?:correo|email|e-mail|mail)\s+(?:institucional|laboral|de\s+trabajo|oficina)\b"]),
}

CUES_ADDRESS = {   # el orden resuelve empates: personal > inmueble > institucional
    "personal": _c([r"\bdomicili\w*", r"\bresidente\b", r"\breside\b", r"\bvive\b",
                    r"\bdireccion\s+particular\b", r"\bdomicilio\s+particular\b"]),
    "inmueble": _c([r"\bterreno\b", r"\binmueble\b", r"\bpredio\b", r"\bretazo\b", r"\bpropiedad\s+fiscal\b",
                    r"\blote\b", r"\brol\s+de\s+avaluo\b", r"\binscrit[oa]\b", r"\bsuperficie\b",
                    r"\bdeslinde\w*", r"\barea\s+verde\b", r"\bconjunto\s+residencial\b", r"\bsitio\b", r"\bparcela\b"]),
    "institucional": _c([r"\boficina\w*", r"\bsede\b", r"\bedificio\b", r"\bministerio\b", r"\bmunicipalidad\b",
                         r"\bsecretaria\b", r"\bservicio\b", r"\bsucursal\b", r"\bdireccion\s+regional\b",
                         r"\bgobierno\b", r"\bintendencia\b", r"\bseremi\b"]),
}

PHONE_CUE_RE = re.compile(r"\b(?:telefono|tel|fono|fax|celular|cel|movil|whatsapp|contacto|anexo)\b")

EMAIL_CUE_RE = re.compile(r"\b(?:correo|e-mail|email|mail)\b")

RUT_CUE_RE = re.compile(r"\b(?:rut|run|cedula|c\.i)\b")

NUMBER_PREFIX_RE = re.compile(r"(?:n[°º]|no\.|ord\.?|oficio|folio|resolucion|exento|decreto|rol)\s*$")

def nearest_group(folded, s, e, groups, max_dist):
    """Nombre del grupo de cues más cercano a [s, e) o None."""
    best = None
    for name, rxs in groups.items():
        for rx in rxs:
            for m in rx.finditer(folded):
                if m.end() <= s:
                    d = s - m.end()
                elif m.start() >= e:
                    d = m.start() - e
                else:
                    d = 0
                if best is None or d < best[0]:
                    best = (d, name)
    return best[1] if best and best[0] <= max_dist else None

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "hotmail.cl", "outlook.com", "outlook.cl", "live.com", "live.cl",
    "yahoo.com", "yahoo.cl", "icloud.com", "msn.com", "proton.me", "protonmail.com", "gmx.com",
}

INSTITUTIONAL_EMAIL_DOMAINS = {"gob.cl"}     # acepta subdominios; agrega los dominios oficiales reales

def _domain_matches(domain, domains):
    return any(domain == d or domain.endswith("." + d) for d in domains)

EMAIL_RE = re.compile(r"[\w.%+\-]+\s*@\s*[\w\-]+(?:\.[\w\-]+)*\.[A-Za-z]{2,}")

def find_emails(text):
    out = []
    ftext = fold(text)
    for m in EMAIL_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s] in ".-%+":
            s += 1
        score = 0.85 + (0.10 if EMAIL_CUE_RE.search(ftext[max(0, s - 25):s]) else 0.0)
        out.append({"s": s, "e": e, "score": min(score, 0.99)})
    return out

def classify_email(value, fmasked, s, e):
    domain = re.sub(r"\s+", "", value.split("@")[-1]).lower()
    kind = nearest_group(fmasked, s, e, CUES_EMAIL, CUE_MAX_DISTANCE)
    if kind == "personal":
        return "correo_personal", "contexto"
    if kind == "institucional":
        return "correo_institucional", "contexto"
    if _domain_matches(domain, PERSONAL_EMAIL_DOMAINS):
        return "correo_revisar", "dominio_personal_sin_titular"
    if _domain_matches(domain, INSTITUTIONAL_EMAIL_DOMAINS):
        return "correo_institucional", "dominio_institucional"
    return "correo_revisar", "sin_evidencia"

RUT_RE = re.compile(r"(?<![\w.,])(\d{1,2}(?:[.,]\d{3}){2}|\d{7,8})\s*[-–—]?\s*([\dkK])(?![\w])")

def _rut_dv(body):
    total, mult = 0, 2
    for ch in reversed(body):
        total += int(ch) * mult
        mult = mult + 1 if mult < 7 else 2
    rest = 11 - (total % 11)
    return "0" if rest == 11 else "K" if rest == 10 else str(rest)


# RNA-05/08/14: la validez del DV no identifica al titular.
RUT_TITULAR_RE = re.compile(
    r"(?P<label>\brun\b|\brut\b|\bcedula\s+de\s+identidad\b|\bc\.i\.)"
    r"\s*(?P<owner>(?:d(?:e|el)\s+)?(?:la\s+|el\s+)?"
    r"(?:persona\s+natural|persona\s+juridica|representante(?:\s+legal)?|"
    r"empresa|sociedad|corporacion|fundacion|personal|particular))?"
    r"\s*(?:(?:n[°ºo.]|numero)\s*)?[:=]?\s*$"
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


def find_ruts(text):
    out = []
    ftext = fold(text)
    for m in RUT_RE.finditer(text):
        valid = _rut_dv(re.sub(r"\D", "", m.group(1))) == m.group(2).upper()
        cue = bool(RUT_CUE_RE.search(ftext[max(0, m.start() - 20):m.start()]))
        pen = bool(NUMBER_PREFIX_RE.search(ftext[max(0, m.start() - 14):m.start()]))
        score = (0.90 if valid else 0.45) + (0.05 if (cue and valid) else 0.20 if cue else 0.0) - (0.30 if pen else 0.0)
        out.append({"s": m.start(), "e": m.end(), "valid": valid, "score": max(0.05, min(score, 0.99))})
    return out

AREA_CODES = {"32", "33", "34", "35", "41", "42", "43", "45", "51", "52", "53", "55",
              "57", "58", "61", "63", "64", "65", "67", "71", "72", "73", "75"}

GROUP_RE = re.compile(r"\+?\(?\d+\)?")

SEP_OK = re.compile(r"^[\s.\-]{1,2}$")

DATE_RE = re.compile(r"^\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}$")

RUT_DOTTED_RE = re.compile(r"^\d{1,2}\.\d{3}\.\d{3}$")

PHONE_BASE = {"movil": 0.60, "fijo": 0.50, "fijo_regional": 0.50, "fijo_8_digitos": 0.40, "servicio_600_800": 0.90}

def _phone_kind(digits, cue):
    national = digits[2:] if digits.startswith("56") and len(digits) in (10, 11) else digits
    if len(national) == 9:
        if national[0] == "9":
            return "movil"
        if national[0] == "2":
            return "fijo"
        if national[:2] in AREA_CODES:
            return "fijo_regional"
    elif len(national) == 10 and national[:3] in {"600", "800"}:
        return "servicio_600_800"
    elif len(national) == 8 and cue:
        return "fijo_8_digitos"
    return None

def find_phones(text, exclude_spans=()):
    folded = fold(text)
    groups = list(GROUP_RE.finditer(text))
    found, i = [], 0
    while i < len(groups):
        j_max = i
        while (j_max + 1 < len(groups) and j_max - i < 5
               and SEP_OK.match(text[groups[j_max].end():groups[j_max + 1].start()])):
            j_max += 1
        hit = None
        for j in range(j_max, i - 1, -1):
            s, e = groups[i].start(), groups[j].end()
            raw = text[s:e]
            digits = re.sub(r"\D", "", raw)
            if not 8 <= len(digits) <= 11:
                continue
            if s > 0 and text[s - 1].isalpha():
                continue
            if _overlaps_any(s, e, exclude_spans):
                continue
            st = raw.strip()
            if DATE_RE.match(st) or RUT_DOTTED_RE.match(st):
                continue
            if re.match(r"\s*-\s*[\dkK]\b", text[e:]):
                continue
            if re.search(r"\d{7,}-\d$", raw.replace(".", "").replace(" ", "")):
                continue
            cue = bool(PHONE_CUE_RE.search(folded[max(0, s - 40):s]))
            kind = _phone_kind(digits, cue)
            if not kind:
                continue
            pen = bool(NUMBER_PREFIX_RE.search(folded[max(0, s - 14):s]))
            score = PHONE_BASE[kind] + (0.25 if cue else 0.0) - (0.30 if pen else 0.0)
            if score < PHONE_MIN_SCORE:
                continue
            hit = (j, {"s": s, "e": s + len(raw.rstrip()), "kind": kind, "score": min(score, 0.99)})
            break
        if hit:
            found.append(hit[1])
            i = hit[0] + 1
        else:
            i += 1
    return found

def classify_phone(p, fmasked):
    if p["kind"] == "servicio_600_800":
        return "telefono_institucional", "numero_600_800"
    kind = nearest_group(fmasked, p["s"], p["e"], CUES_PHONE, CUE_MAX_DISTANCE)
    if kind == "personal":
        return "telefono_personal", "contexto"
    if kind == "institucional":
        return "telefono_institucional", "contexto"
    return "telefono_revisar", "sin_evidencia"

STREET_TYPES = r"(?:calle|avenida|avda\.?|av\.?|pasaje|pje\.?|camino|ruta|poblaci[oó]n|villa|parcela|condominio)"

ADDRESS_RE = re.compile(
    rf"(?i)\b{STREET_TYPES}\s+(?:[\w'’.\-]+\s+){{0,6}}?(?:n[°º]?\.?\s*|#\s*)?\d{{1,5}}\b"
)

DOMICILE_ADDRESS_RE = re.compile(
    r"(?i)\b(?:domicilio|domiciliado|domiciliada|residente|reside|vive)"
    r"\s+(?:en\s+)"
    r"(?P<addr>(?:[\w'’.\-]+\s+){1,5}(?:n[°º]?\.?\s*|#\s*)?\d{1,5}\b)"
)

def find_addresses(text):
    found = []

    for m in ADDRESS_RE.finditer(text):
        found.append({"s": m.start(), "e": m.end()})

    for m in DOMICILE_ADDRESS_RE.finditer(text):
        found.append({
            "s": m.start("addr"),
            "e": m.end("addr")
        })

    # Eliminar coincidencias duplicadas
    unique = {}
    for item in found:
        unique[(item["s"], item["e"])] = item

    return list(unique.values())

def classify_address(fmasked, s, e):
    kind = nearest_group(fmasked, s, e, CUES_ADDRESS, ADDRESS_CUE_DISTANCE)
    return {"personal": ("direccion_personal", "contexto"),
            "inmueble": ("direccion_inmueble", "contexto"),
            "institucional": ("direccion_institucional", "contexto")}.get(kind, ("direccion_revisar", "sin_evidencia"))

ESTADO_CIVIL_RE = re.compile(
    r"(?i)\b(?:"
    r"soltero|soltera|"
    r"casado|casada|"
    r"divorciado|divorciada|"
    r"viudo|viuda|"
    r"conviviente\s+civil|conviviente|separado|separada"
    r")\b"
)

def find_estado_civil(text):
    return [
        {"s": m.start(), "e": m.end()}
        for m in ESTADO_CIVIL_RE.finditer(text)
    ]

_BIRTH_CUE = r"(?:fecha\s+de\s+nacimiento|nacimiento|nacid[oa]s?|nacio\b|nac\.)"

BIRTH_CUE_RE = re.compile(r"\b" + _BIRTH_CUE)

_DATE_PAT = (r"(?:\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}"
             r"|\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}"
             r"|\d{1,2}\s+de\s+[a-z]+\s+(?:de|del)\s+\d{4})")

BIRTH_DATE_RE = re.compile(r"\b" + _BIRTH_CUE + r"[^\d\n]{0,25}(?P<date>" + _DATE_PAT + ")")

def find_birthdates(text):
    """Fechas precedidas de un cue de nacimiento. Se busca sobre texto plegado (mismo largo que el original)."""
    ftext = fold(text)
    return [{"s": m.start("date"), "e": m.end("date")} for m in BIRTH_DATE_RE.finditer(ftext)]

NON_ID_CONTEXT_RE = re.compile(
    r"\b(?:fojas|plano|catastro|inscripcion|expediente|decreto|ley|d\.s\.|dfl|resolucion|exenta|rol|id|oficio|folio)"
    r"\s*(?:n[°º]?\.?\s*)?$")

TITLE_RE = re.compile(r"^(?:sr\.?|sra\.?|srta\.?|don|do[ñn]a|dr\.?|dra\.?|ing\.?|abog\.?|prof\.?)\s+", re.I)

TOPONYM_LEFT_RE = re.compile(
    r"\b(?:comuna|municipalidad|provincia|region|ciudad|localidad|sector|distrito|villa|poblacion)\s+(?:del?\s+(?:la\s+|los\s+|las\s+)?)?$")

SANTO_LEFT_RE = re.compile(r"\b(?:san|santa|santo)\s*$")

STREET_LEFT_RE = re.compile(
    r"\b(?:calle|avenida|avda|av|pasaje|pje|camino|ruta|poblacion|villa|plaza|parque|escuela|colegio|liceo|hospital|conjunto residencial)\.?\s*[\"“'«]?\s*$")

GAZETTEER_LUGARES = set()

def load_gazetteer():
    global GAZETTEER_LUGARES
    if GAZETTEER_FILE.exists():
        GAZETTEER_LUGARES = {" ".join(re.findall(r"[^\W\d_]+", fold(l))) for l in
                             GAZETTEER_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}
        print("Diccionario de lugares:", len(GAZETTEER_LUGARES), "entradas")
    else:
        print("Sin diccionario de comunas (opcional):", GAZETTEER_FILE)

def refine_person(text, s, e):
    """Devuelve (categoria, s, e, basis, flags). categoria = 'persona' o 'ignorar:...'."""
    m = TITLE_RE.match(text[s:e])
    if m:
        s += m.end()
    value = text[s:e]
    flags = []
    toks = [fold(t) for t in re.findall(r"[^\W\d_]+", value)]
    if sum(len(t) for t in toks) < 2:
        return "ignorar:ruido", s, e, "sin_letras", flags
        # Descartar términos legales o administrativos que no son nombres
    palabras_utiles = [
        t for t in toks
        if t not in {
            "el", "la", "los", "las", "de", "del",
            "en", "y", "a", "al", "por"
        }
    ]

    if palabras_utiles and all(t in NON_PERSON_WORDS for t in palabras_utiles):
        return "ignorar:no_es_persona", s, e, "diccionario_no_persona", flags
    left = fold(text[max(0, s - 40):s])
    right = fold(text[e:e + 40])
    if any(t in INSTITUTION_WORDS for t in toks):
        return "ignorar:organizacion", s, e, "diccionario_institucion", flags
    if SANTO_LEFT_RE.search(left):
        return "ignorar:lugar", s, e, "prefijo_santo", flags
    if TOPONYM_LEFT_RE.search(left):
        return "ignorar:lugar", s, e, "contexto_toponimo", flags
    if STREET_LEFT_RE.search(left):
        return "ignorar:contexto", s, e, "contexto_calle_o_establecimiento", flags
    if " ".join(toks) in GAZETTEER_LUGARES:
        return "ignorar:lugar", s, e, "diccionario_lugar", flags
    if toks and toks[0] in {"san", "santa", "santo"}:
        flags.append("posible_toponimo")
    near = set(re.findall(r"[a-z]+", left[-60:] + " " + right[:60]))
    if near & CARGO_WORDS:
        flags.append("cargo_publico?")
    return "persona", s, e, "gliner", flags

def _is_upper_segment(seg):
    letters = [c for c in seg if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.8

def ner_view(block):
    """Texto que ve el NER: tramos casi todo en MAYÚSCULAS pasan a formato título (el largo se conserva)."""
    text = block["text"]
    chars = list(text)
    for sp in block["spans"]:
        seg = text[sp["block_start"]:sp["block_end"]]
        if _is_upper_segment(seg):
            t = seg.title()
            if len(t) == len(seg):
                chars[sp["block_start"]:sp["block_end"]] = list(t)
    return "".join(chars)

def run_gliner(texts):
    """Una lista de entidades por texto, corriendo todas las pasadas (ver gliner_multipass)."""
    return [gliner_multipass(t) for t in texts]

def run_spanish_ner(texts):
    if es_pipe is None:
        return [[] for _ in texts]
    try:
        raw = es_pipe(texts)
    except Exception:
        raw = []
        for t in texts:
            try:
                raw.append(es_pipe(t))
            except Exception:
                raw.append([])
    out = []
    for res in raw:
        out.append([{"group": str(r.get("entity_group", "")).upper(), "score": float(r.get("score", 0.0)),
                     "start": int(r["start"]), "end": int(r["end"])} for r in res])
    return out

def mk(text, s, e, cat, score, sources, basis, flags=None):
    return {"s": s, "e": e, "text": text[s:e].replace("\n", " "), "category": cat,
            "score": round(float(score), 3), "sources": list(sources), "basis": basis, "flags": list(flags or [])}

def _trim(text, s, e):
    while s < e and (text[s].isspace() or text[s] in ".,;:()\"'«»"):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in ".,;:()\"'«»"):
        e -= 1
    return s, e

# Los candidatos genéricos no deben duplicar un candidato contextual del mismo dato.
_SPECIFIC = {
    "run_rut_personal": {"run_rut", "run_rut_revisar"}, "run_rut_juridico": {"run_rut", "run_rut_revisar"},
    "persona_menor": {"persona", "persona_revisar"}, "persona_funcionario": {"persona", "persona_revisar"},
    "direccion_personal": {"direccion_revisar"}, "direccion_inmueble": {"direccion_revisar"},
    "direccion_institucional": {"direccion_revisar"}, "correo_personal": {"correo_revisar"},
    "correo_institucional": {"correo_revisar"}, "telefono_personal": {"telefono_revisar"},
    "telefono_institucional": {"telefono_revisar"}, "nacionalidad_personal": {"nacionalidad"},
    "dato_financiero_personal": {"dato_financiero"}, "dato_previsional_personal": {"dato_previsional"},
    "dato_salud_personal": {"dato_salud"}, "dato_menor": {"dato_menor_revisar"},
    "codigo_validacion": {"codigo_validacion_revisar"},
}


def resolve_overlaps(ents):
    unique = {}
    for x in ents:
        key = (x["s"], x["e"], x["category"])
        if key not in unique:
            unique[key] = x
        else:
            old = unique[key]
            # Preferir evidencia contextual, no solo el puntaje arbitrario.
            chosen, other = (x,old) if (bool(x.get("owner")),x["basis"]!="solo_gliner",x["score"]) > (bool(old.get("owner")),old["basis"]!="solo_gliner",old["score"]) else (old,x)
            chosen["sources"] = sorted(set(chosen["sources"]+other["sources"]))
            chosen["flags"] = sorted(set(chosen["flags"]+other["flags"]))
            unique[key] = chosen
    vals=list(unique.values()); keep=[]
    for x in vals:
        winners=[y for y in vals if y is not x and y["s"]<=x["s"] and x["e"]<=y["e"] and x["category"] in _SPECIFIC.get(y["category"],set()) and x["basis"]!="titularidad_contradictoria"]
        if winners:
            for y in winners: y["flags"].append("hipotesis_generica_resuelta:"+x["category"])
            continue
        # Si dos candidatos de igual categoría están contenidos, preservar el tramo más largo
        # solo cuando cuenta con evidencia explícita (p. ej. domicilio con localidad).
        wider=[y for y in vals if y is not x and y["category"]==x["category"] and y["s"]<=x["s"] and x["e"]<=y["e"] and (y["s"],y["e"])!=(x["s"],x["e"]) and y["basis"] in {"domicilio_explicito","ubicacion_inmueble_explicita","dato_sensible_contextual","url_validacion_explicita"}]
        if wider: continue
        keep.append(x)
    return sorted(keep,key=lambda x:(x["s"],x["e"],x["category"]))


# Contexto local: se limita a cláusulas; no se infiere titularidad por un apellido aislado.
_PUBLIC_ROLE = r"(?:ministro|ministra|subsecretari[oa]|secretari[oa]\s+regional\s+ministerial|seremi|alcalde|alcaldesa|director(?:a)?\s+(?:regional|nacional)|funcionari[oa]\s+public[oa]|jefe\s+de\s+(?:division|departamento|unidad))"
_PERSON_ROLE = r"(?:arrendatari[oa]|solicitante|beneficiari[oa]|representante\s+legal|compareciente|titular|persona\s+natural)"
# Mayúsculas de origen (incluye documentos completamente en mayúsculas), hasta cinco palabras.
_NAME_WORD = r"[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ'-]{1,30}"
_NAME = rf"{_NAME_WORD}(?:[ \t]+(?:(?:de|del|la|las|los|y)[ \t]+)?{_NAME_WORD}){{1,4}}"
_NAME_CUE = re.compile(rf"(?i:\b(?:don|doña|dona|sr\.?|sra\.?|menor(?:\s+de\s+edad)?|niñ[oa]|adolescente|{_PERSON_ROLE}|{_PUBLIC_ROLE})\s*[:：]?\s+)(?P<name>{_NAME})")
_AGE = re.compile(r"\b(?P<age>\d{1,2}\s+anos(?:\s+de\s+edad)?)\b")
_MINOR_CUE = re.compile(r"\b(?:menor(?:\s+de\s+edad)?|nina|nino|adolescente)\b")
_BOUNDARY = re.compile(r";|\n\s*\n|(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])")
_ORG_SUFFIX = re.compile(r"\b(?:spa|ltda|s\.a\.|eirl|empresa|sociedad|fundacion|corporacion)\b")
_PUBLIC_FIELD = re.compile(r"\b(?:inmueble|predio|terreno|arriendo|canon|renta\s+de\s+arriendo|precio\s+del\s+inmueble|monto\s+adeudado|garantia|expediente|oficio|decreto|resolucion)\b")
_CONTACT_CUE = {
    "correo": re.compile(r"\b(?:correo(?:\s+electronico)?|email|e-mail|mail)\s*(?:de\s+contacto)?\s*[:：]?\s*$"),
    "telefono": re.compile(r"\b(?:telefono|celular|fono|movil|whatsapp|contacto)\s*[:：]?\s*$"),
    "direccion": re.compile(r"\b(?:domicilio(?:\s+particular)?|domiciliad[oa]|reside|vive|direccion\s+particular)\s*(?:en\s+)?[:：]?\s*$"),
}
_NATIONALITY = r"(?:chilen[oa]|argentin[oa]|peruan[oa]|bolivian[oa]|colombian[oa]|venezolan[oa]|ecuatorian[oa]|brasilen[oa]|uruguay[oa]|paraguay[oa]|mexican[oa]|espanol[ae]?|espanola|frances[ae]?|italian[oa]|alemana?|estadounidense|canadiense|haitian[oa]|cuban[oa]|dominican[oa]|chin[oa]|japones[ae]?|extranjer[oa])"
# Capturar solo el valor; no la etiqueta ni datos de otra cláusula.
_CONTEXT_FIELDS = [
    ("dato_salud_personal", r"\bcondicion medica(?: consignada)?\s+(?:es|:)\s*(?P<v>[^.;,\n]{2,75})"),
    ("dato_financiero_personal", r"\b(?:cuenta\s+(?:bancaria|corriente|vista|de\s+ahorro|rut))\s*(?:personal|particular|del\s+(?:solicitante|titular))?\s*(?:n[°ºo.]\s*|es\s+)?[:：]?\s*(?P<v>\d[\d. -]{3,24}\d)(?!\d)"),
    ("dato_financiero_personal", r"\b(?:saldo\s+(?:bancario|disponible|de\s+(?:su\s+)?cuenta)|ingreso\s+mensual|sueldo)\s*(?:personal|particular|individual)?\s*(?:de|es|alcanza)?\s*[:：]?\s*(?P<v>\$\s*\d[\d.,]*\d|\$\s*\d)"),
    ("dato_previsional_personal", r"\b(?:afiliad[oa]\s+a|afiliacion\s+(?:previsional\s+)?a)\s+(?P<v>afp\s+[a-z][a-z0-9-]*(?:\s+(?:de|del)\s+[a-z][a-z0-9-]*)?)\b"),
    ("dato_previsional_personal", r"\b(?:cotizaciones?\s+previsionales?|cotizaciones?\s+personales?)\s*(?:personales?|individuales?)?\s*(?:alcanzan?|ascienden?\s+a|de|es)?\s*[:：]?\s*(?P<v>\$\s*\d[\d.,]*\d|\$\s*\d)"),
    ("dato_previsional_personal", r"\b(?:numero\s+de\s+)?afiliacion\s+individual\s*(?:es\s+)?[:：]?\s*(?P<v>[a-z0-9-]*\d[a-z0-9-]*)"),
    ("dato_salud_personal", r"\b(?:diagnostico\s+medico|tratamiento\s+medico|discapacidad)\s*(?:de\s+tipo|de|con|:)\s*(?P<v>[^;,.]{2,75}?)(?=\s+(?:y\s+(?:tratamiento|controles|reside|presenta)|se\s+indica|el\s+acta)|[;,.]|$)"),
    ("dato_salud_personal", r"\bcondicion\s+de\s+salud\s*(?:de\s+|:\s*)?(?P<v>[^;,.]{2,65}?)(?=\s+y\s+(?:controles|tratamiento|reside)|[;,.]|$)"),
    ("dato_salud_personal", r"\b(?:afiliacion\s+de\s+salud\s+personal\s+a)\s+(?P<v>fonasa(?:\s+y\s+tramo\s+[abcd])?|isapre\s+[a-z]+)\b"),
    ("dato_financiero_personal", r"\b(?:cuenta\s+(?:bancaria|corriente|vista|de\s+ahorro)|cuenta\s+rut)\s*(?:n[°ºo.]\s*)?[:：]?\s*(?P<v>\d[\d. -]{3,24}\d)(?!\d)"),
    ("dato_financiero_personal", r"\b(?:saldo\s+(?:bancario|de\s+(?:su\s+)?cuenta)|saldo\s+personal|ingreso\s+mensual\s+personal|sueldo\s+personal)\s*(?:de\s+)?[:：]?\s*(?P<v>\$\s*\d[\d.,]*)"),
    ("dato_previsional_personal", r"\b(?:afiliad[oa]\s+a\s+|afiliacion\s+(?:previsional\s+)?(?:a\s+)?|prevision\s*[:：]\s*)?(?P<v>afp\s+[a-z][a-z0-9-]*(?:\s+(?:capital|habitat|modelo|planvital|provida|cuprum|uno))?)\b"),
    ("dato_previsional_personal", r"\b(?:cotizacion(?:es)?\s+previsional(?:es)?|cotizacion(?:es)?\s+(?:de\s+)?afp)\s*[:：]?\s*(?P<v>\$\s*\d[\d.,]*|\d+(?:[.,]\d+)?\s*%)"),
    ("dato_salud_personal", r"\b(?:afiliad[oa]\s+a\s+|sistema\s+de\s+salud\s*[:：]\s*|prevision\s+de\s+salud\s*[:：]\s*)?(?P<v>fonasa(?:\s+tramo\s+[abcd])?|isapre\s+[a-z]+)\b"),
    ("dato_salud_personal", r"\b(?:diagnostico(?:\s+medico)?|enfermedad|patologia|condicion\s+medica|tratamiento\s+medico|discapacidad)\s*[:：]\s*(?P<v>[^;\n,.]{2,65})"),
    ("dato_salud_personal", r"\b(?:padece|diagnosticad[oa]\s+con|sufre\s+de)\s+(?P<v>[^;\n,.]{2,65})"),
    ("dato_salud_personal", r"\b(?P<v>diabetes(?:\s+(?:mellitus\s+)?tipo\s+[12])?|hipertension\s+arterial|discapacidad\s+(?:fisica|visual|auditiva|intelectual|psicosocial))\b"),
]


def _clause(text, s, e):
    """Límites originales, sin cruzar fin de oración ni bloque."""
    a, b = 0, len(text)
    for m in _BOUNDARY.finditer(text):
        if m.end() <= s:
            a = m.end()
        elif m.start() >= e:
            b = m.start(); break
    return a, b


def _add_candidate(ents, text, s, e, cat, basis, score=.9, owner=None):
    s, e = _trim(text, s, e)
    if e <= s:
        return
    x = mk(text, s, e, cat, score, ["regla_contextual"], basis)
    if owner:
        x["owner"] = owner["text"]
        x["owner_role"] = owner.get("role", "natural")
    ents.append(x)


def nombre_plausible(value):
    f = fold(value)
    return (len(f.split()) >= 2 and not _ORG_SUFFIX.search(f)
            and not re.search(r"\d|[@/]|^(?:pasaje|calle|avenida|avda|camino|ruta|parcela|sector|colegio|escuela|hospital|centro|inmueble|predio|domicilio|correo|resolucion|expediente)\b", f))


def contextualizar(text, ents, role=""):
    """Relaciona candidatos en una cláusula. Evidencia débil queda en revisión."""
    f = fold(text)
    # Aumentar recall de nombres con roles explícitos aunque GLiNER no los encuentre.
    for m in _NAME_CUE.finditer(text):
        s, e = m.span("name")
        # No incluir RUN/RUT, siglas administrativas o cargo en el nombre.
        stop = re.search(r"\s+(?:RUN|RUT|CEDULA|CÉDULA|DECRETO|RESOLUCION|RESOLUCIÓN|NACIONALIDAD|DOMICILIO|SECRETARIA|SECRETARIO|MINISTRA|MINISTRO)\b", text[s:e], re.I)
        if stop: e = s + stop.start()
        if len(text[s:e].split()) < 2 or not text[s:e].strip(): continue
        if not any(x["category"].startswith("persona") and _ov(s,e,x["s"],x["e"]) for x in ents):
            cat, s, e, _, flags = refine_person(text, s, e)
            if cat == "persona": _add_candidate(ents,text,s,e,"persona","nombre_con_rol",.85)
    people = sorted([x for x in ents if x["category"] in {"persona", "persona_revisar"}], key=lambda x:(x["s"],-x["e"]))
    # Solo una hipótesis de nombre por tramo para asociar titularidad.
    owners = []
    for p in people:
        if any(_ov(p["s"],p["e"],q["s"],q["e"]) for q in owners): continue
        if not nombre_plausible(p["text"]): continue
        if any(x is not p and x["category"].startswith(("direccion_", "correo_"))
               and x["s"] <= p["s"] and p["e"] <= x["e"] for x in ents): continue
        a,b = _clause(text,p["s"],p["e"])
        left, right = f[max(a,p["s"]-85):p["s"]],f[p["e"]:min(b,p["e"]+90)]
        public = bool(re.search(rf"\b{_PUBLIC_ROLE}\s*[:,]?\s*$",left) or re.match(rf"\s*[,(-]?\s*{_PUBLIC_ROLE}\b",right))
        # También firma en línea siguiente: nombre seguido inmediatamente de cargo público explícito.
        public |= bool(re.match(rf"[ \t]*\n[ \t]*{_PUBLIC_ROLE}\b",f[p["e"]:p["e"]+95]))
        if public and _ORG_SUFFIX.search(left+" "+right): public=False
        minor = bool(re.search(r"\b(?:menor(?:\s+de\s+edad)?|nina|nino|adolescente)\s*[:,]?\s*$",left))
        age = re.match(r"\s*,?\s*(?:de\s+)?(\d{1,2})\s+anos\b",right)
        minor |= bool(age and int(age.group(1)) < 18 and not public)
        # Edad puede ir después del RUN inmediatamente asociado.
        age_after_rut = re.match(r"\s*,?\s*(?:run|rut)\s*[:：]?\s*[\d.,-]+[kK]?\s*,?\s*(?:de\s+)?(\d{1,2})\s+anos\b",right)
        minor |= bool(age_after_rut and int(age_after_rut.group(1)) < 18)
        p["role"] = "ambiguo" if public and minor else "menor" if minor else "funcionario" if public else "natural"
        p["owner"] = p["text"]; p["owner_role"] = p["role"]
        if p["role"] == "menor": p.update(category="persona_menor",basis="menor_explicito")
        elif p["role"] == "funcionario": p.update(category="persona_funcionario",basis="cargo_publico_explicito")
        owners.append(p)

    def owner_for(s,e):
        a,b = _clause(text,s,e)
        prev = [p for p in owners if a <= p["s"] and p["e"] <= s and s-p["e"] <= 180]
        if not prev and role in {"text", "paragraph", "content", "region"}:
            # Una referencia explícita puede continuar la oración precedente.
            prefix = f[a:s].strip()
            if re.match(r"(?:su(?:s)?\b|la (?:condicion medica|menor|persona)\b|el interesado\b)", prefix):
                previous, _ = _clause(text, max(0,a-2), max(0,a-1))
                possible = [p for p in owners if previous <= p["s"] and p["e"] <= a and s-p["e"] <= 360]
                if prefix.startswith("la menor"):
                    possible = [p for p in possible if p["role"]=="menor"] if len(possible)==1 else []
                if len(possible) == 1:
                    prev = possible
        if not prev: return None
        p = max(prev,key=lambda q:q["e"])
        between = f[p["e"]:s]
        if "table" in str(role).lower() and "\n" in between: return None
        if _ORG_SUFFIX.search(between) or re.search(r"\b(?:otra\s+persona|representad[oa]\s+por|inmueble\s+fiscal|predio\s+fiscal|expediente|resolucion|decreto)\b",between): return None
        return p if p["role"] != "ambiguo" else None

    for x in list(ents):
        s,e,cat = x["s"],x["e"],x["category"]
        a,b = _clause(text,s,e); left=f[max(a,s-100):s]
        owner=owner_for(s,e)
        if owner and cat.startswith(("run_rut", "correo", "telefono", "direccion", "dato_", "fecha_nacimiento", "estado_civil", "nacionalidad")):
            x["owner"], x["owner_role"] = owner["text"],owner["role"]
        if cat == "run_rut_revisar" and owner and x["basis"] != "titularidad_contradictoria":
            if re.search(r"\b(?:rut|run|cedula(?:\s+de\s+identidad)?)\s*(?:n[°ºo.]\s*)?[:：]?\s*$",left) and not _ORG_SUFFIX.search(f[owner["e"]:s]):
                # Rango jurídico contradictorio con el nombre requiere revisión, no decisión automática.
                key=clave_rut(x["text"])
                if key[:-1].isdigit() and 0<int(key[:-1])<30000000:
                    x.update(category="run_rut_personal",basis="rut_asociado_nombre")
        elif cat.startswith(("correo_","telefono_","direccion_")):
            family=cat.split("_")[0]
            groups={"correo":CUES_EMAIL,"telefono":CUES_PHONE,"direccion":CUES_ADDRESS}[family]
            # Una indicación explícita próxima prevalece sobre el dominio.
            local=f[max(a,s-75):s]
            explicit = next((kind for kind,patterns in groups.items() if any(re.search(rx.pattern+r"\s*[:：]?\s*$",local) for rx in patterns)),None)
            if family == "telefono" and owner and owner["role"] in {"menor","natural"} and _CONTACT_CUE["telefono"].search(local):
                explicit = "personal"
            if family == "direccion":
                # Un domicilio explícito vence los términos generales del inmueble.
                if _CONTACT_CUE[family].search(local) or re.search(r"\bdeclara\s+residir\s+en\s*$",local): explicit="personal"
                elif re.search(r"\b(?:inmueble|predio|terreno|propiedad\s+fiscal)\b[^;]{0,40}\b(?:ubicad[oa]|direccion)\s*(?:en\s+)?[:：]?\s*$",local): explicit="inmueble"
            if explicit:
                x.update(category=family+"_"+explicit,basis="contexto_local_explicito")
            elif family in {"correo", "telefono"} and cat.endswith("_institucional") and x["basis"]=="contexto":
                x.update(category=family+"_revisar",basis="contexto_no_vinculado")
            elif owner and _CONTACT_CUE[family].search(local) and cat not in {family+"_institucional",family+"_inmueble"}:
                # Contacto genérico de funcionario no es necesariamente privado.
                if owner["role"] in {"natural","menor"}:
                    x.update(category=family+"_personal",basis="contacto_asociado_nombre")
            elif x["basis"]=="contexto" and cat.endswith("_personal"):
                x.update(category=family+"_revisar",basis="contexto_no_vinculado")
        elif cat == "estado_civil" and not (owner or re.search(r"\b(?:estado\s+civil|de\s+estado\s+civil)\s*[:：]?\s*$",left)):
            x.update(category="estado_civil_revisar",basis="estado_civil_sin_titular")
        elif cat == "nacionalidad" and (owner or re.search(r"\bnacionalidad\s*(?:es\s+)?[:：]?\s*$",left)) and re.fullmatch(_NATIONALITY,fold(x["text"])):
            x.update(category="nacionalidad_personal",basis="nacionalidad_contextual")

    for m in re.finditer(rf"\b(?P<v>{_NATIONALITY})\b",f):
        s,e=m.span("v"); a,_=_clause(text,s,e); left=f[max(a,s-60):s]; owner=owner_for(s,e)
        explicit=bool(re.search(r"\b(?:de\s+)?nacionalidad\s*(?:es\s+)?[:：]?\s*$",left))
        attached=bool(owner and re.fullmatch(r"\s*,?\s*",f[owner["e"]:s]))
        if explicit or attached: _add_candidate(ents,text,s,e,"nacionalidad_personal","nacionalidad_contextual",owner=owner)
        elif owner: _add_candidate(ents,text,s,e,"nacionalidad","gentilicio_sin_vinculo_claro",.6,owner)
    # Una nacionalidad explícita fuera de la lista no se pierde: queda pendiente.
    for m in re.finditer(r"\bnacionalidad\s*(?:es\s+)?[:：]?\s*(?P<v>[a-z][a-z-]{2,35})\b",f):
        s,e=m.span("v")
        if f[s:e] not in {"de","del","persona","empresa","sociedad","no","sin","desconocida"} and not re.fullmatch(_NATIONALITY,f[s:e]):
            _add_candidate(ents,text,s,e,"nacionalidad","nacionalidad_fuera_diccionario",.65,owner_for(s,e))
    # Direcciones rurales / sin número, únicamente tras un rótulo inequívoco.
    for m in re.finditer(r"\b(?:domicilio(?:\s+particular)?|domiciliad[oa]|declara\s+residir|reside)\s*(?:en\s+|[:：]\s*)(?P<v>[^;\n,.]{4,90}(?:,\s*(?:villa|poblacion|sector|parcela|comuna|departamento|depto\.?|casa)\s+[^;\n,.]{1,40})?)",f):
        s,e=m.span("v"); val=f[s:e]
        if re.search(r"\b(?:calle|pasaje|camino|ruta|sector|parcela|villa|poblacion|avenida|avda|s/n|sin\s+numero)\b",val):
            # Cortar antes de otros campos y no invadir contenido de un acto.
            cut=re.search(r"\s+(?:rut|run|correo|telefono|nacionalidad|estado\s+civil|inmueble\s+fiscal|y\s+cursa|y\s+estudia|y\s+asiste)\b",val)
            if cut:e=s+cut.start()
            _add_candidate(ents,text,s,e,"direccion_personal","domicilio_explicito",owner=owner_for(s,e))
    for m in re.finditer(r"\b(?:inmueble|predio|terreno|propiedad\s+fiscal)\b[^;\n,.]{0,40}\b(?:ubicad[oa]\s+en|direccion\s*[:：])\s*(?P<v>[^;\n,.]{4,90}(?:,\s*(?:villa|poblacion|sector|comuna)\s+[^;\n,.]{1,40})?)", f):
        s,e=m.span("v")
        if re.search(r"\b(?:calle|pasaje|camino|ruta|sector|parcela|villa|poblacion|avenida|avda)\b",f[s:e]):
            _add_candidate(ents,text,s,e,"direccion_inmueble","ubicacion_inmueble_explicita")
    if re.search(r"\bdeslinde[s]?\b",f) or len(re.findall(r"\b(?:norte|sur|este|oeste|oriente|poniente)\s*:",f))>=2:
        for m in re.finditer(r"\b(?:norte|sur|este|oeste|oriente|poniente)\s*:\s*(?P<v>[^;\n]{2,100}?)(?=;|\n|$|\s+(?:norte|sur|este|oeste|oriente|poniente)\s*:)",f):
            _add_candidate(ents,text,*m.span("v"),"deslindes","deslinde_explicito")
    for m in re.finditer(r"\b(?:rol\s+(?:de\s+)?avaluo|inscripcion\s+conservatoria)\s*[:：]?\s*(?P<v>[^;\n,.]{2,50})",f):
        _add_candidate(ents,text,*m.span("v"),"ubicacion_inmueble_revisar","definicion_pendiente_rna09")
    for cat,pat in _CONTEXT_FIELDS:
        for m in re.finditer(pat,f):
            s,e=m.span("v"); owner=owner_for(s,e); a,_=_clause(text,s,e); prefix=f[max(a,m.start()-65):m.start()]
            cut=re.search(r"\s+(?:y\s+)?(?:rut|run|correo|telefono|nacionalidad|domicilio|inmueble|expediente|resolucion|decreto|cotizacion|saldo)\b",f[s:e])
            if cut:e=s+cut.start()
            # Sin un titular, una etiqueta de AFP/diagnóstico puede referir a una institución o norma.
            explicit=bool(re.search(r"\b(?:personal|personales|particular|individual|del\s+titular|del\s+solicitante|del\s+representante)\b",f[m.start():s]) or re.search(r"\b(?:su|sus)\s*$",prefix))
            if cat=="dato_financiero_personal" and _PUBLIC_FIELD.search(prefix) and not explicit: owner=None
            if owner and owner["role"]=="funcionario" and not explicit: owner=None
            # Un rótulo inequívoco de afiliación individual identifica el carácter privado.
            explicit |= bool(re.search(r"\b(?:afiliacion\s+individual|cotizaciones?\s+previsionales?\s+personales?)\b", f[m.start():s]))
            if owner or explicit:
                target=cat
            else: target=cat.removesuffix("_personal")
            _add_candidate(ents,text,s,e,target,"dato_sensible_contextual" if target==cat else "sensible_sin_titular",.9 if target==cat else .65,owner)
    # Edad y establecimiento del menor: solo vínculo local comprobado, o etiqueta explícita.
    for m in _AGE.finditer(f):
        s,e=m.span("age"); owner=owner_for(s,e)
        n=int(re.match(r"\d+",f[s:e]).group())
        if n<18 and owner and owner["role"]=="menor": _add_candidate(ents,text,s,e,"dato_menor","edad_menor_explicita",owner=owner)
        elif n<18 and _MINOR_CUE.search(f[max(0,s-45):s]): _add_candidate(ents,text,s,e,"dato_menor_revisar","edad_menor_sin_nombre",.7)
    for m in re.finditer(r"\b(?:estudia\s+en|asiste\s+al?(?:\s+establecimiento)?|registra\s+controles\s+en|establecimiento\s+educacional\s*[:：]|colegio\s*[:：])\s*(?P<v>[^;\n,.]{3,70})",f):
        s,e=m.span("v"); owner=owner_for(s,e)
        stop = re.search(r"\s+y\s+(?:reside|cursa|asiste|registra|declara)\b", f[s:e])
        if stop: e = s + stop.start()
        if owner and owner["role"]=="menor": _add_candidate(ents,text,s,e,"dato_menor","establecimiento_menor",owner=owner)
        elif _MINOR_CUE.search(f[max(0,s-100):s]): _add_candidate(ents,text,s,e,"dato_menor_revisar","establecimiento_sin_titular",.7)
    for m in re.finditer(r"\bcursa\s+(?P<v>(?:primer[oa]?|segund[oa]|tercer[oa]?|cuart[oa]|quint[oa]|sext[oa]|septim[oa]|octav[oa]|[1-8][º°]?)\s+(?:basico|medio))\b",f):
        start,end = m.span("v")
        owner = owner_for(start,end)
        if owner and owner["role"] == "menor":
            _add_candidate(ents,text,start,end,"dato_menor","curso_menor",owner=owner)
    for m in re.finditer(r"https?://[^\s<>;,]+",text,re.I):
        if re.search(r"(?:validar|validacion|verificar|verificacion)",fold(m.group())):
            _add_candidate(ents,text,m.start(),m.end(),"vinculo_validacion","url_validacion_explicita")
    return ents


def detect_block(block, gl_ents, es_ents):
    text = block["text"]
    ftext = fold(text)
    ents = []

    # ---- 1) Reglas ----
    emails = find_emails(text)
    email_spans = [(m["s"], m["e"]) for m in emails]
    fmasked = fold(mask_spans(text, email_spans))
    for m in emails:
        cat, basis = classify_email(text[m["s"]:m["e"]], fmasked, m["s"], m["e"])
        ents.append(mk(text, m["s"], m["e"], cat, m["score"], ["regex"], basis))

    ruts = find_ruts(text)
    phones = find_phones(text, email_spans)
    phone_labels = set()
    for phone in phones:
        prefix = ftext[max(0, phone["s"]-45):phone["s"]]
        if re.search(r"\b(?:celular|movil|telefono(?: particular| personal| institucional)?|mesa central)\s*[:：]?\s*$", prefix):
            phone_labels.add((phone["s"], phone["e"]))
        # ---- Estado civil ----
    for m in find_estado_civil(text):
        ents.append(
            mk(
                text,
                m["s"],
                m["e"],
                "estado_civil",
                0.95,
                ["regex"],
                "coincidencia_lista"
            )
        )
    for m in find_birthdates(text):
        ents.append(mk(text, m["s"], m["e"], "fecha_nacimiento", 0.90, ["regex"], "cue_nacimiento"))
    for m in ruts:
        if (m["s"], m["e"]) in phone_labels:
            continue
        cat, basis = clasificar_rut(text, m["s"], m["e"], m["valid"])
        ents.append(mk(text, m["s"], m["e"], cat, m["score"], ["regex"], basis))

    for p in phones:
        if (p["s"], p["e"]) not in phone_labels and any(_ov(p["s"],p["e"],r["s"],r["e"]) for r in ruts):
            continue
        cat, basis = classify_phone(p, fmasked)
        ents.append(mk(text, p["s"], p["e"], cat, p["score"], ["regex"], basis))

    regex_spans = [(x["s"], x["e"]) for x in ents]

    # ---- 2) Candidatos de dirección (regex + GLiNER) ----
    addr = {}                                   # (s, e) -> {"score", "sources"}
    for m in find_addresses(text):
        addr[(m["s"], m["e"])] = {"score": 0.60, "sources": ["regex"]}

    # ---- 3) GLiNER ----
    person_cands = []
    for g in gl_ents:
        label = g.get("label")
        kind = GLINER_LABELS.get(label)
        score = float(g.get("score", 0.0))
        if kind is None or score < GLINER_THRESHOLDS.get(label, GLINER_MIN_THRESHOLD):
            continue
        s, e = _trim(text, int(g["start"]), int(g["end"]))
        if e <= s:
            continue
        if kind == "persona":
            person_cands.append((s, e, score))
        elif kind == "direccion":
            cur = addr.get((s, e))
            addr[(s, e)] = {"score": max(score, cur["score"] + 0.10) if cur else score,
                            "sources": (cur["sources"] if cur else []) + ["gliner"]}
        elif kind == "correo":
            if "@" in text[s:e] and not _overlaps_any(s, e, regex_spans):
                ents.append(mk(text, s, e, "correo_revisar", score * 0.8, ["gliner"], "solo_gliner"))
        elif kind == "rut":
            mm = RUT_RE.search(text[s:e])
            if mm and not _overlaps_any(s, e, regex_spans):
                valid = _rut_dv(re.sub(r"\D", "", mm.group(1))) == mm.group(2).upper()
                rs, re_ = s + mm.start(), s + mm.end()
                cat, basis = clasificar_rut(text, rs, re_, valid)
                ents.append(mk(text, rs, re_, cat, score * (0.9 if valid else 0.7), ["gliner"], basis))
        elif kind == "fecha_nacimiento":
            if _overlaps_any(s, e, regex_spans):
                continue
            if BIRTH_CUE_RE.search(ftext[max(0, s - BIRTH_CUE_DISTANCE):s]):
                ents.append(mk(text, s, e, "fecha_nacimiento", score * 0.9, ["gliner"], "gliner+cue_nacimiento"))
            else:
                ents.append(mk(text, s, e, "ignorar:fecha", score, ["gliner"], "fecha_sin_cue_nacimiento"))
        elif kind in ("dato_financiero", "dato_previsional"):
            if not _overlaps_any(s, e, regex_spans) and not NON_ID_CONTEXT_RE.search(ftext[max(0,s-25):s]):
                ents.append(mk(text,s,e,kind,score * 0.8,["gliner"],"solo_gliner"))
        elif kind == "dato_salud":
            if not _overlaps_any(s, e, regex_spans):
                ents.append(mk(text, s, e, "dato_salud", score * 0.8, ["gliner"], "solo_gliner"))
        elif kind == "telefono":
            digits = re.sub(r"\D", "", text[s:e])
            left = ftext[max(0, s - 25):s]
            st = text[s:e].strip()
            if (8 <= len(digits) <= 11 and not DATE_RE.match(st) and not RUT_DOTTED_RE.match(st)
                    and not NON_ID_CONTEXT_RE.search(left) and not _overlaps_any(s, e, regex_spans)):
                ents.append(mk(text, s, e, "telefono_revisar", score * 0.8, ["gliner"], "solo_gliner"))
        elif kind in {"nacionalidad", "deslindes", "recinto", "dato_pendiente", "dato_menor_revisar"}:
            if kind != "dato_menor_revisar" or re.search(r"\b(?:menor|nino|nina|adolescente|\d{1,2}\s+anos)\b",ftext[max(0,s-100):e+100]):
                ents.append(mk(text, s, e, kind, score, ["gliner"], "requiere_confirmacion"))
        elif kind.startswith("ignorar:"):
            ents.append(mk(text, s, e, kind, score, ["gliner"], "gliner"))

    # ---- 4) Direcciones: fusionar solapes y clasificar ----
    merged = []
    for (s, e), info in sorted(addr.items()):
        if merged and _ov(s, e, merged[-1]["s"], merged[-1]["e"]):
            merged[-1]["e"] = max(merged[-1]["e"], e)
            merged[-1]["score"] = max(merged[-1]["score"], info["score"]) + 0.05
            merged[-1]["sources"] += info["sources"]
        else:
            merged.append({"s": s, "e": e, "score": info["score"], "sources": list(info["sources"])})
    for a in merged:
        cat, basis = classify_address(fmasked, a["s"], a["e"])
        ents.append(mk(text, a["s"], a["e"], cat, min(a["score"], 0.99), sorted(set(a["sources"])), basis))

    # ---- 5) Personas: vetos y segunda opinión ----
    def es_overlap(s, e):
        return [x for x in es_ents if _ov(s, e, x["start"], x["end"])]

    for s, e, score in person_cands:
        cat, s2, e2, basis, flags = refine_person(text, s, e)
        sources = ["gliner"]
        if cat == "persona":
            over = es_overlap(s2, e2)
            has_per = any(x["group"].startswith("PER") and x["score"] >= SPA_VETO_SCORE for x in over)
            veto = [x for x in over if x["group"][:3] in ("ORG", "LOC") and x["score"] >= SPA_VETO_SCORE]
            if has_per:
                sources.append("ner_es")
            elif veto and score < PERSON_STRONG:
                cat = "ignorar:organizacion" if veto[0]["group"].startswith("ORG") else "ignorar:lugar"
                basis = "ner_es"
        ents.append(mk(text, s2, e2, cat, score, sources, basis, flags))

    if ACEPTAR_PERSONA_SOLO_NER_ES:
        for x in es_ents:
            if x["group"].startswith("PER") and x["score"] >= SPA_PERSON_ADD_SCORE:
                s, e = _trim(text, x["start"], x["end"])
                if e <= s or any(_ov(s, e, k["s"], k["e"]) for k in ents):
                    continue
                cat, s2, e2, basis, flags = refine_person(text, s, e)
                if cat == "persona":
                    ents.append(mk(text, s2, e2, "persona_revisar", x["score"] * 0.9, ["ner_es"], "solo_ner_es", flags))

    for s, e, cat in candidatos_adicionales(text):
        if cat == "codigo_validacion" and not (len(text[s:e]) >= 4 and any(c.isdigit() for c in text[s:e])):
            cat = "codigo_validacion_revisar"
        basis = "codigo_validacion_explicito" if cat == "codigo_validacion" else "requiere_confirmacion"
        ents.append(mk(text, s, e, cat, 0.95 if cat == "codigo_validacion" else 0.5, ["regla_contextual"], basis))
    return resolve_overlaps(filtrar_candidatos(text, contextualizar(text, ents, block.get("role", ""))))

STOP_TOKENS = {"de", "del", "la", "las", "los", "el", "y", "e", "da", "do", "van", "von"}

def _name_tokens(value):
    return [t for t in re.findall(r"[^\W\d_]+", value) if len(t) >= 2]

def _variants(toks):
    out = set()
    if len(toks) >= 3:
        for combo in ((0, 1), (-2, -1), (0, -1)):
            v = (toks[combo[0]], toks[combo[1]])
            if not (v[0].lower() in STOP_TOKENS or v[1].lower() in STOP_TOKENS):
                out.add(v)
    return out

def propagate_names(blocks, block_ents):
    full, variants, singles = {}, {}, {}
    for ents in block_ents:
        for x in ents:
            if x["category"] == "persona" and x["score"] >= PROPAGATE_MIN_SCORE and "propagado" not in x["flags"]:
                toks = _name_tokens(x["text"])
                if len(toks) >= 2:
                    full[tuple(fold(t) for t in toks)] = toks
                    if PROPAGAR_VARIANTES:
                        for v in _variants(toks):
                            variants[tuple(fold(t) for t in v)] = v
                if PROPAGAR_TOKENS_SUELTOS:
                    for t in toks:
                        if len(t) >= 4 and fold(t) not in STOP_TOKENS:
                            singles[(fold(t),)] = [t]

    def rx(key):
        return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(k) for k in key) + r"(?!\w)")

    plans = ([(k, rx(k), "persona", 0.70) for k in full] +
             [(k, rx(k), "persona_revisar", 0.50) for k in variants if k not in full] +
             [(k, rx(k), "persona_revisar", 0.45) for k in singles if k not in full and k not in variants])

    new_block_ents = []
    for block, ents in zip(blocks, block_ents):
        text, ftext = block["text"], fold(block["text"])
        current = list(ents)
        for key, pattern, cat, score in plans:
            for m in pattern.finditer(ftext):
                s, e = m.start(), m.end()
                if any(_ov(s, e, k["s"], k["e"]) for k in current):
                    continue
                cat2, s2, e2, basis, flags = refine_person(text, s, e)
                if cat2 != "persona":
                    continue
                current.append(mk(text, s2, e2, cat, score, ["propagado"], "propagado", flags + ["propagado"]))
        new_block_ents.append(sorted(current, key=lambda x: x["s"]))
    return new_block_ents

def to_fragments(block, s, e):
    frags = []
    for sp in block["spans"]:
        a, b = max(s, sp["block_start"]), min(e, sp["block_end"])
        if a < b:
            frags.append((sp["line_index"], sp["line_start"] + (a - sp["block_start"]),
                          sp["line_start"] + (b - sp["block_start"])))
    return frags

def geometry_for_span(line, start, end):
    """Caracteres exactos primero; nunca autorizar una palabra cortada sin cajas precisas."""
    required = {i for i in range(start,end) if not line["text"][i].isspace()}
    def valid_box(b):
        return (isinstance(b,(tuple,list)) and len(b)==4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in b)
                and b[0]<b[2] and b[1]<b[3])
    def union(boxes):
        return [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
    def rectangle(b):
        x0,y0,x1,y1=b
        return [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]
    cb=line.get("char_boxes") or []
    if required and len(cb)>=end and all(valid_box(cb[i]) for i in required):
        boxes=[cb[i] for i in sorted(required)]
        # No aprobar cajas compartidas por caracteres públicos y reservados.
        outside=[cb[i] for i in range(min(len(cb),len(line["text"]))) if i not in required and not line["text"][i].isspace() and valid_box(cb[i])]
        box=union(boxes)
        if not any(box[0]<b[2] and b[0]<box[2] and box[1]<b[3] and b[1]<box[3] for b in outside):
            return box,"character_boxes_union",[rectangle(b)[0] for b in boxes]
    polygons=[]; covered=set(); partial=False
    for word in line.get("words",[]):
        a,b=int(word.get("start",0)),int(word.get("end",0)); poly=word.get("polygon")
        if not poly or not (a<end and start<b): continue
        polygons.append(poly); covered.update(range(max(start,a),min(end,b)))
        if any(not line["text"][i].isspace() for i in range(max(0,a),min(b,len(line["text"]))) if not start<=i<end): partial=True
    if polygons and required and required<=covered:
        pts=[q for poly in polygons for q in poly]
        box=[min(q[0] for q in pts),min(q[1] for q in pts),max(q[0] for q in pts),max(q[1] for q in pts)]
        return box,"partial_word_approximation" if partial else "word_polygons",polygons
    if line.get("polygon"):
        poly=line["polygon"]; box=[min(q[0] for q in poly),min(q[1] for q in poly),max(q[0] for q in poly),max(q[1] for q in poly)]
        return box,"line_polygon_approximation",[poly]
    if valid_box(line.get("bbox")):
        box=line["bbox"]; return box,"line_bbox_approximation",rectangle(box)
    return None,"no_coordinates",[]

def pad_bbox(b, size, pad=BOX_PADDING_PX):
    if not b:
        return None
    w, h = size
    return [max(0.0, b[0] - pad), max(0.0, b[1] - pad), min(float(w), b[2] + pad), min(float(h), b[3] + pad)]

def build_detections(page, blocks, block_ents):
    lines = {l["line_index"]: l for l in page["lines"]}
    offsets = page.get("line_offsets", {})
    size = (page["width_px"], page["height_px"])
    best = {}                                        # dedupe entre bloques que se solapan
    for block, ents in zip(blocks, block_ents):
        for x in ents:
            frags = tuple(to_fragments(block, x["s"], x["e"]))
            if not frags:
                continue
            key = (frags, x["category"])
            if key not in best or x["score"] > best[key][1]["score"]:
                best[key] = (block["block_id"], x, frags)

    dets, n = [], 0
    for (frags, _), (block_id, x, _f) in sorted(best.items(), key=lambda kv: (kv[1][2][0][0], kv[1][2][0][1])):
        n += 1
        eid = f"{page['source_file']}-p{page['page']}-e{n}"
        policy = politica(x["category"], base=x["basis"], flags=x["flags"])
        for li, ls, le in frags:
            line = lines[li]
            bbox, precision, polygons = geometry_for_span(line, ls, le)
            base = int(offsets.get(str(li), 0))
            dets.append({
                "source_file": page["source_file"], "page": page["page"], "entity_id": eid,
                "block_id": block_id, "line_index": li, "start": ls, "end": le,
                "page_start": base + ls, "page_end": base + le,
                "text": x["text"], "fragment_text": line["text"][ls:le],
                "category": x["category"], "tarjar": policy, "score": x["score"],
                "sources": x["sources"], "classification_basis": x["basis"], "flags": x["flags"],
                "owner_name": x.get("owner"), "owner_role": x.get("owner_role"),
                "ocr_confidence": line.get("ocr_confidence"),
                "line_role": line.get("role"), "bbox_xyxy": bbox, "polygons": polygons,
                "bbox_padded_xyxy": pad_bbox(bbox, size),
                "location_precision": precision, "dpi": page["dpi"],
                "coordinate_system": "rendered_page_pixels_top_left_origin",
            })
    return [aplicar(d) for d in dets]

LABEL_SETS_TEST = {
    "A_solo_person": ["person"],
    "B_config_actual": list(GLINER_LABELS.keys()),     # TODAS las etiquetas en una sola pasada (para comparar con las pasadas)
    # Sinónimos de las etiquetas nuevas: el nombre exacto cambia el resultado
    "C_sinonimos": ["street address", "email address", "national identification number", "birth date",
                    "telephone number", "bank account", "health insurance id", "illness"],
}

for _name, _labs in GLINER_PASSES.items():             # una entrada por pasada, tal como las corre el pipeline
    LABEL_SETS_TEST[_name] = list(_labs.keys())

RAW_THRESHOLD = 0.05      # umbral mínimo para ver "todo lo que GLiNER considera"

PALETTE = ["#ffcdd2", "#bbdefb", "#c8e6c9", "#fff9c4", "#e1bee7", "#ffe0b2", "#b2ebf2", "#d7ccc8", "#f8bbd0", "#dcedc8"]

POLICY_BG = {"si": "#ef9a9a", "revisar": "#ffcc80", "no": "#e0e0e0"}

def _block(text):
    """Bloque como los que arma el notebook A: un tramo por línea del texto pegado."""
    spans, pos = [], 0
    for i, line in enumerate(text.split("\n")):
        spans.append({"line_index": i, "block_start": pos, "block_end": pos + len(line), "line_start": 0})
        pos += len(line) + 1
    return {"block_id": 0, "role": "cuerpo", "text": text, "spans": spans}

def chunk_text(text, max_words=220):
    """Trozos (inicio, fin) de a lo más max_words palabras, cortando en saltos de línea cuando se puede."""
    segs, pos = [], 0
    for line in text.split("\n"):
        words = list(re.finditer(r"\S+", line))
        if len(words) <= max_words:
            segs.append((pos, pos + len(line), len(words)))
        else:
            for i in range(0, len(words), max_words):
                part = words[i:i + max_words]
                segs.append((pos + part[0].start(), pos + part[-1].end(), len(part)))
        pos += len(line) + 1
    chunks, cs, ce, cw = [], None, None, 0
    for s, e, w in segs:
        if cs is not None and cw + w > max_words:
            chunks.append((cs, ce))
            cs, cw = None, 0
        if cs is None:
            cs, cw = s, 0
        ce, cw = e, cw + w
    if cs is not None:
        chunks.append((cs, ce))
    return [(s, e) for s, e in chunks if text[s:e].strip()]

def gliner_raw(view, labels, threshold=RAW_THRESHOLD):
    out = []
    for s, e in chunk_text(view):
        for x in ner_model.predict_entities(view[s:e], labels, threshold=threshold):
            out.append({"start": s + int(x["start"]), "end": s + int(x["end"]),
                        "label": str(x["label"]), "score": float(x["score"])})
    return sorted(out, key=lambda x: (x["start"], -x["score"]))

def gliner_multipass(view, passes=None, threshold=RAW_THRESHOLD):
    """Corre GLiNER una vez por pasada y une los resultados.
    Si la misma etiqueta sale en dos pasadas para el mismo tramo (p. ej. `location`), queda el mayor puntaje."""
    best = {}
    for pname, labels in (passes or GLINER_PASSES).items():
        for r in gliner_raw(view, list(labels.keys()), threshold):
            key = (r["start"], r["end"], r["label"])
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = {**r, "pasada": pname}
    return sorted(best.values(), key=lambda x: (x["start"], -x["score"]))

def es_raw(view):
    out = []
    if es_pipe is None:
        return out
    for s, e in chunk_text(view):
        for r in run_spanish_ner([view[s:e]])[0]:
            out.append({**r, "start": r["start"] + s, "end": r["end"] + s})
    return out

def render_marks(text, marks):
    from IPython.display import HTML
    marks = sorted(marks, key=lambda m: (m["start"], -(m["end"] - m["start"])))
    out, pos = [], 0
    for m in marks:
        if m["start"] < pos:
            continue
        out.append(_html.escape(text[pos:m["start"]]))
        out.append(f'<mark style="background:{m["color"]};color:#000;padding:1px 3px;border-radius:3px" '
                   f'title="{_html.escape(m["tip"])}">{_html.escape(text[m["start"]:m["end"]])}'
                   f'<sub style="font-size:70%"> {_html.escape(m["tag"])}</sub></mark>')
        pos = m["end"]
    out.append(_html.escape(text[pos:]))
    box = ('<div style="white-space:pre-wrap;font-family:monospace;line-height:1.9;border:1px solid #999;'
           'padding:10px;border-radius:6px;background:#fff;color:#000">' + "".join(out) + "</div>")
    return HTML(box)

def analizar(texto, umbral_visual=0.30, usar_pipeline=True, propagar=True, mostrar_conjuntos=True, conjuntos=None):
    """`conjuntos`: lista de nombres de LABEL_SETS_TEST a mostrar (None = todos)."""
    import pandas as pd
    from IPython.display import display
    texto = texto.replace("\r\n", "\n")
    if not texto.strip():
        print("Pega un texto en TEXTO.")
        return
    block = _block(texto)
    view = ner_view(block)

    if mostrar_conjuntos:
        for name, labels in LABEL_SETS_TEST.items():
            if conjuntos and name not in conjuntos:
                continue
            raw = gliner_raw(view, labels)
            colors = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}
            shown = [r for r in raw if r["score"] >= umbral_visual]
            print(f"\n=== GLiNER crudo · {name} · etiquetas: {labels}")
            print(f"    {len(shown)} entidades con puntaje >= {umbral_visual} (de {len(raw)} con puntaje >= {RAW_THRESHOLD})")
            display(render_marks(texto, [{"start": r["start"], "end": r["end"], "color": colors[r["label"]],
                                          "tag": f"{r['label']} {r['score']:.2f}",
                                          "tip": f"{r['label']} · puntaje {r['score']:.3f}"} for r in shown]))
            if raw:
                df = pd.DataFrame([{
                    "texto": texto[r["start"]:r["end"]].replace("\n", " "), "etiqueta": r["label"],
                    "puntaje": round(r["score"], 3),
                    "pasa_umbral_de_B": r["score"] >= GLINER_THRESHOLDS.get(r["label"], GLINER_MIN_THRESHOLD),
                    "inicio": r["start"], "fin": r["end"]} for r in raw])
                display(df)

    if usar_pipeline:
        gl = gliner_multipass(view)
        es = es_raw(view)
        ents = detect_block(block, gl, es)
        if propagar and PROPAGAR_NOMBRES:
            ents = propagate_names([block], [ents])[0]
        print("\n=== PIPELINE COMPLETO (GLiNER en pasadas: " + ", ".join(GLINER_PASSES) + ") ===")
        print("    Rojo = se tarja · naranja = revisar · gris = no se tarja / ignorado")
        marks, rows = [], []
        for x in ents:
            pol = politica(x["category"], base=x["basis"], flags=x["flags"])
            marks.append({"start": x["s"], "end": x["e"], "color": POLICY_BG.get(pol, "#e0e0e0"),
                          "tag": f"{x['category']} {x['score']:.2f}",
                          "tip": f"{x['category']} · {x['basis']} · fuentes {x['sources']} · flags {x['flags']}"})
            rows.append({"texto": x["text"], "categoria": x["category"], "tarjar": pol, "puntaje": x["score"],
                         "fuentes": ", ".join(x["sources"]), "base_de_la_decision": x["basis"],
                         "flags": ", ".join(x["flags"])})
        display(render_marks(texto, marks))
        if rows:
            display(pd.DataFrame(rows))
        else:
            print("El pipeline no detectó nada.")
        if es_pipe is None:
            print("(NER en español desactivado o no disponible: sin vetos ni recall extra de esa fuente.)")

CONTEXT_TEMPLATES = [
    "{x}",
    "{x} fue notificado.",
    "Se notifica a {x} de la presente resolución.",
    "Municipalidad de San {x}",
    "la comuna de San {x}",
    "Don {x} Pérez González",
    "{x} Pérez",
    "Ministerio de {x}",
]

def probar_contexto(palabra, plantillas=None):
    """Puntaje de GLiNER para `palabra` dentro de distintas frases (conjuntos A y B_config_actual)."""
    import pandas as pd
    rows = []
    for tpl in (plantillas or CONTEXT_TEMPLATES):
        text = tpl.format(x=palabra)
        s = text.index(palabra)
        e = s + len(palabra)
        row = {"frase": text}
        for name in ("A_solo_person", "B_config_actual"):
            hits = [r for r in gliner_raw(text, LABEL_SETS_TEST[name]) if r["start"] < e and s < r["end"]]
            best = max(hits, key=lambda r: r["score"]) if hits else None
            row[name] = "no detectado" if best is None else f"{best['label']} @{best['score']:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)

def _bbox_from_polygon(poly):
    pts = [[float(p[0]), float(p[1])] for p in poly]
    return [min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts)]


# Etiquetas completas, no coincidencias por distancia entre filas de una tabla.
_TABLE_FIELDS = [
    (r"(?:run|cedula(?: de identidad)?|identificador personal)(?: (?:del? |de la )?(?:solicitante|representante(?: legal)?|menor|afiliado|titular|compareciente|interesad[oa]|adult[oa]|personal))?(?: \(dv err[oó]neo\)|, variante \d+)?|rut (?:personal|adult[oa]|del representante(?: legal)?)|rut / run adult[oa]", "run_rut_personal"),
    (r"(?:rut (?:de la )?(?:empresa|sociedad|persona juridica)|identificador persona juridica)", "run_rut_juridico"),
    (r"(?:correo|email) (?:personal|particular)(?: .+)?", "correo_personal"),
    (r"(?:correo|email) institucional(?: .+)?|oficina de partes", "correo_institucional"),
    (r"(?:telefono|celular|contacto) (?:personal|particular|del (?:adulto|solicitante))", "telefono_personal"),
    (r"(?:telefono institucional|mesa central)", "telefono_institucional"),
    (r"(?:domicilio|direccion) particular(?: repetida)?", "direccion_personal"),
    (r"direccion (?:del )?(?:inmueble fiscal|bien fiscal|predio fiscal)", "direccion_inmueble"),
    (r"(?:nombre completo de la menor|nombre menor|nombre del menor)", "persona_menor"),
    (r"(?:edad / curso menor|colegio menor|establecimiento del menor)", "dato_menor"),
    (r"nacionalidad(?: de la menor| del menor)?", "nacionalidad_personal"),
    (r"fecha (?:de )?nacimiento", "fecha_nacimiento"),
    (r"estado civil", "estado_civil"),
    (r"(?:cuenta bancaria del solicitante|cuenta bancaria personal|saldo (?:disponible|bancario) personal|ingreso mensual personal)", "dato_financiero_personal"),
    (r"(?:afp del afiliado|cotizacion individual|cotizaciones personales|numero (?:de )?afiliacion)", "dato_previsional_personal"),
    (r"(?:diagnostico titular|tratamiento titular|dato salud menor|tratamiento medico|afiliacion de salud de .+)", "dato_salud_personal"),
    (r"(?:codigo de (?:validacion|verificacion))", "codigo_validacion"),
    (r"vinculo de validacion", "vinculo_validacion"),
    (r"(?:funcionario responsable|nombre funcionari[oa])", "persona_funcionario"),
    (r"(?:nombre adult[oa]|arrendatario|nombre de familiar adulto|vecino en deslinde|profesion(?: declarada)?|numero de garantia|garantia|monto adeudado|profesional informante.*)", "dato_pendiente"),
    (r"(?:plano(?: del inmueble)?|referencia de plano fiscal)", "plano"),
    (r"(?:deslinde )?(?:norte|sur|este|oeste|oriente|poniente)", "deslindes"),
    (r"(?:rol de avaluo|superficie|canon mensual fiscal|fecha (?:acto|del informe|de recepcion)|numero resolucion|numero de resolucion|folio administrativo|documento de referencia|cargo(?: publico)?|entidad general.*)", "identificador_administrativo"),
]


def categoria_rotulo(text):
    f = ' '.join(fold(text).strip(' :：').split())
    named = re.fullmatch(r"(?i:celular|móvil|movil) de ([A-ZÁÉÍÓÚÑ][\wáéíóúñ-]+(?: (?:de |del |la )?[A-ZÁÉÍÓÚÑ][\wáéíóúñ-]+){1,4})",text.strip(" :："))
    if named and nombre_plausible(named[1]) and not re.search(r"\b(?:ministerio|institucion|oficina|municipalidad|servicio|sociedad)\b",fold(named[1])):
        return "telefono_personal"
    return next((cat for rx,cat in _TABLE_FIELDS if re.fullmatch(rx,f)),None)


def valor_compatible(cat, value):
    v=value.strip(); f=fold(v)
    if cat.startswith('run_rut_'): return bool(RUT_RE.fullmatch(v))
    if cat.startswith('correo_'): return bool(EMAIL_RE.fullmatch(v))
    if cat.startswith('telefono_'): return 8<=len(re.sub(r'\D','',v))<=11 and bool(re.fullmatch(r'[+\d ()-]+',v))
    if cat in {'persona_menor','persona_funcionario'}: return nombre_plausible(v)
    if cat=='nacionalidad_personal': return bool(re.fullmatch(_NATIONALITY,f))
    if cat=='fecha_nacimiento': return bool(re.fullmatch(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2} de [a-z]+ de \d{4}',f))
    if cat=='estado_civil': return bool(find_estado_civil(v))
    if cat=='codigo_validacion': return bool(re.fullmatch(r'[\w-]{4,}',v) and re.search(r'\d',v))
    if cat=='vinculo_validacion': return bool(re.fullmatch(r'https?://\S+',v,re.I))
    if cat=='dato_financiero_personal': return bool(re.search(r'\d',v))
    return 2<=len(v)<=160 and categoria_rotulo(v) is None


def pares_tabla(lines, cells=(), details=False):
    """Emparejar solo rótulos a la izquierda con valores de la misma fila; abstenerse ante ambigüedad."""
    def cy(l): return (l['bbox'][1]+l['bbox'][3])/2
    def h(l): return l['bbox'][3]-l['bbox'][1]
    rows=[l for l in lines if l.get('bbox') and l['text'].strip()]
    def cat_of(l):
        f=' '.join(fold(l['text']).split())
        cat=categoria_rotulo(l['text'])
        if f=='curso escolar' and any(categoria_rotulo(x['text'])=='persona_menor' for x in rows): return 'dato_menor'
        return cat
    labels=[l for l in rows if cat_of(l)]
    values=[l for l in rows if not cat_of(l)]
    def containing(line):
        x = (line["bbox"][0]+line["bbox"][2])/2
        y = cy(line)
        matches = [c for c in cells if len(c)==4 and all(isinstance(v,(int,float)) for v in c)
                   and c[0]<=x<c[2] and c[1]<=y<c[3]]
        return matches[0] if len(matches)==1 else None

    proposals = {}
    for label in labels:
        cat = cat_of(label)
        origin = containing(label)
        possible = []
        for value in values:
            if not valor_compatible(cat,value["text"]):
                continue
            target = containing(value)
            if origin and target:
                right = abs(target[0]-origin[2])<=5 and origin[1]<=cy(value)<=origin[3]
                below = abs(target[1]-origin[3])<=5 and min(origin[2],target[2])-max(origin[0],target[0])>.8*(origin[2]-origin[0])
                matches = origin!=target and (right or below)
            else:
                matches = value["bbox"][0]>=label["bbox"][2]-3 and abs(cy(label)-cy(value))<=max(h(label),h(value))*.65
            if matches:
                possible.append(value["line_index"])
        for index in possible:
            proposals.setdefault(index,[]).append((cat,label["text"],len(possible)))
    result = {}
    for index, options in proposals.items():
        cat,label,count = options[0]
        if len(options)>1 or count>1:
            cat = "tabla_ambigua"
        result[index] = (cat,label) if details else cat
    return result


def campos_administrativos(text):
    f=fold(text)
    for m in re.finditer(r'\b(?:resolucion(?:\s+exenta)?(?:\s+de\s+\w+)?|decreto(?:\s+exento)?|oficio)\s+(?:n[°ºo.]\s*|numero\s+)([\w./-]*\d[\w./-]*?)(?=\s|expediente|[;,]|$)|\b(?:expediente|folio administrativo)\s*[:：]?\s*([\w.-]*\d[\w.-]*)',f):
        group=1 if m.group(1) else 2
        start,end = m.span(group)
        # Una coincidencia administrativa no puede terminar dentro de un RUN.
        if re.match(r"[,\.]\d", text[end:]) or (RUT_RE.fullmatch(text[start:end]) and ("-" in text[start:end] or "." in text[start:end])):
            continue
        yield start,end


def filtrar_candidatos(text, ents):
    for a,b in campos_administrativos(text):
        _add_candidate(ents,text,a,b,'identificador_administrativo','identificador_acto_explicito',.99)
    # No permitir que una hipótesis de nombre cree un menor dentro de otro tipo de campo.
    strong=[x for x in ents if x['category'].startswith(('direccion_','correo_','telefono_')) and x['basis'] not in {'solo_gliner','sin_evidencia','contexto_no_vinculado'}]
    out=[]
    for x in ents:
        cat=x['category']; val=fold(x['text']).strip()
        if cat.startswith('persona') and ((len(x['text'].split())>=2 and not nombre_plausible(x['text'])) or any(y['s']<=x['s'] and x['e']<=y['e'] for y in strong)):
            continue
        # Una etiqueta de profesión aislada no basta para convertir encabezados,
        # nombres institucionales o palabras genéricas en campos RNA-13.
        if cat=='dato_pendiente' and x['sources']==['gliner'] and not re.search(r'\b(?:profesion|oficio|garantia|adeudad[oa]|vecino|familiar|tercero|postulante)\b', fold(text[max(0,x['s']-40):min(len(text),x['e']+30)])):
            generic = val in {'ejercicio','ejercicio tecnico','inmueble fiscal','tratamiento'} or any(y['category'].startswith('correo_') and y['s']<=x['s'] and x['e']<=y['e'] for y in ents)
            org = any(y['category'] in {'ignorar:organizacion','ignorar:institucion'} and y['s']<=x['s'] and x['e']<=y['e'] for y in ents)
            if generic or org: continue
        if x['sources']==['gliner'] and cat in {'dato_pendiente','deslindes','direccion_inmueble'}:
            evidence = {'dato_pendiente': 'run_rut_personal', 'deslindes': 'dato_salud_personal',
                        'direccion_inmueble': 'direccion_personal'}[cat]
            if cat=="dato_pendiente" and re.fullmatch(r"(?:run|rut|cedula)\s*[:：]?\s*[\d.,-]+[\dk]?",val):
                if any(y["category"]==evidence and x["s"]<=y["s"] and y["e"]<=x["e"]
                       and y["basis"].startswith("titular_natural") for y in ents):
                    continue
            if any(y['category']==evidence and y['s']<=x['s'] and x['e']<=y['e']
                   and y['basis'] not in {'solo_gliner','requiere_confirmacion'} for y in ents):
                continue
        if cat=='dato_salud' and val in {'tratamiento','diagnostico','salud','enfermedad','medico','medica'}: continue
        out.append(x)
    return out

def _convert_ocr_page(source_page):
    """Convierte el esquema de exportación OCR al formato interno del notebook GLiNER."""
    lines = []
    for item in source_page.get("ocr_lines", []):
        line = {
            "line_index": int(item["line_index"]),
            "text": str(item.get("text", "")),
            "ocr_confidence": item.get("ocr_confidence"),
            "polygon": item.get("polygon"),
            "bbox": item.get("bbox_xyxy") or (_bbox_from_polygon(item["polygon"]) if item.get("polygon") else None),
            "char_boxes": item.get("char_boxes_xyxy"),
            "words": item.get("words", []),
            "role": item.get("layout_label", "unassigned"),
            "layout_block_id": item.get("layout_block_id"),
            "layout_block_order": item.get("layout_block_order"),
        }
        lines.append(line)

    lines_by_id = {line["line_index"]: line for line in lines}
    page_blocks = []
    for region_index, region in enumerate(source_page.get("ordered_regions", [])):
        if region.get("label") == "table":
            cells = [lines_by_id[int(i)] for i in region.get("line_indices", []) if int(i) in lines_by_id]
            pares = pares_tabla(cells,source_page.get("table_cells",[]),details=True)
            for line in cells:
                t=line["text"]
                if not t.strip(): continue
                page_blocks.append({"block_id": region.get("block_id", region_index), "role":"table",
                                    "text":t,"spans":[{"line_index":line["line_index"],"block_start":0,"block_end":len(t),"line_start":0}],
                                    "block_order":region.get("block_order"),"table_category":pares.get(line["line_index"],(None,None))[0],
                                    "table_source_label":pares.get(line["line_index"],(None,None))[1],
                                    "table_label":bool(categoria_rotulo(t))})
            continue
        spans, parts, pos = [], [], 0
        for line_index in region.get("line_indices", []):
            line = lines_by_id.get(int(line_index))
            if line is None:
                continue
            if parts:
                pos += 1  # salto de línea insertado al unir líneas
            text = line["text"]
            spans.append({
                "line_index": int(line_index),
                "block_start": pos,
                "block_end": pos + len(text),
                "line_start": 0,
            })
            parts.append(text)
            pos += len(text)
        block_text = "\n".join(parts)
        if block_text.strip():
            page_blocks.append({
                "block_id": region.get("block_id", region_index),
                "role": region.get("label", "region"),
                "text": block_text,
                "spans": spans,
                "block_order": region.get("block_order"),
            })

    # Evita perder líneas si el JSON contiene líneas OCR que no aparecen en ordered_regions.
    used = {sp["line_index"] for b in page_blocks for sp in b["spans"]}
    unassigned = [ln for ln in lines if ln["line_index"] not in used]
    if unassigned:
        unassigned.sort(key=lambda x: (x["bbox"][1] if x.get("bbox") else 0,
                                       x["bbox"][0] if x.get("bbox") else 0))
        spans, parts, pos = [], [], 0
        for line in unassigned:
            if parts:
                pos += 1
            spans.append({"line_index": line["line_index"], "block_start": pos,
                          "block_end": pos + len(line["text"]), "line_start": 0})
            parts.append(line["text"])
            pos += len(line["text"])
        page_blocks.append({
            "block_id": "unassigned_recovery",
            "role": "unassigned_recovery",
            "text": "\n".join(parts),
            "spans": spans,
            "block_order": None,
        })

    offsets = {}
    running = 0
    for line in sorted(lines, key=lambda x: x["line_index"]):
        offsets[str(line["line_index"])] = running
        running += len(line["text"]) + 1

    return {
        "source_file": source_page.get("source_file", "documento"),
        "page": int(source_page.get("page_number", 1)),
        "width_px": int(source_page.get("image_width_px", 0)),
        "height_px": int(source_page.get("image_height_px", 0)),
        "dpi": int(source_page.get("render_dpi", 220)),
        "pdf_page_rotation": int(source_page.get("pdf_page_rotation", 0)),
        "pdf_page_rect_points": source_page.get("pdf_page_rect_points"),
        "lines": lines,
        "line_offsets": offsets,
        "blocks": page_blocks,
    }

def vista_ocr(text):
    """Mapa de límites de la vista normalizada al texto de la imagen."""
    cuts = set()
    for pattern in (r"(?<=[a-záéíóúñ])(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])",
                    r"(?<=RUN)(?=\d)|(?<=RUT)(?=\d)|(?<=[a-z])(?=RUN(?:\b|\d))|(?<=SIM)(?=Expediente\b)",
                    r"(?<=,)(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])"):
        cuts.update(m.start() for m in re.finditer(pattern,text))
    # Conservar intactos identificadores electrónicos y URL.
    for m in re.finditer(r"\S+@\S+|https?://\S+",text,re.I):
        cuts.difference_update(range(m.start()+1,m.end()))
    chars, boundaries = [], [0]
    for i,char in enumerate(text):
        if i in cuts:
            chars.append(" ")
            boundaries.append(i)
        chars.append(char)
        boundaries.append(i+1)
    return "".join(chars), boundaries


def bloques_con_continuaciones(page):
    """Unir solo párrafos próximos de la misma columna con referencia explícita."""
    lines = {line["line_index"]:line for line in page["lines"]}
    result = []
    narrative = {"paragraph","text","content","region"}
    for block in page["blocks"]:
        previous = result[-1] if result else None
        if previous and block["role"] in narrative and previous["role"] in narrative and re.match(
                r"^(?:su(?:s)?\b|la (?:condicion medica|menor|persona)\b|el interesado\b)",fold(block["text"]).strip()):
            a = lines[previous["spans"][-1]["line_index"]]["bbox"]
            b = lines[block["spans"][0]["line_index"]]["bbox"]
            if a and b and abs(a[0]-b[0])<=2*max(a[3]-a[1],b[3]-b[1]) and 0<=b[1]-a[3]<=2*(b[3]-b[1]):
                offset = len(previous["text"])+1
                previous["text"] += "\n"+block["text"]
                previous["spans"] += [dict(sp,block_start=sp["block_start"]+offset,block_end=sp["block_end"]+offset)
                                      for sp in block["spans"]]
                continue
        result.append(dict(block,spans=list(block["spans"])))
    return result


def analizar_pagina_ocr(source_page, propagar=True):
    page = _convert_ocr_page(source_page)
    page["blocks"] = bloques_con_continuaciones(page)
    block_entities = []
    for block in page["blocks"]:
        if block.get("table_category"):
            t=block["text"]; cat=block["table_category"]
            x=mk(t,0,len(t),cat,.95,["regla_contextual","tabla_geometrica"],"rotulo_tabla_explicito")
            if cat.startswith("persona_"): x.update(owner=t,owner_role="menor" if cat=="persona_menor" else "funcionario")
            if cat == "telefono_personal" and " de " in (block.get("table_source_label") or ""):
                x["owner"] = block["table_source_label"].partition(" de ")[2]
            if cat == "tabla_ambigua":
                x["basis"] = "varias_celdas_compatibles"
                # La ambigüedad del rótulo no debe ocultar los candidatos del valor.
                candidates = detect_block(block,gliner_multipass(ner_view(block)),es_raw(ner_view(block)))
                for candidate in candidates:
                    candidate["flags"].append("conflicto_solapamiento")
                block_entities.append([x,*candidates])
            else:
                block_entities.append([x])
            continue
        if block.get("table_label"):
            block_entities.append([]); continue
        normalized, boundaries = vista_ocr(block["text"])
        starts = {old:new for new,old in enumerate(boundaries)}
        spans = [dict(sp, block_start=starts[sp["block_start"]], block_end=starts[sp["block_end"]])
                 for sp in block["spans"]]
        normalized_block = dict(block,text=normalized,spans=spans)
        view = ner_view(normalized_block)
        candidates = detect_block(normalized_block, gliner_multipass(view), es_raw(view))
        for x in candidates:
            x["s"], x["e"] = boundaries[x["s"]], boundaries[x["e"]]
            x["text"] = block["text"][x["s"]:x["e"]]
        block_entities.append(candidates)

    if propagar and PROPAGAR_NOMBRES and page["blocks"]:
        block_entities = propagate_names(page["blocks"], block_entities)

    detections = build_detections(page, page["blocks"], block_entities)
    return page, block_entities, detections


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
            if len(k)==11 and k.startswith("56"): k=k[2:]
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
                   and "consistencia_documental" not in d.get("sources", [])
                   and d.get("classification_basis") not in {"contexto","sin_evidencia","solo_gliner"}]
        for d in grupo:
            if categorias[d["category"]][1] == "no" and d.get("classification_basis") == "contexto" and d not in opuestas:
                # No propagar una inferencia por proximidad como veto documental.
                d["category"] = {"correo":"correo_revisar","telefono":"telefono_revisar","direccion":"direccion_revisar","rut":"run_rut_revisar"}[categorias[d["category"]][0]]
                d["classification_basis"]="contexto_no_vinculado"; d["tarjar"]="revisar"
        if opuestas or any(d.get("classification_basis") == "titularidad_contradictoria" for d in grupo):
            # Ante conflicto no reafirmar automáticamente una categoría personal.
            if semillas:
                for d in grupo:
                    if categorias[d["category"]][1] is None or "consistencia_documental" in d.get("sources", []):
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



def consolidar_menores(detecciones):
    """Repite solo nombre completo exacto de un menor sin rol contradictorio."""
    groups={}
    for d in detecciones:
        if d['category'] not in {'persona','persona_revisar','persona_menor','persona_funcionario'}: continue
        k=' '.join(fold(d['text']).split())
        if not nombre_plausible(d['text']): continue
        groups.setdefault((d['source_file'],k),[]).append(d)
    for group in groups.values():
        seeds=[d for d in group if d['category']=='persona_menor' and d.get('classification_basis') in {'menor_explicito','rotulo_tabla_explicito'} and d.get('tarjar')=='si']
        if not seeds: continue
        if any(d['category']=='persona_funcionario' for d in group):
            for d in group:
                d['classification_basis']='conflicto_contextual_documental'
            continue
        for d in group:
            if d['category'] in {'persona','persona_revisar','persona_menor'} and d.get('classification_basis') not in {'menor_explicito','rotulo_tabla_explicito'}:
                d.update(category='persona_menor', classification_basis='repeticion_menor_confirmada', owner_role='menor', owner_name=seeds[0]['text'], repetition_seed=seeds[0].get('entity_id'))


def resolver_conflictos_documento(detecciones):
    """Elimina duplicados exactos y bloquea cualquier tarjado que choque con una protección explícita.
    No usar etiquetas genéricas ignorar:lugar como prueba de que el dato sea público.
    """
    unique={}
    for d in detecciones:
        k=(d['source_file'],d['page'],d['line_index'],d['start'],d['end'],d['category'])
        old=unique.get(k)
        if old is None: unique[k]=d; continue
        # Priorizar una evidencia contradictoria sobre una confirmación automática.
        conflict=lambda x: x.get('classification_basis') in {'conflicto_contextual_documental','titularidad_contradictoria'} or 'conflicto_solapamiento' in x.get('flags',[])
        rank=lambda x:(conflict(x),x.get('classification_basis')!='revisar_contexto_repeticion',bool(x.get('owner_name')),x.get('score',0))
        if rank(d)>rank(old): unique[k]=d
    vals=list(unique.values()); by_line={}
    for d in vals: by_line.setdefault((d['source_file'],d['page'],d['line_index']),[]).append(d)
    result=[]
    for group in by_line.values():
        kept=[]
        for d in group:
            if d['category'] != 'identificador_administrativo' and any(x['category']=='identificador_administrativo' and x['start']<=d['start'] and d['end']<=x['end'] for x in group):
                if d.get('classification_basis') not in {'rotulo_tabla_explicito','codigo_validacion_explicito','titular_natural_explicito'}:
                    continue
            generic=[x for x in group if x is not d and x['start']<=d['start'] and d['end']<=x['end'] and d['category'] in _SPECIFIC.get(x['category'],set()) and x['tarjar']!='revisar' and d.get('classification_basis')!='titularidad_contradictoria']
            if generic:
                for x in generic: x['flags'].append('hipotesis_generica_resuelta:'+d['category'])
                continue
            kept.append(d)
        # Usar estado original para no dejar que el primer conflicto oculte los siguientes.
        initial=[d['tarjar'] for d in kept]
        for i,d in enumerate(kept):
            if initial[i]!='si': continue
            for j,other in enumerate(kept):
                if i==j or not _ov(d['start'],d['end'],other['start'],other['end']): continue
                explicit_public=initial[j]=='no' and not other['category'].startswith('ignorar:')
                # Un dato definido como pendiente por RNA-13 no debe quedar cubierto por otra hipótesis.
                pending=other['category'] in {'dato_pendiente','profesion_revisar','garantia_revisar','monto_adeudado_revisar','tabla_ambigua'} and other['start']<=d['start'] and d['end']<=other['end']
                contradictory=other.get('classification_basis')=='titularidad_contradictoria'
                if explicit_public or pending or contradictory:
                    d['flags'].append('conflicto_solapamiento')
                    d['flags'].append('solapa:'+other['category'])
                    other['flags'].append('conflicto_solapamiento')
        result.extend(aplicar(d) for d in kept)
    return sorted(result,key=lambda d:(d['source_file'],d['page'],d['line_index'],d['start'],d['end'],d['category']))


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

def procesar(entrada, salida=None, device="cpu", ner_es=False, modelo_ner_es=None):
    global ner_model, es_pipe
    inicio_total = perf_counter()
    print("Etapa B iniciada: leyendo el JSON del OCR...", flush=True)
    OCR_JSON_PATH = Path(entrada)
    destino = Path(salida) if salida else OCR_JSON_PATH.with_name("sistred_gliner_detections.json")
    csv_privado = destino.with_suffix(".detecciones_privadas.csv")
    if destino.exists() or csv_privado.exists():
        raise ValueError("El archivo de salida o CSV ya existe; use otro nombre")
    OCR_DATA = json.loads(OCR_JSON_PATH.read_text(encoding="utf-8"))
    if OCR_DATA.get("schema_version") != "sistred_ocr_layout_v2" or not OCR_DATA.get("pages"):
        raise ValueError("JSON de OCR incompatible o vacío")
    import importlib
    destino.parent.mkdir(parents=True, exist_ok=True)
    progreso = destino.with_suffix(".progreso.json")
    inicio_carga = perf_counter()
    gliner_module = ejecutar_etapa("Importación de GLiNER y sus dependencias", lambda: importlib.import_module("gliner"), progreso)
    GLiNER = gliner_module.GLiNER
    print(f"Modelo: {NER_MODEL_NAME}. Dispositivo: {device}. La primera ejecución puede descargar pesos.", flush=True)
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise ValueError("Se solicitó CUDA, pero PyTorch no tiene una GPU disponible. Use --device cpu o instale PyTorch con CUDA.")
    ner_model = ejecutar_etapa("Carga de GLiNER (descarga si falta en caché)", lambda: GLiNER.from_pretrained(NER_MODEL_NAME).to(device), progreso)
    es_pipe = None
    if ner_es:
        if not modelo_ner_es:
            raise ValueError("Indique un modelo NER español disponible con --modelo-ner-es. El identificador anterior ya no está disponible.")
        transformers_module = ejecutar_etapa("Importación del NER español", lambda: importlib.import_module("transformers"), progreso)
        es_pipe = ejecutar_etapa("Carga del modelo NER español", lambda: transformers_module.pipeline(
            "token-classification", model=modelo_ner_es, aggregation_strategy="simple", device=device), progreso)
    else:
        print("NER español auxiliar desactivado. Se utilizan GLiNER y las reglas; las propuestas requieren revisión humana.", flush=True)
    segundos_carga = perf_counter() - inicio_carga
    load_gazetteer()
    tipos_documento = {}
    for nombre in {p.get("source_file") for p in OCR_DATA["pages"]}:
        tipo, evidencia = clasificar_acto([p for p in OCR_DATA["pages"] if p.get("source_file") == nombre])
        tipos_documento[nombre] = {"tipo": tipo, "evidencia": evidencia}
        print("Tipo propuesto:", nombre, tipo, evidencia, flush=True)
    OCR_PAGES_INTERNAL = []
    ALL_DETECTIONS = []
    PAGE_ERRORS = []
    tiempos_paginas = []

    for page_idx, source_page in enumerate(OCR_DATA.get("pages", []), start=1):
        label = f"{source_page.get('source_file')} — página {source_page.get('page_number')}"
        print(f"Analizando {label} ({page_idx}/{len(OCR_DATA.get('pages', []))})", flush=True)
        inicio_pagina = perf_counter()
        estado_pagina = "ok"
        try:
            page, block_entities, detections = ejecutar_etapa(f"Detección página {page_idx}", lambda: analizar_pagina_ocr(source_page), progreso)
            OCR_PAGES_INTERNAL.append(page)
            for d in detections:
                d["tipo_acto"] = tipos_documento[d["source_file"]]["tipo"]
                d.update(aplicar(d))
            ALL_DETECTIONS.extend(detections)
            print("  regiones:", len(page["blocks"]), "| detecciones:", len(detections), flush=True)
        except Exception as exc:
            estado_pagina = "error"
            PAGE_ERRORS.append({"source_file": source_page.get("source_file"),
                                "page_number": source_page.get("page_number"),
                                "error": repr(exc),
                                "traceback": traceback.format_exc()})
            print("  ERROR:", repr(exc), flush=True)
        tiempos_paginas.append({"archivo": source_page.get("source_file"),
                                 "pagina": source_page.get("page_number"),
                                 "segundos_analisis_b": round(perf_counter() - inicio_pagina, 6),
                                 "estado_b": estado_pagina,
                                 "error_ocr": bool(source_page.get("processing_error"))})
        parcial = destino.with_suffix(".parcial.json")
        parcial.write_text(json.dumps({"schema_version": "sistred_detections_checkpoint_v1", "pages_processed": page_idx,
                           "detections": ALL_DETECTIONS, "page_errors": PAGE_ERRORS}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTotal detecciones:", len(ALL_DETECTIONS))
    print("Páginas con error:", len(PAGE_ERRORS))

    inicio_consolidacion = perf_counter()
    # Consistencia documental: buscar repeticiones exactas normalizadas de candidatos.
    # Cada aparición continúa siendo una propuesta y conserva su contexto/geometría.
    for document_name in {p["source_file"] for p in OCR_PAGES_INTERNAL}:
        document_pages = [p for p in OCR_PAGES_INTERNAL if p["source_file"] == document_name]
        seeds = {fold(d["text"]): d for d in ALL_DETECTIONS
                 if d["source_file"] == document_name and d.get("tarjar") == "si" and len(d["text"]) >= 4
                 and d["category"] in {"run_rut_personal","correo_personal","telefono_personal","direccion_personal","persona_menor"}
                 and (d["category"] != "persona_menor" or nombre_plausible(d["text"]))
                 and "consistencia_documental" not in d.get("sources",[])}
        for page in document_pages:
            extra = []
            for block in page["blocks"]:
                found = []
                if block.get("table_label") or block.get("table_category"):
                    extra.append(found); continue
                for value, seed in seeds.items():
                    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in value.split()) + r"(?!\w)"
                    for match in re.finditer(pattern, fold(block["text"])):
                        if seed["category"] == "persona_menor" and refine_person(block["text"],match.start(),match.end())[0] != "persona": continue
                        found.append(mk(block["text"], match.start(), match.end(), seed["category"],
                                        seed["score"], ["consistencia_documental"], "revisar_contexto_repeticion"))
                extra.append(found)
            existing = {(d["line_index"], d["start"], d["end"], d["category"]) for d in ALL_DETECTIONS
                        if d["source_file"] == document_name and d["page"] == page["page"]}
            for d in build_detections(page, page["blocks"], extra):
                key = (d["line_index"], d["start"], d["end"], d["category"])
                if key not in existing and not any(li==d["line_index"] and a<=d["start"] and d["end"]<=b for li,a,b,c in existing):
                    d["tarjar"] = "revisar"
                    d["tipo_acto"] = tipos_documento[document_name]["tipo"]
                    d["entity_id"] += "-repeticion"
                    ALL_DETECTIONS.append(d)
                    existing.add(key)

    ALL_DETECTIONS = resolver_conflictos_documento(ALL_DETECTIONS)
    consolidar_repeticiones_seguras(ALL_DETECTIONS)
    consolidar_menores(ALL_DETECTIONS)
    ALL_DETECTIONS = resolver_conflictos_documento([aplicar(d) for d in ALL_DETECTIONS])

    segundos_consolidacion = perf_counter() - inicio_consolidacion
    # Identificador único por fragmento final, estable para enlazar B con C.
    conteos = {}
    for i, d in enumerate(ALL_DETECTIONS, 1):
        d["id_deteccion"] = f"B{i:07d}"
        conteos.setdefault((d["source_file"], d["page"]), Counter())[d.get("tarjar", "desconocido")] += 1
    for t in tiempos_paginas:
        t["fragmentos_por_propuesta"] = dict(conteos.get((t["archivo"], t["pagina"]), {}))
    tiempos = {(t["archivo"], t["pagina"]): t["segundos_analisis_b"] for t in tiempos_paginas}
    inicio_csv = perf_counter()
    escribir_csv(csv_privado, COLUMNAS, (fila_base(d, tiempos) for d in ALL_DETECTIONS))
    segundos_csv = perf_counter() - inicio_csv
    DETECTIONS_OUTPUT = {
        "rendimiento": {"device_solicitado": device,
                         "segundos_carga_modelos": round(segundos_carga, 6),
                         "segundos_consolidacion": round(segundos_consolidacion, 6),
                         "segundos_csv": round(segundos_csv, 6),
                         "segundos_b_hasta_csv": round(perf_counter() - inicio_total, 6),
                         "paginas": tiempos_paginas},
        "tipos_documento": tipos_documento,
        "schema_version": "sistred_gliner_detections_v1",
        "source_schema": OCR_DATA.get("schema_version"),
        "coordinate_system": OCR_DATA.get("coordinate_system", {
            "units": "pixels", "origin": "top_left"
        }),
        "model": NER_MODEL_NAME,
        "secondary_ner": {"enabled": ner_es, "model": modelo_ner_es if ner_es else None},
        "rules_version": RULES_VERSION,
        "source_ocr_sha256": hashlib.sha256(OCR_JSON_PATH.read_bytes()).hexdigest(),
        "detections": ALL_DETECTIONS,
        "page_errors": PAGE_ERRORS,
    }


    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destino.open("x", encoding="utf-8") as archivo:
            json.dump(DETECTIONS_OUTPUT, archivo, ensure_ascii=False, indent=2)
    except BaseException:
        csv_privado.unlink(missing_ok=True)
        raise
    progreso.write_text(json.dumps({"estado": "completo", "paginas": len(OCR_DATA["pages"]), "errores": len(PAGE_ERRORS)}), encoding="utf-8")
    print("Detecciones guardadas:", destino.resolve())
    print("CSV privado de detecciones:", csv_privado.resolve())
    return destino

def main(argv=None):
    parser = argparse.ArgumentParser(description="B: detectar candidatos a tarjado en el OCR local")
    parser.add_argument("entrada", type=Path, help="diagnostico_ocr_layout.json")
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu o cuda:0 según la instalación")
    parser.add_argument("--sin-ner-es", action="store_true", help="Mantener desactivado el auxiliar NER (comportamiento predeterminado)")
    parser.add_argument("--modelo-ner-es", help="Activar un modelo auxiliar disponible con etiquetas PER/LOC/ORG")
    args = parser.parse_args(argv)
    try:
        procesar(args.entrada, args.salida, args.device, bool(args.modelo_ner_es) and not args.sin_ner_es, args.modelo_ner_es)
    except (ImportError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\nRevise las dependencias y las rutas.\n")

if __name__ == "__main__":
    main()
