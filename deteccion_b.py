"""Etapa B: detección local sobre el JSON de OCR; no altera el PDF."""
import argparse
import re
import json
import hashlib
import traceback
import unicodedata
import html as _html
from pathlib import Path
from reglas import aplicar, candidatos_adicionales, VERSION as RULES_VERSION
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
        "phone number": "telefono",
        "mobile phone number": "telefono",
    },
    "P5_contexto": {
        "nationality": "nacionalidad",
        "property boundary": "deslindes",
        "military facility name": "recinto",
        "profession": "dato_pendiente",
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

PROPAGAR_NOMBRES = True

PROPAGATE_MIN_SCORE = 0.55

PROPAGAR_VARIANTES = True          # primer+segundo, último+penúltimo y primero+último de nombres de >= 3 palabras

PROPAGAR_TOKENS_SUELTOS = False    # propagar palabras sueltas (apellidos) como persona_revisar

POLITICA = {
    "persona": "no",
    "persona_revisar": "revisar",
    "direccion_personal": "si",
    "direccion_revisar": "revisar",
    "direccion_inmueble": "no",        # PENDIENTE de criterio
    "direccion_institucional": "no",        # confirmado: no se tarja
    "correo_personal": "si",
    "correo_institucional": "no",      # PENDIENTE de criterio
    "correo_revisar": "revisar",
    "run_rut": "revisar",
    "run_rut_personal": "si",
    "run_rut_juridico": "no",
    "codigo_validacion": "si",
    "run_rut_revisar": "revisar",
    "telefono_personal": "si",
    "telefono_institucional": "no",
    "telefono_revisar": "revisar",
    "estado_civil": "si",
    "fecha_nacimiento": "si",          # solo con cue de nacimiento (RNA-06)
    # RNA-05: financiero / previsional / salud. Quedan en "revisar" hasta medir la precisión de GLiNER; luego pasar a "si".
    "dato_financiero": "revisar",
    "dato_previsional": "revisar",
    "dato_salud": "revisar",
}

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
    if _domain_matches(domain, PERSONAL_EMAIL_DOMAINS):
        return "correo_personal", "dominio_personal"
    if _domain_matches(domain, INSTITUTIONAL_EMAIL_DOMAINS):
        return "correo_institucional", "dominio_institucional"
    kind = nearest_group(fmasked, s, e, CUES_EMAIL, CUE_MAX_DISTANCE)
    if kind == "personal":
        return "correo_personal", "contexto"
    if kind == "institucional":
        return "correo_institucional", "contexto"
    return "correo_revisar", "sin_evidencia"

RUT_RE = re.compile(r"(?<![\w.,])(\d{1,2}(?:[.,]\d{3}){2}|\d{7,8})\s*-?\s*([\dkK])(?![\w])")

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
    r"conviviente"
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

def _prio(cat):
    if cat.startswith(("run_rut", "correo", "telefono", "fecha_nacimiento")):
        return 3
    if cat.startswith(("direccion", "dato_")):
        return 2
    return 1

def resolve_overlaps(ents):
    # Conservar hipótesis superpuestas para decisión humana (no perder reservas).
    unique = {}
    for x in ents:
        key = (x["s"], x["e"], x["category"])
        if key not in unique or x["score"] > unique[key]["score"]:
            unique[key] = x
    return sorted(unique.values(), key=lambda x: x["s"])

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
    rut_spans = [(m["s"], m["e"]) for m in ruts]
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
        cat, basis = clasificar_rut(text, m["s"], m["e"], m["valid"])
        ents.append(mk(text, m["s"], m["e"], cat, m["score"], ["regex"], basis))

    for p in find_phones(text, email_spans + rut_spans):
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
            digits = re.sub(r"\D", "", text[s:e])
            left = ftext[max(0, s - 25):s]
            if (len(digits) < ID_MIN_DIGITS or NON_ID_CONTEXT_RE.search(left)
                    or _overlaps_any(s, e, regex_spans)):
                continue
            ents.append(mk(text, s, e, kind, score * 0.8, ["gliner"], "solo_gliner"))
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
        elif kind in {"nacionalidad", "deslindes", "recinto", "dato_pendiente"}:
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
    return resolve_overlaps(ents)

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
    """
    Recupera polígonos originales de palabras que se cruzan con [start, end).
    Cuando no hay geometría de palabra, usa cajas por carácter si existen.
    Como último recurso devuelve el polígono/caja de línea y marca la aproximación.
    """
    selected_polygons = []
    selected_boxes = []
    covered = set()

    for word in line.get("words", []):
        if int(word.get("start", 0)) < end and start < int(word.get("end", 0)):
            poly = word.get("polygon")
            if poly:
                covered.update(range(max(start, int(word["start"])), min(end, int(word["end"]))))
                selected_polygons.append([[float(p[0]), float(p[1])] for p in poly])
                box = word.get("bbox_xyxy")
                if box:
                    selected_boxes.append([float(v) for v in box])

    required = {i for i in range(start, end) if not line["text"][i].isspace()}
    if selected_polygons and required <= covered:
        if selected_boxes:
            bbox = [min(b[0] for b in selected_boxes), min(b[1] for b in selected_boxes),
                    max(b[2] for b in selected_boxes), max(b[3] for b in selected_boxes)]
        else:
            pts = [p for poly in selected_polygons for p in poly]
            bbox = [min(p[0] for p in pts), min(p[1] for p in pts),
                    max(p[0] for p in pts), max(p[1] for p in pts)]
        return bbox, "word_polygons", selected_polygons

    cb = line.get("char_boxes")
    if cb:
        sel = [b for b in cb[start:end] if b is not None]
        if sel and len(cb) >= end and all(cb[i] is not None for i in required):
            bbox = [min(b[0] for b in sel), min(b[1] for b in sel),
                    max(b[2] for b in sel), max(b[3] for b in sel)]
            x0, y0, x1, y1 = bbox
            poly = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            return bbox, "character_boxes_union", [poly]

    if line.get("polygon"):
        poly = [[float(p[0]), float(p[1])] for p in line["polygon"]]
        pts = poly
        bbox = [min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts)]
        return bbox, "line_polygon_approximation", [poly]
    if line.get("bbox"):
        bbox = list(map(float, line["bbox"]))
        x0, y0, x1, y1 = bbox
        return bbox, "line_bbox_approximation", [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]
    return None, "no_coordinates", []

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
        policy = POLITICA.get(x["category"], "no" if x["category"].startswith("ignorar") else "revisar")
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
            pol = POLITICA.get(x["category"], "no" if x["category"].startswith("ignorar") else "revisar")
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

def analizar_pagina_ocr(source_page, propagar=True):
    page = _convert_ocr_page(source_page)
    block_entities = []
    for block in page["blocks"]:
        view = ner_view(block)
        gl_entities = gliner_multipass(view)
        es_entities = es_raw(view)
        block_entities.append(detect_block(block, gl_entities, es_entities))

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
    print("Etapa B iniciada: leyendo el JSON del OCR...", flush=True)
    OCR_JSON_PATH = Path(entrada)
    destino = Path(salida) if salida else OCR_JSON_PATH.with_name("sistred_gliner_detections.json")
    if destino.exists():
        raise ValueError("El archivo de salida ya existe; use otro nombre")
    OCR_DATA = json.loads(OCR_JSON_PATH.read_text(encoding="utf-8"))
    if OCR_DATA.get("schema_version") != "sistred_ocr_layout_v2" or not OCR_DATA.get("pages"):
        raise ValueError("JSON de OCR incompatible o vacío")
    import importlib
    destino.parent.mkdir(parents=True, exist_ok=True)
    progreso = destino.with_suffix(".progreso.json")
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
    load_gazetteer()
    tipos_documento = {}
    for nombre in {p.get("source_file") for p in OCR_DATA["pages"]}:
        tipo, evidencia = clasificar_acto([p for p in OCR_DATA["pages"] if p.get("source_file") == nombre])
        tipos_documento[nombre] = {"tipo": tipo, "evidencia": evidencia}
        print("Tipo propuesto:", nombre, tipo, evidencia, flush=True)
    OCR_PAGES_INTERNAL = []
    ALL_DETECTIONS = []
    PAGE_ERRORS = []

    for page_idx, source_page in enumerate(OCR_DATA.get("pages", []), start=1):
        label = f"{source_page.get('source_file')} — página {source_page.get('page_number')}"
        print(f"Analizando {label} ({page_idx}/{len(OCR_DATA.get('pages', []))})", flush=True)
        try:
            page, block_entities, detections = ejecutar_etapa(f"Detección página {page_idx}", lambda: analizar_pagina_ocr(source_page), progreso)
            OCR_PAGES_INTERNAL.append(page)
            for d in detections:
                d["tipo_acto"] = tipos_documento[d["source_file"]]["tipo"]
                d.update(aplicar(d))
            ALL_DETECTIONS.extend(detections)
            print("  regiones:", len(page["blocks"]), "| detecciones:", len(detections), flush=True)
        except Exception as exc:
            PAGE_ERRORS.append({"source_file": source_page.get("source_file"),
                                "page_number": source_page.get("page_number"),
                                "error": repr(exc),
                                "traceback": traceback.format_exc()})
            print("  ERROR:", repr(exc), flush=True)
        parcial = destino.with_suffix(".parcial.json")
        parcial.write_text(json.dumps({"schema_version": "sistred_detections_checkpoint_v1", "pages_processed": page_idx,
                           "detections": ALL_DETECTIONS, "page_errors": PAGE_ERRORS}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTotal detecciones:", len(ALL_DETECTIONS))
    print("Páginas con error:", len(PAGE_ERRORS))

    # Consistencia documental: buscar repeticiones exactas normalizadas de candidatos.
    # Cada aparición continúa siendo una propuesta y conserva su contexto/geometría.
    for document_name in {p["source_file"] for p in OCR_PAGES_INTERNAL}:
        document_pages = [p for p in OCR_PAGES_INTERNAL if p["source_file"] == document_name]
        seeds = {fold(d["text"]): d for d in ALL_DETECTIONS
                 if d["source_file"] == document_name and d.get("tarjar") != "no" and len(d["text"]) >= 4}
        for page in document_pages:
            extra = []
            for block in page["blocks"]:
                found = []
                for value, seed in seeds.items():
                    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in value.split()) + r"(?!\w)"
                    for match in re.finditer(pattern, fold(block["text"])):
                        found.append(mk(block["text"], match.start(), match.end(), seed["category"],
                                        seed["score"], ["consistencia_documental"], "revisar_contexto_repeticion"))
                extra.append(found)
            existing = {(d["line_index"], d["start"], d["end"], d["category"]) for d in ALL_DETECTIONS
                        if d["source_file"] == document_name and d["page"] == page["page"]}
            for d in build_detections(page, page["blocks"], extra):
                key = (d["line_index"], d["start"], d["end"], d["category"])
                if key not in existing:
                    d["tarjar"] = "revisar"
                    d["tipo_acto"] = tipos_documento[document_name]["tipo"]
                    d["entity_id"] += "-repeticion"
                    ALL_DETECTIONS.append(d)
                    existing.add(key)

    consolidar_repeticiones_seguras(ALL_DETECTIONS)
    ALL_DETECTIONS = [aplicar(d) for d in ALL_DETECTIONS]

    DETECTIONS_OUTPUT = {
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
    destino.write_text(json.dumps(DETECTIONS_OUTPUT, ensure_ascii=False, indent=2), encoding="utf-8")
    progreso.write_text(json.dumps({"estado": "completo", "paginas": len(OCR_DATA["pages"]), "errores": len(PAGE_ERRORS)}), encoding="utf-8")
    print("Detecciones guardadas:", destino.resolve())
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
