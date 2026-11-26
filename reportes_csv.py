"""Informes privados para evaluación del OCR/tarjado. Nunca son publicables.
CSV UTF-8 con BOM y separador ; (Excel en español). Una fila por fragmento.
Los campos *_json conservan el valor original, incluso ceros y signos iniciales.
"""
import csv
import json
from pathlib import Path


def _json(valor):
    return json.dumps(valor, ensure_ascii=False, separators=(',', ':'))


def _celda(valor):
    if valor is None:
        return ''
    # Evitar fórmulas al abrir datos OCR no confiables en una hoja de cálculo.
    if isinstance(valor, str) and (valor.startswith(('\t', '\r', '\n')) or valor.lstrip().startswith(('=', '+', '-', '@'))):
        return "'" + valor
    return valor


COLUMNAS = [
    'id_deteccion', 'id_entidad', 'archivo', 'pagina', 'bloque', 'linea_ocr',
    'inicio_linea', 'fin_linea', 'inicio_pagina', 'fin_pagina',
    'texto_entidad', 'texto_fragmento', 'texto_entidad_json', 'texto_fragmento_json',
    'categoria', 'propuesta_b', 'tipo_acto', 'reglas', 'version_reglas',
    'puntaje_deteccion', 'confianza_ocr', 'fuentes', 'fundamento', 'flags',
    'titular', 'titular_tipo', 'precision_b', 'requiere_revision_geometria_b',
    'bbox_b_json', 'poligonos_b_json', 'dpi', 'semilla_repeticion',
    'segundos_analisis_pagina_b',
]
COLUMNAS_C = COLUMNAS + [
    'id_revision', 'categoria_c', 'reglas_c', 'propuesta_c', 'decision_c', 'resultado_c', 'motivo',
    'tipo_pendiente', 'color_recuadro', 'recuadro_json', 'precision_c',
    'requiere_revision_geometria_c', 'poligonos_finales_json', 'franjas_aplicadas_json',
]


def fila_base(d, tiempos=None):
    t = d.get('text')
    f = d.get('fragment_text')
    key = (d.get('source_file'), d.get('page'))
    return dict(zip(COLUMNAS, [
        d.get('id_deteccion'), d.get('entity_id'), d.get('source_file'), d.get('page'),
        d.get('block_id'), d.get('line_index'), d.get('start'), d.get('end'),
        d.get('page_start'), d.get('page_end'), t, f, _json(t), _json(f),
        d.get('category'), d.get('tarjar'), d.get('tipo_acto'), _json(d.get('rule_ids', [])),
        d.get('rules_version'), d.get('score'), d.get('ocr_confidence'),
        _json(d.get('sources', [])), d.get('classification_basis'), _json(d.get('flags', [])),
        d.get('owner_name') or d.get('owner') or d.get('titular') or d.get('owner_id') or d.get('holder_id'),
        d.get('owner_role') or d.get('titular_tipo') or d.get('owner_type') or d.get('holder_type'),
        d.get('location_precision'), bool(d.get('requires_geometry_review')),
        _json(d.get('bbox_xyxy')), _json(d.get('polygons', [])), d.get('dpi'),
        d.get('repetition_seed'), (tiempos or {}).get(key),
    ]))


def fila_resultado(original, final, tiempos=None, *, recuadro=None):
    # En el modo manual puede haber zonas nuevas sin candidato B.
    base = fila_base(original or {}, tiempos)
    if original is None:
        for k in ('archivo', 'pagina', 'categoria'):
            base[k] = final.get({'archivo':'source_file','pagina':'page','categoria':'category'}[k])
    decision = final.get('decision')
    tipo = final.get('motivo_pendiente_tipo')
    resultado = ('TARJADO_APLICADO' if decision == 'aceptar' else
                 'CONSERVADO' if decision == 'rechazar' else
                 'PENDIENTE_GEOMETRIA' if tipo == 'geometria' else
                 'PENDIENTE_CLASIFICACION' if tipo == 'clasificacion' else 'PENDIENTE')
    base.update(id_revision=final.get('review_id'), categoria_c=final.get('category'),
                reglas_c=_json(final.get('rule_ids', [])), propuesta_c=final.get('tarjar'),
                decision_c=decision, resultado_c=resultado,
                motivo=final.get('motivo_pendiente') or final.get('motivo'),
                tipo_pendiente=tipo,
                color_recuadro=('azul' if tipo == 'geometria' else 'naranja') if recuadro is not None else '',
                recuadro_json=_json(recuadro), precision_c=final.get('location_precision'),
                requiere_revision_geometria_c=bool(final.get('requires_geometry_review')),
                poligonos_finales_json=_json(final.get('polygons', [])),
                franjas_aplicadas_json=_json(final.get('polygons', []) if decision == 'aceptar' else []))
    return base


def escribir_csv(destino, columnas, filas):
    """No sobreescribe archivos. Limpia un CSV incompleto si falla la escritura."""
    destino = Path(destino)
    creado = False
    try:
        with destino.open('x', encoding='utf-8-sig', newline='') as f:
            creado = True
            w = csv.DictWriter(f, fieldnames=columnas, delimiter=';', quoting=csv.QUOTE_ALL)
            w.writeheader()
            for fila in filas:
                w.writerow({k: _celda(v) for k, v in fila.items()})
    except BaseException:
        if creado:
            destino.unlink(missing_ok=True)
        raise
    return destino
