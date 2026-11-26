"""Ejecuta OCR, detección y tarjado automático para una carpeta de PDF."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter


PROYECTO = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROYECTO / "pdfs")
    parser.add_argument("--salida", type=Path, default=PROYECTO / "tarjados")
    parser.add_argument("--device", default="gpu:0", help="Dispositivo para el OCR: gpu:0 o cpu")
    args = parser.parse_args(argv)

    if not args.entrada.is_dir():
        parser.error(f"No existe la carpeta de entrada: {args.entrada}")
    if args.entrada.resolve() == args.salida.resolve():
        parser.error("Indique carpetas distintas para los originales y las copias")
    pdfs = sorted(pdf for pdf in args.entrada.iterdir()
                  if pdf.is_file() and pdf.suffix.lower() == ".pdf")
    if not pdfs:
        parser.error(f"No se encontraron PDF en {args.entrada}")

    args.salida.mkdir(parents=True, exist_ok=True)
    resultados = PROYECTO / "resultados"
    resultados.mkdir(exist_ok=True)
    entorno = os.environ.copy()
    entorno.setdefault("PADDLE_PDX_CACHE_HOME", str(PROYECTO / ".paddlex-cache"))
    entorno.setdefault("HF_HOME", str(PROYECTO / ".hf-cache"))
    completados = omitidos = fallidos = 0
    paginas_completadas = 0
    tiempo_completados = 0.0
    inicio_lote = perf_counter()

    for indice, pdf in enumerate(pdfs, 1):
        print(f"\n[{indice}/{len(pdfs)}] {pdf.name}", flush=True)
        salida = args.salida / f"{pdf.stem}_tarjado_dudoso.pdf"
        registro = salida.with_suffix(".revision_privada.json")
        if salida.is_file() and registro.is_file():
            print("Omitido: ya existen la copia y su registro.")
            omitidos += 1
            continue
        if salida.exists() or registro.exists():
            print(f"Error: salida incompleta en {args.salida}. Use otra carpeta de salida.", file=sys.stderr)
            fallidos += 1
            continue

        inicio_documento = perf_counter()
        trabajo = Path(mkdtemp(prefix="lote_", dir=resultados))
        ocr = trabajo / "diagnostico_ocr_layout.json"
        detecciones = trabajo / "sistred_gliner_detections.json"
        etapas = (
            ("ocr_a.py", pdf, "--salida", trabajo, "--device", args.device),
            ("deteccion_b.py", ocr, "--device", "cpu"),
            ("tarjado_c.py", pdf, ocr, detecciones, "--automatico", salida),
        )
        for script, *argumentos in etapas:
            proceso = subprocess.run([sys.executable, "-u", PROYECTO / script, *argumentos], env=entorno)
            if proceso.returncode:
                if proceso.returncode in {-2, 130, 0xC000013A, -1073741510}:
                    return 130
                print(f"Falló {script} para {pdf.name}. Diagnósticos: {trabajo}", file=sys.stderr)
                fallidos += 1
                break
        else:
            completados += 1
            tiempo_completados += perf_counter() - inicio_documento
            with ocr.open(encoding="utf-8") as archivo:
                paginas_completadas += len(json.load(archivo)["pages"])

    tiempo_total = perf_counter() - inicio_lote
    print(f"\nCompletados: {completados} | Omitidos: {omitidos} | Fallidos: {fallidos}")
    print(f"Tiempo total del lote: {tiempo_total:.2f} s")
    if paginas_completadas:
        promedio = tiempo_completados / paginas_completadas
        print(f"Promedio por página: {promedio:.2f} s ({paginas_completadas} páginas completadas)")
    else:
        print("Promedio por página: sin páginas completadas")
    return int(fallidos > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
