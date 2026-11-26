"""Regresiones V5. Ejecutar con python -m unittest -v test_v5."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

import deteccion_b as b
import ocr_a as a
from deteccion_visual import regiones_visuales
from reglas import politica
from tarjado_c import Sesion


def detectar(text, gl=()):
    return b.detect_block(dict(b._block(text),role="text"),gl,[])


def pares(text):
    return {(x["category"],x["text"]) for x in detectar(text)}


class Reglas(unittest.TestCase):
    def test_offsets_originales(self):
        raw = "ElRUN17,896,543-4 y correo aBc@example.invalid"
        view,offset = b.vista_ocr(raw)
        self.assertIn("El RUN 17,896,543-4",view)
        self.assertIn("aBc@example.invalid",view)
        hit = next(x for x in detectar(view) if x["category"]=="run_rut_personal")
        self.assertEqual(raw[offset[hit["s"]]:offset[hit["e"]]],"17,896,543-4")

    def test_telefono(self):
        for text in ("celular 912345678","mesa central 912345678"):
            hits = pares(text)
            self.assertTrue(any(c.startswith("telefono") for c,_ in hits))
            self.assertFalse(any(c.startswith("run_rut") for c,_ in hits))
        self.assertIn(("telefono_institucional","912345678"),pares("mesa central 912345678"))

    def test_titulares_rut(self):
        self.assertIn(("run_rut_personal","12.345.678-5"),pares("RUT del representante legal: 12.345.678-5"))
        self.assertIn(("run_rut_juridico","76.123.456-0"),pares("RUT de la empresa: 76.123.456-0"))

    def test_administrativo(self):
        text = "Resolución exenta N° 117-SIM expediente:12,345,678-5."
        self.assertEqual([text[s:e] for s,e in b.campos_administrativos(text)],["117-SIM"])

    def test_direcciones(self):
        raw = "La persona declara residir en Sector Quebrada Inventada, parcela sin número."
        self.assertIn(("direccion_personal","Sector Quebrada Inventada, parcela sin número"),pares(raw))
        self.assertIn(("direccion_inmueble","Camino Fiscal 55"),pares("El inmueble fiscal ubicado en Camino Fiscal 55."))
        self.assertEqual(politica("direccion_inmueble","arriendo"),"no")
        self.assertEqual(politica("direccion_inmueble","destinacion_ffaa"),"si")

    def test_nacionalidad(self):
        self.assertIn(("nacionalidad_personal","chilena"),pares("Su nacionalidad es chilena."))
        self.assertNotIn(("nacionalidad_personal","chilena"),pares("Se aplica la legislación chilena."))

    def test_menor(self):
        text = "La menor Elisa Prado Ficticia cursa séptimo básico y asiste al Colegio Horizonte y reside en Pasaje Modelo 482."
        hits = pares(text)
        self.assertIn(("dato_menor","séptimo básico"),hits)
        self.assertIn(("dato_menor","Colegio Horizonte"),hits)

    def test_diagnostico(self):
        text = "El solicitante Arturo Prado tiene cuenta bancaria personal 1234567. La condición médica consignada es diabetes mellitus tipo 2."
        hits = detectar(text)
        self.assertTrue(any(x["category"]=="dato_salud_personal" and x.get("owner")=="Arturo Prado" for x in hits))
        self.assertFalse(any(c=="dato_salud_personal" for c,_ in pares("La condición médica consignada es diabetes mellitus tipo 2.")))

    def test_misma_columna(self):
        def line(i,box): return {"line_index":i,"bbox":box}
        def block(i,text,role): return dict(b._block(text),role=role,spans=[dict(line_index=i,block_start=0,block_end=len(text),line_start=0)])
        page = {"lines":[line(0,[0,0,100,10]),line(1,[0,12,100,22])],
                "blocks":[block(0,"El solicitante Arturo Prado declara.","text"),block(1,"Su condición médica es diabetes.","text")]}
        self.assertEqual(len(b.bloques_con_continuaciones(page)),1)
        page["blocks"][0]["role"] = "title"
        self.assertEqual(len(b.bloques_con_continuaciones(page)),2)

    def test_tablas(self):
        def line(i,t,box): return {"line_index":i,"text":t,"bbox":box}
        label = line(0,"Celular de Inés Lago Inventada",[2,2,80,16])
        phone = line(1,"+56 9 2345 6789",[4,24,85,38])
        self.assertEqual(b.pares_tabla([label,phone],[[0,0,100,20],[0,20,100,40]]),{1:"telefono_personal"})
        phone["bbox"] = [100,2,160,16]
        other = line(2,"912345678",[170,2,240,16])
        self.assertEqual(set(b.pares_tabla([label,phone,other]).values()),{"tabla_ambigua"})

    def test_hipotesis_generica(self):
        raw = "RUN 12345678-5"
        explicit = next(x for x in detectar(raw) if x["category"]=="run_rut_personal")
        weak = b.mk(raw,4,len(raw),"dato_pendiente",.85,["gliner"],"requiere_confirmacion")
        self.assertEqual(b.filtrar_candidatos(raw,[explicit,weak]),[explicit])


class Imagenes(unittest.TestCase):
    def test_franja_inclinada(self):
        session = object.__new__(Sesion)
        session.imagenes = [Image.new("RGB",(140,100))]
        poly = [[5,15],[110,40],[110,50],[5,25]]
        adjacent = [[5,35],[110,60],[110,70],[5,45]]
        hit = dict(page=1,polygons=[poly],start=0,end=4,line_index=0)
        lines = {0:dict(text="dato",polygon=poly),1:dict(polygon=adjacent)}
        self.assertTrue(session._franja_automatica(hit,lines))
        lines[1]["polygon"] = poly
        with self.assertRaises(ValueError):
            session._franja_automatica(hit,lines)

    def test_geometria_parcial(self):
        line = {"text":"RUN12345","words":[dict(start=0,end=8,polygon=[[0,0],[10,0],[10,5],[0,5]])]}
        self.assertEqual(b.geometry_for_span(line,3,8)[1],"partial_word_approximation")

    def test_inclinacion(self):
        image = np.full((500,700,3),255,np.uint8)
        for y in range(60,450,50):
            cv2.line(image,(60,y),(600,y+38),(0,0,0),6)
        with patch.object(a,"cv2",cv2,create=True),patch.object(a,"np",np,create=True):
            _,inv = a.preparar_analisis(image)
            self.assertIsNotNone(inv)
            pts = np.array([[60.,60.],[600.,98.]])
            forward = cv2.invertAffineTransform(inv)
            deskew = pts @ forward[:,:2].T + forward[:,2]
            self.assertLess(abs(deskew[0,1]-deskew[1,1]),2)
            np.testing.assert_allclose(deskew @ inv[:,:2].T + inv[:,2],pts,atol=.001)

    def test_firmas(self):
        image = Image.new("RGB",(320,150),"white")
        for text,basis in [("Firma del solicitante","firma_particular_propuesta"),("Firma institucional","firma_institucional_propuesta")]:
            meta = {"layout_blocks":[dict(label="image",bbox_xyxy=[20,20,170,70])],
                    "ocr_lines":[dict(text=text,bbox_xyxy=[20,75,170,90])]}
            cat,_,why,_ = regiones_visuales(image,meta)[0]
            self.assertEqual((cat,why),("firma",basis))
            self.assertEqual(politica(cat,base=why),"revisar")

    def test_firma_atribuida(self):
        image = Image.new("RGB",(320,150),"white")
        for label,role,decision in [("Firma del solicitante","natural","si"),("Firma institucional","funcionario","no")]:
            meta = {"layout_blocks":[dict(label="signature",bbox_xyxy=[20,20,170,70])],
                    "ocr_lines":[dict(text=label+" Juan Soto",line_index=4,bbox_xyxy=[20,75,170,90])]}
            person = dict(category="persona",owner_name="Juan Soto",owner_role=role,line_index=4)
            category,_,basis,_ = regiones_visuales(image,meta,[person])[0]
            self.assertEqual(politica(category,base=basis),decision)

    def test_qr_pdf(self):
        code = cv2.QRCodeEncoder_create().encode("https://validacion.example.invalid/actos/X123")
        code = cv2.resize(code,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
        code = cv2.copyMakeBorder(code,30,30,30,30,cv2.BORDER_CONSTANT,value=255)
        size = code.shape[0]
        buf = io.BytesIO()
        Image.fromarray(code).convert("RGB").save(buf,format="PNG")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root/"original.pdf"
            with fitz.open() as doc:
                page = doc.new_page(width=size,height=size)
                page.insert_image(page.rect,stream=buf.getvalue())
                doc.save(original)
                pix = page.get_pixmap(alpha=False)
            sha = lambda data: hashlib.sha256(data).hexdigest()
            meta = dict(source_file=original.name,source_sha256=sha(original.read_bytes()),page_number=1,
                        render_dpi=72,render_sha256=sha(pix.samples),image_width_px=size,image_height_px=size,ocr_lines=[])
            ocr,det = root/"ocr.json",root/"det.json"
            ocr.write_text(json.dumps(dict(schema_version="sistred_ocr_layout_v2",pages=[meta])))
            det.write_text(json.dumps(dict(schema_version="sistred_gliner_detections_v1",source_ocr_sha256=sha(ocr.read_bytes()),detections=[])))
            session = Sesion(original,ocr,det)
            self.assertEqual(session.detecciones[0]["classification_basis"],"qr_validacion")
            dest = session.exportar_automatico(root/"exportado.pdf")
            with fitz.open(dest) as doc:
                self.assertEqual(len(doc),2)
                self.assertFalse(doc[0].get_text())
                pix = doc[0].get_pixmap(alpha=False)
                pixels = np.frombuffer(pix.samples,np.uint8).reshape(size,size,3)
                self.assertFalse(cv2.QRCodeDetector().detectAndDecode(pixels)[0])
                self.assertEqual(pixels[size//2,size//2].tolist(),[0,0,0])
