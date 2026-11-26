"""Propuestas en coordenadas del render que C verifica contra el PDF."""
import re
import unicodedata

import cv2
import numpy as np


def normalizar(text):
    return "".join(c for c in unicodedata.normalize("NFD",text.casefold())
                   if not unicodedata.combining(c))


def rectangulo(box):
    x,y,r,b = box
    return [[x,y],[r,y],[r,b],[x,b]]


def categoria_rotulo(text):
    text = normalizar(text)
    for word,category in ((r"firma","firma"),(r"fotografia|foto","fotografia"),
                          (r"huella(?: dactilar)?","huella"),(r"timbre|sello","timbre")):
        if re.match(r"^(?:"+word+r")\b", text.strip()):
            return category


def regiones_visuales(image, meta, personas=()):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    height,width = gray.shape
    detector = cv2.QRCodeDetector()
    multiple,points = detector.detectMulti(gray)
    if not multiple:
        single,points = detector.detect(gray)
        if not single:
            points = []
    regions = []
    for corners in points:
        if len(corners)!=4 or not np.isfinite(corners).all():
            continue
        try:
            payload, _ = detector.decode(gray, corners.reshape(1,4,2))
        except cv2.error:
            payload = ""
        # No guardar en el registro el contenido del QR.
        validation = bool(re.search(r"validaci[oó]n|validar|verificaci[oó]n|verificar",payload,re.I))
        poly = np.rint(corners).astype(int).tolist()
        if not all(0<=x<width and 0<=y<height for x,y in poly):
            continue
        regions.append(("qr",poly,"qr_validacion" if validation else "qr_sin_finalidad",None))
    lines = [line for line in meta.get("ocr_lines",[]) if line.get("bbox_xyxy")]
    labels = [(line,categoria_rotulo(line["text"])) for line in lines if categoria_rotulo(line["text"])]
    boxes = [b["bbox_xyxy"] for b in meta.get("layout_blocks",[])
             if b.get("label","").lower() in {"image","figure","seal","picture","signature","stamp"} and b.get("bbox_xyxy")]
    if labels:
        # Excluir letras antes de agrupar los trazos próximos a los rótulos.
        ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        for line in lines:
            x,y,r,b = map(int,line["bbox_xyxy"])
            cv2.rectangle(ink,(x,y),(r,b),0,-1)
        k = max(3,round(meta.get("render_dpi",220)/24))
        joined = cv2.dilate(ink,np.ones((k,k),np.uint8))
        contours,_ = cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w>=3*k and h>=2*k and w<.75*width and h<.45*height:
                if not any(l<=x+w/2<=r and t<=y+h/2<=b for l,t,r,b in boxes):
                    boxes.append([x,y,x+w-1,y+h-1])
    for x,y,r,b in boxes:
        x,y,r,b = max(0,x),max(0,y),min(width-1,r),min(height-1,b)
        if x>=r or y>=b:
            continue
        if any(all(x<=qx<=r and y<=qy<=b for qx,qy in poly) for cat,poly,_,_ in regions if cat=="qr"):
            continue
        near = []
        for line,category in labels:
            l,t,right,bottom = line["bbox_xyxy"]
            gap = max(y-bottom,t-b,0)
            if gap<=3*(bottom-t) and max(x,l)<min(r,right):
                near.append((line,category))
        if not near and [x,y,r,b] not in [item.get("bbox_xyxy") for item in meta.get("layout_blocks",[])]:
            continue
        category = near[0][1] if len(near)==1 else "imagen"
        caption = near[0][0]["text"] if len(near)==1 else ""
        basis = "region_visual_por_confirmar"
        if category=="firma":
            if re.search(r"\b(?:solicitante|particular|arrendatari[oa])\b",normalizar(caption)):
                basis = "firma_particular_propuesta"
            elif re.search(r"\b(?:funcionari[oa]|seremi|ministr[oa]|institucional)\b",normalizar(caption)):
                basis = "firma_institucional_propuesta"
        # Exigir región especializada, nombre y rol para resolver una firma.
        signature = any(block.get("label","").lower()=="signature" and block.get("bbox_xyxy")==[x,y,r,b]
                        for block in meta.get("layout_blocks",[]))
        if signature and category=="firma" and len(near)==1:
            line_id = near[0][0].get("line_index")
            signers = {(person.get("owner_name"),person.get("owner_role")) for person in personas
                       if line_id is not None and person.get("line_index")==line_id
                       and person.get("category","").startswith("persona") and person.get("owner_name")
                       and normalizar(person["owner_name"]) in normalizar(caption)}
            crosses_text = any(max(x,l)<min(r,right) and max(y,t)<min(b,bottom)
                               for l,t,right,bottom in (line["bbox_xyxy"] for line in lines))
            if len(signers)==1 and not crosses_text:
                _,role = signers.pop()
                if role=="funcionario" and basis=="firma_institucional_propuesta":
                    basis = "firma_institucional_confirmada"
                elif role in {"natural","menor"} and basis=="firma_particular_propuesta":
                    basis = "firma_particular_confirmada"
        regions.append((category,rectangulo([x,y,r,b]),basis,caption))
    return regions
