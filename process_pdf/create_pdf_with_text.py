import os
from pathlib import Path
import json
import re
import io
import numpy as np
import cv2
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
from collections import OrderedDict
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
#from reportlab.pdfbase.pdfmetrics import getAscent, getDescent

SRC_PDF="PremierePartie1a21700_p1_10.pdf"
OUTPUT_FOLDER_IMG="./output/images"
OUTPUT_FOLDER_TEXT="./output/text"
OUTPUT_FOLDER_OUT="./output/out_pdf"
FONT_TO_PIXEL_RATIO=0.75

ACQUISITION_WIDTH=1250

def fit_paragraph_to_box(text, width, height, min_font=6, max_font=24, style_name="Normal"):
    print("call paragraph")
    """
    Crée un Paragraph Platypus qui tient dans une zone de largeur x hauteur en ajustant la taille de police.
    
    Arguments :
        text : str - le texte à afficher
        width : float - largeur disponible en points
        height : float - hauteur disponible en points
        min_font : int - taille minimale de police autorisée
        max_font : int - taille maximale de police
        style_name : str - nom du style à utiliser depuis getSampleStyleSheet()
    
    Retour :
        Paragraph prêt à être utilisé
    """
    
    styles = getSampleStyleSheet()
    base_style = styles[style_name]    
    # On part de la taille maximale
    font_size = max_font
    
    while font_size >= min_font:    
        print("font_size")
        print(font_size)
        style = ParagraphStyle(
            name="fitStyle",
            parent=base_style,
            fontSize=font_size,
            leading=int(font_size * 1.2)  # interligne = 120% de la police
        )
        para = Paragraph(text, style)
        
        w, h = para.wrap(width, height)
        if h <= height and w < width:
            print("returned")
            return para
        font_size -= 1  # réduire la police et réessayer
    print("font_size")
    print(font_size)
    # Si on arrive ici, même la police minimale dépasse la zone
    style = ParagraphStyle(
        name="fitStyle",
        parent=base_style,
        fontSize=min_font,
        leading=int(min_font * 1.2)
    )

    return Paragraph(text, style)
 
def define_paragraph(text, font_size=24, style_name="Normal"):   
    styles = getSampleStyleSheet()
    base_style = styles[style_name]    
    # On part de la taille maximale
    style = ParagraphStyle(
            name="fitStyle",
            parent=base_style,
            fontSize=font_size,
            leading=int(font_size * 1.2)  # interligne = 120% de la police
        )
    para = Paragraph(text, style)
    return para
    
def get_text_height(p_font, p_size):
    ascent = getAscent(p_font) * p_size / 1000
    descent = abs(getDescent(p_font) * p_size / 1000)


def pix_to_image(pix):
    bytes = np.frombuffer(pix.samples, dtype=np.uint8)
    img = bytes.reshape(pix.height, pix.width, pix.n)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
    
def w_ratio(width, reference_width):
    print(width)
    print(reference_width)
    ratio=width/reference_width
    print(ratio)
    return ratio
    
def process_page(f_path, p_pdf_page, p_canvas, p_size_ratio=1, p_opacity=0.0):
    global ACQUISITON_WIDTH
    print("------------------------------------")
    print("PAGE")
    pix = page.get_pixmap()
    image=pix_to_image(pix) 
    global FONT_TO_PIXEL_RATIO
    if image is not None:
        color_image = Color( 0, 0, 0, alpha=1.0)
        color_transparent = Color( 0, 0, 0, alpha=p_opacity)
        p_canvas.setFillColor(color_image)
        print("----->")
        print(image.shape)
        ref_h, ref_w, _ =image.shape
        resize_ratio=w_ratio(ref_w, ACQUISITION_WIDTH)
        p_canvas.setPageSize((ref_w, ref_h)) 
        is_success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        pdf_tmp_buffer=ImageReader(io.BytesIO(buffer.tobytes()))
        p_canvas.drawImage(pdf_tmp_buffer, 0, 0, width=ref_w, height=ref_h) 
        p_canvas.setFillColor(color_transparent)
        
        with open(f_path) as f:
            d = json.load(f)            
            if "results" in d:
                for key, elem in d["results"].items():
                    bbox=None
                    text_raw=None
                    status=False
                    if "status" in elem:
                        if elem["status"]=="ok":
                            status=True
                    if "bbox" in elem:
                        bbox=elem["bbox"]
                    if "text_raw" in elem:
                        text_raw=elem["text_raw"]
                    if status and text_raw is not None:
                        print(bbox)
                        min_h, max_h, min_w, max_w=bbox
                        text_raw=re.sub("\n+", "\n", text_raw)
                        text_raw=re.sub("\n", "<br/>", text_raw)
                        #text_raw=re.sub("\n", "", text_raw)
                        print(text_raw)
                        rel_h=max_h-min_h
                        rel_w=max_w-min_w
                        rel_h=rel_h*resize_ratio
                        rel_w=rel_w*resize_ratio
                        style = getSampleStyleSheet()["Normal"]
                        
                        min_w=min_w*resize_ratio
                        max_h=max_h*resize_ratio
                        base_h=ref_h-max_h
                        #p =  fit_paragraph_to_box(text_raw, max_w, rel_h, min_font=6, max_font=24, style_name="Normal")
                        p=define_paragraph(text_raw, 10)
                        print([min_w, max_h, rel_w, rel_h])
                        frame = Frame(min_w, base_h, rel_w, rel_h)
                        frame.addFromList([p], p_canvas)
                        #font_size=rel_h*FONT_TO_PIXEL_RATIO
                        #can.setFont("Courier", font_size)
                    else:
                        print("segment not acquired")
                        min_h, max_h, min_w, max_w=bbox                       
                        rel_h=max_h-min_h
                        rel_w=max_w-min_w
                        p_canvas.setFont("Courier", 10)                            
                        p_canvas.drawString( min_w, max_h ,"segment_not_acquired")  
    p_canvas.showPage()
    print("endpage")

if __name__ == "__main__":
    existing_pdf = fitz.open(SRC_PDF)
    nb_pages=existing_pdf.page_count
    print(nb_pages)
    file_name_pdf=Path(os.path.basename(SRC_PDF)).stem
    output_folder_text_pdf=OUTPUT_FOLDER_TEXT+"/"+file_name_pdf
    output_folder_out_pdf=OUTPUT_FOLDER_OUT+"/"+file_name_pdf
    print(output_folder_text_pdf)
    list_files=[]
    for file in os.listdir(output_folder_text_pdf):    
        if file.endswith(".json"):
            f=os.path.join(output_folder_text_pdf, file)
            list_files.append(f)
            
    list_files=sorted(list_files)
    print(list_files)
    dict_files=OrderedDict()
    for f in list_files:
        file_name_pdf_pdf=Path(os.path.basename(f)).stem
        tmp=file_name_pdf_pdf.split("_")
        if len(tmp)>1:
            last=tmp[-1]
            if last.isnumeric():
                last_int=int(last)
                dict_files[last_int]=f
    print(dict_files)
    can = canvas.Canvas(output_folder_out_pdf+"_ocr.pdf")
    for i in range(0, nb_pages):
        if i in dict_files:
            page = existing_pdf[i]   
            process_page(dict_files[i], page, can)
        else:
            print("page_not scanned")
    can.save()
    print("save")
    """
    for i in list_files:
        process_page(f)
    """