from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
import cv2
import os
import io
import math


class CreatePdfPanaf():
    
    FONT_TO_PIXEL_RATIO=0.75

    def __init__(self, p_out_pdf, p_text_opacity, p_convert_to_jpeg, p_jpeg_ratio):        
        self.out_pdf=p_out_pdf
        self.can = canvas.Canvas(self.out_pdf)
        self.text_opacity=p_text_opacity
        self.convert_to_jpeg=p_convert_to_jpeg
        self.jpeg_ratio=p_jpeg_ratio
        self.color_image = Color( 0, 0, 0, alpha=1.0)
        self.color_transparent = Color( 0, 0, 0, alpha=self.text_opacity)
        
        
    def add_page(self, p_main_image_file, p_tiles, p_rescale_size_ratio=1.0):
        #text=p_tiles["result"]
        #box=p_tiles["box"]
        image_ori = cv2.imread(p_main_image_file)
        new_width=image_ori.shape[1]
        new_height=image_ori.shape[0]
        max_line_height=self.get_max_line_height(p_tiles, new_height)
        filename, file_extension = os.path.splitext(p_main_image_file)
        print(file_extension)
        if file_extension.lower not in ['.jpg', ".jpeg"] and self.convert_to_jpeg:
            print("JPEG")
            is_success, buffer = cv2.imencode(".jpg", image_ori, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_ratio])
            pdf_tmp_buffer=ImageReader(io.BytesIO(buffer.tobytes()))     
            image= cv2.imdecode(buffer,cv2.IMREAD_COLOR)
            self.can.setFillColor(self.color_image)
            self.can.setPageSize((new_width,new_height))
            self.can.drawImage(pdf_tmp_buffer, 0, 0, width=new_width, height=new_height)                   
            j=0
            self.can.setFillColor(self.color_transparent  )
            self.draw_text(self.can, p_tiles,CreatePdfPanaf.FONT_TO_PIXEL_RATIO, new_height, new_width,max_line_height, p_rescale_size_ratio )
        else:
            print("KEEP")
            self.can.setFillColor(self.color_image)
            self.can.setPageSize((new_width,new_height))
            self.can.drawImage(p_main_image_file, 0, 0, width=new_width, height=new_height)
            self.can.setFillColor(self.color_transparent)            
            self.draw_text(self.can, p_tiles, CreatePdfPanaf.FONT_TO_PIXEL_RATIO, new_height, new_width,max_line_height,p_rescale_size_ratio )
     
    def get_max_line_height(self,p_data, p_height, p_ratio_text=0.9 ):
        """
        nb_lines=len(p_data)
        print(f"{nb_lines=}")
        p_new_height=math.floor(p_height*p_ratio_text)
        print(f"{p_new_height=}")
        max_line_size=math.floor(p_new_height/(nb_lines*4))
        print(f"{max_line_size=}")
        """
        nb_lines=0
        for text in p_data:
            tmp=text['result']
            if isinstance(tmp, list):
                acc=[]
                for t in tmp:
                    acc.append(t)
                tmp=" ".join(acc)
            exploded=tmp.split("\n")
            nb_lines=nb_lines+len(exploded)
        print(f"{nb_lines=}")
        p_new_height=math.floor(p_height*p_ratio_text)
        print(f"{p_new_height=}")
        max_line_size=math.floor(p_new_height/(nb_lines*4))
        print(f"{max_line_size=}")
        return max_line_size
        
    def draw_text(self, p_canvas, p_data, p_font_to_pixel_ratio, p_height, p_width, p_max_line_height, p_rescale_size_ratio=1.0):        
        for data in p_data:
            box=data["box"]
            text=data["result"]
            if isinstance(text,list):
                acc=[]
                for t in text:
                    acc.append(t)
                text=" ".join(acc)
            print(box)
            line_height=(box["y2"]-box["y1"])
            print(line_height)
            line_height=line_height*p_rescale_size_ratio
            print(line_height)
            left=box["x1"]
            top=box["y1"]
            print(p_font_to_pixel_ratio)
            print(p_rescale_size_ratio)
            font_size=line_height*p_rescale_size_ratio*p_font_to_pixel_ratio 
            font_size=min(font_size, p_max_line_height)
            str_y=left*p_rescale_size_ratio
            str_y=max(str_y,10)
            str_x=math.floor(p_height)-(top*p_rescale_size_ratio)-(line_height*p_rescale_size_ratio)
            p_canvas.setFont("Courier", font_size)  
            print(str_x)
            sys.exit()
            p_canvas.drawString( int(str_y), int(str_x) ,text)
            print("font_size")
            print(font_size)
            print(str_y)
            print(str_x)
            print(text)
            
        p_canvas.showPage()  
        print("PAGES")
        
    def commit(self):
        self.can.save()    
            
            
            