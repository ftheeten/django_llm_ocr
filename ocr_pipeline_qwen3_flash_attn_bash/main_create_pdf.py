import argparse
import traceback
import os
import json
import datetime
from lib.create_pdf import CreatePdfPanaf
import sys


#python main_create_pdf.py --project_name=panafgeo_mrac_004 --input_folder=/opt/panafgeo_ict/src/panafgeo_franck_2026/out --out=my_pdf --opacity=1.0

class CustomException(Exception):
    def __init__(self,msg):
        #self.msg=msg
        super().__init__(msg)
        
        
class CreatePdf():
    
    def process(self, p_project_name, p_input_folder, p_out_pdf, p_opacity, p_convert_to_jpeg=True, p_jpeg_ratio=90):
        self.project_name=p_project_name
        self.input_folder=p_input_folder
        self.out_pdf=p_out_pdf
        self.opacity=p_opacity
        self.convert_to_jpeg=p_convert_to_jpeg
        self.jpeg_ratio=p_jpeg_ratio
        new_dir=os.path.join(self.input_folder, self.project_name)
        self.new_dir_ocr=os.path.join(new_dir, "ocr")
        self.result_file=os.path.join(self.new_dir_ocr, "ocr_result.json")
        if not os.path.exists(self.result_file):
            raise CustomException("Error : OCR file not found") 
        with open(self.result_file, 'r') as f_in:
            data = json.load(f_in)            
            if "images" in data:
                pdf_obj=CreatePdfPanaf(p_out_pdf,self.opacity, self.convert_to_jpeg, self.jpeg_ratio)
                for  image_obj in data["images"]: 
                    print(image_obj)                    
                    path=image_obj["main_image_file"]
                    tiles=image_obj["tiles"]
                    pdf_obj.add_page(path, tiles)
                pdf_obj.commit()
                
    
   
if __name__ == "__main__": 
    try:
        parser = argparse.ArgumentParser()  
        parser.add_argument("--project_name",help= "project_name", required=True)  
        parser.add_argument("--input_folder",help= "input folder", required=True) 
        parser.add_argument("--out_pdf",help= "output_document", required=True) 
        parser.add_argument("--opacity",help= "Opacity", default="0.0") 
        args = parser.parse_args()
        if not os.path.exists(args.input_folder):
            raise CustomException("Error : Input folder not found")         
        opacity=float(args.opacity)    
        create_pdf=CreatePdf()
        create_pdf.process(args.project_name, args.input_folder, args.out_pdf, opacity )
    except Exception as e:
        print("EXCEPTION")
        print(e)
        print(traceback.print_exc())
        print(traceback.print_stack())
        print(traceback.format_exc())        