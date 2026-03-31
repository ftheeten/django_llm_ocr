import argparse
import configparser
import traceback
import os
import json
import pandas as pnd


class CustomException(Exception):
    def __init__(self,msg):
        #self.msg=msg
        super().__init__(msg)
        
        
class DocumentBuilder():
    
    list_mode=["SIMPLE_CSV", "SIMPLE_EXCEL"]
    
    """
    def __init__(self):
        self.mode=p_mode
    """    
        
    def process(self, p_project_name, p_input_folder, p_output_file, p_mode):
        self.elems={}
        self.project_name=p_project_name
        self.input_folder=p_input_folder
        self.output_file=p_output_file
        self.mode=p_mode
        new_dir=os.path.join(self.input_folder, self.project_name)
        self.new_dir_ocr=os.path.join(new_dir, "ocr")
        self.result_file=os.path.join(self.new_dir_ocr, "ocr_result.json")
        if not os.path.exists(self.result_file):
            raise CustomException("Error : OCR file not found") 
        with open(self.result_file, 'r') as f_in:
            data = json.load(f_in)
            if "images" in data:
                for  image_obj in data["images"]:
                    self.process_element(image_obj)
        print("---------")            
        print(self.elems)
        print("---------")     
        if p_mode=="SIMPLE_EXCEL":
            self.write_excel(p_output_file)
        
    
    def write_excel(self, p_name_file):
        writer = pnd.ExcelWriter( p_name_file, engine='xlsxwriter')
        excel_pages=[]
        for key, texts in self.elems.items():
            print(key)
            print(texts)
            
            df = pnd.DataFrame()
            df["text"]=""
            for text in texts:
                print(text)
                rows=text.replace("\r","\n").replace("\n\n","\n").split("\n")
                print(rows)
                for row in rows:
                    df.loc[len(df)]=[row]
                    #df.index = df.index + 1
                #df = df.sort_index()
            print(df) 
            excel_pages.append(df)                
            for i, frame in enumerate(excel_pages):
                print(i)
                print(frame)
                sheet_name="page_"+str(i).zfill(3)
                print(sheet_name)
                frame.to_excel(writer, sheet_name = sheet_name, index=False)
        
        writer.close()        
                
    def process_element(self, p_image_obj):        
        #print(p_image_obj)
        main_image_file=p_image_obj["main_image_file"]
        tiles=p_image_obj["tiles"]
        print(main_image_file)
        #print(tiles)
        for tile in tiles:
            if "result" in tile and "tile_file" in tile:
                tile_file=tile["tile_file"]
                result=tile["result"]
                print(result)
                self.elems[tile_file]=result

if __name__ == "__main__": 
    try:
        parser = argparse.ArgumentParser()  
        parser.add_argument("--project_name",help= "project_name", required=True)  
        parser.add_argument("--input_folder",help= "input folder", required=True) 
        parser.add_argument("--output_file",help= "input folder", required=True) 
        parser.add_argument("--mode",help= "input folder", required=True) 
        args = parser.parse_args()
        if not os.path.exists(args.input_folder):
            raise CustomException("Error : Input folder not found") 
        if not args.mode in DocumentBuilder.list_mode:
            available_modes=','.join(DocumentBuilder.list_mode)
            raise CustomException("Error : mode not supported. Possible values : "+available_modes) 
        builder=DocumentBuilder()
        builder.process(args.project_name, args.input_folder, args.output_file, args.mode)
    except Exception as e:
        print("EXCEPTION")
        print(e)
        print(traceback.print_exc())
        print(traceback.print_stack())
        print(traceback.format_exc())         