import argparse
import configparser
import traceback
import os
import json
import datetime
from lib.classifier import Classifier
import sys



#python main_classification.py --project_name=arnaud_ocr --input_folder=/opt/panafgeo_ict/src/panafgeo_franck_2026/out --mode='camembert'

class CustomException(Exception):
    def __init__(self,msg):
        #self.msg=msg
        super().__init__(msg)
        
class ClassificationParser():   
    
    list_mode={"CAMEMBERT":
                    {"local_path":"/opt/panafgeo_ict/src/local_models/camembert",
                     "model_name": "Jean-Baptiste/camembert-ner"
                    },
               "OPEN_MED_SPECIES":
                   {
                     "local_path":"/opt/panafgeo_ict/src/local_models/openmed_ner_speciesdetect",
                     "model_name": "OpenMed/OpenMed-NER-SpeciesDetect-BigMed-560M"
                   },
               "GEOBERT":
                   {
                     "local_path":"/opt/panafgeo_ict/src/local_models/geobert",
                     "model_name": "botryan96/GeoBERT"
                   }
        }
    
    def process(self, p_project_name, p_input_folder, p_mode, p_aggregation_strategy):
        self.elems={}
        self.project_name=p_project_name
        self.input_folder=p_input_folder
        self.mode=p_mode
        self.model_path=ClassificationParser.list_mode[self.mode]
        self.classifier=Classifier(self.model_path["local_path"], p_aggregation_strategy)
        new_dir=os.path.join(self.input_folder, self.project_name)
        print(self.input_folder)
        self.new_dir_ocr=os.path.join(new_dir, "ocr")
        self.new_dir_classification=os.path.join(new_dir, "classification")
        print(self.new_dir_ocr)
        print( self.new_dir_classification)
        os.makedirs(self.new_dir_classification, exist_ok=True )
        self.result_file=os.path.join(self.new_dir_ocr, "ocr_result.json")
        if not os.path.exists(self.result_file):
            raise CustomException("Error : OCR file not found") 
        with open(self.result_file, 'r') as f_in:
            data = json.load(f_in)
            if "images" in data:
                for  image_obj in data["images"]:                    
                    #print(image_obj)
                    self.process_element(image_obj)
        print(self.elems)
        
        results=self.classifier.process(self.elems)
        json_result_file=os.path.join(self.new_dir_classification, "classification_result.json")
        print(results)
        general_result={}
        current_date=datetime.datetime.now().isoformat()
        general_result["ocr_params"]={"model":self.model_path["model_name"], "date_classification":current_date }
        general_result["results"]=results
        print(json_result_file)
        with open(json_result_file, 'w') as f:
            json.dump(general_result, f, indent=4)  
        
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
                #print(result)
                self.elems[tile_file]=result
        
    
if __name__ == "__main__": 
    try:
        parser = argparse.ArgumentParser()  
        parser.add_argument("--project_name",help= "project_name", required=True)  
        parser.add_argument("--input_folder",help= "input folder", required=True) 
        parser.add_argument("--mode",help= "input folder", required=True) 
        parser.add_argument("--aggregation_strategy",help= "Aggregation strategy for NER", default="first") 
        args = parser.parse_args()
        if not os.path.exists(args.input_folder):
            raise CustomException("Error : Input folder not found") 
        if not args.mode in ClassificationParser.list_mode.keys():
            available_modes=','.join(ClassificationParser.list_mode.keys())
            raise CustomException("Error : mode not supported. Possible values : "+available_modes) 
            
        classification=ClassificationParser()
        classification.process(args.project_name, args.input_folder, args.mode,args.aggregation_strategy )
    except Exception as e:
        print("EXCEPTION")
        print(e)
        print(traceback.print_exc())
        print(traceback.print_stack())
        print(traceback.format_exc())        