import argparse
import configparser
import traceback
import os
import cv2
import math
import json
import sys
from collections import OrderedDict
import datetime
from lib.segmentation_lines import SegmentationLines
from lib.segmentation_columns import SegmentationColumns
from lib.remove_lines import RemoveLines
from lib.ocr_qwen3_8b import OcrQwen3b8
from lib.ocr_tesseract import OcrTesseract

#python main.py --project_name=arnaud_ocr_debug --input_folder=/opt/panafgeo_ict/src/panafgeo_franck_2026/src/arnaud_ocr_debug/ --output_folder=/opt/panafgeo_ict/src/panafgeo_franck_2026/out  --remove_v_h_lines --len_token=8092


class CustomException(Exception):
    def __init__(self,msg):
        #self.msg=msg
        super().__init__(msg)
        #print( 'custom exception occurred')
        #print(self.msg)

class OCRParser():
    
    ocr_types=["QWEN3_8B_VL", "PYTESSERACT_WORD", "PYTESSERACT_LINES"]
    
    def __init__(self, p_ocr_mode, p_autorotate, p_autorotate_max, p_mode_grey_contrast, p_grey_threshold_min, p_grey_threshold_max, p_grey_blur_kernel):
        self.ocr_mode=p_ocr_mode
        self.autorotate=p_autorotate
        self.autrotate_max=p_autorotate_max
        self.mode_grey_contrast=p_mode_grey_contrast
        self.grey_threshold_min=p_grey_threshold_min
        self.grey_threshold_max=p_grey_threshold_max
        self.rey_blur_kernel=p_grey_blur_kernel
    
    def pad_box_x(self, p_box, min_x, max_x, ratio_pad_left=0.1, ratio_pad_right=0.5, force_min_x=None, force_max_x=None):
        x1, y1, x2, y2=p_box
        width=x2-x1
        pad_left=width*ratio_pad_left
        pad_right=width*ratio_pad_right
        if force_min_x is None:
            new_x1=math.ceil(max(min_x, x1-pad_left))
        else:
            new_x1=force_min_x
        if force_max_x is None:
            new_x2=math.floor(min(max_x, x2+ratio_pad_left))
        else:
            new_x2=force_max_x
        return [new_x1, y1, new_x2, y2 ]
      
    def split_columns(self, p_img, p_min_width=10, ratio_pad_left=0.1, ratio_pad_right=0.5, force_min_x=None, force_max_x=None):
        seg=SegmentationColumns(p_img)
        bboxes=seg.process()
        print(bboxes)
        height, width, _ = p_img.shape
        i=0
        max_i=len(bboxes)
        bboxes2=[]
        for i in range(0, max_i):
            box=bboxes[i]
            small=p_img[ box[1]:box[3], box[0]:box[2]]       
            if (box[2]- box[0])>p_min_width:
                if i==0:
                    box_pad=self.pad_box_x(box, 0, width, force_min_x=0)
                elif i==max_i-1:
                    box_pad=self.pad_box_x(box, 0, width, force_max_x=width)
                else:     
                    box_pad=self.pad_box_x(box, 0, width)            
                small_padded=p_img[ box_pad[1]:box_pad[3], box_pad[0]:box_pad[2]]
                bboxes2.append(box_pad)
            i=i+1
        return bboxes2
        
    
    
    def process(self, p_prompt, p_project_name, p_input_folder, p_output_folder, p_len_token, p_segmentation_columns, p_segmentation_lines, p_remove_h_lines, p_remove_v_lines, p_remove_v_h_lines,p_remove_h_v_lines, ratio_pad_left=0.1, ratio_pad_right=0.5, force_min_x=None, force_max_x=None):   
        self.project_name=p_project_name
        self.input_folder=p_input_folder
        self.output_folder=p_output_folder
        json_file=self.process_img(p_segmentation_columns, p_segmentation_lines, p_remove_h_lines, p_remove_v_lines, p_remove_v_h_lines, p_remove_h_v_lines, ratio_pad_left, ratio_pad_right, force_min_x, force_max_x)
        print(json_file)
        self.read_img_json(p_prompt, json_file, p_len_token)
       
    def read_img_json(self,  p_prompt, p_file, p_len_token=4092):
        current_date=datetime.datetime.now().isoformat()
        json_result={}        
        json_result["ocr_params"]={}
        
        json_result["ocr_params"]["len_token"]=p_len_token
        json_result["ocr_params"]["date_ocr"]=current_date
        json_result["images"]=[]
        print("submission")
        with open(p_file, 'r') as f_in:
            data = json.load(f_in)
            print("--------------------")
            print("submitted")
            print(data)
            list_imgs=[]
            
            for img_src, img_desc in data.items():
                print(img_src)
                print(img_desc)
                for tile in img_desc["tiles"]:
                    bbox=tile["box"]
                    tmp_file=tile["file"]
                    print(tmp_file)
                    list_imgs.append(tmp_file)
            print(list_imgs)
            print(p_prompt)
            if self.ocr_mode=="QWEN3_8B_VL":
                json_result["ocr_params"]["model"]="QWEN_3_VL_8B"
                ocr_obj=OcrQwen3b8(p_prompt, p_len_token)
                ocr_result=ocr_obj.process(list_imgs)
                for img_src, img_desc in data.items():
                    tmp_json={}
                    tmp_json["main_image_file"]=img_src
                    tmp_json["tiles"]=[]
                    print(img_src)
                    print(img_desc)
                    for tile in img_desc["tiles"]:
                        file=tile["file"]
                        print(file)                        
                        if file in ocr_result:
                            print("found")                            
                            print(ocr_result[file])
                            json_tile={}
                            json_tile["tile_file"]=file
                            json_tile["box"]=tile["box"]
                            json_tile["result"]=ocr_result[file]
                            tmp_json["tiles"].append(json_tile)
                    json_result["images"].append(tmp_json)
            elif self.ocr_mode=="PYTESSERACT_WORD" or self.ocr_mode=="PYTESSERACT_LINES":
                #TODO
                print("TESSERACT_W")
                
                if self.ocr_mode=="PYTESSERACT_LINES":
                    mode_tesseract="LINE"                    
                else:
                    mode_tesseract="WORD"
                json_result["ocr_params"]["model"]=self.ocr_mode
                ocr_obj=OcrTesseract(mode_tesseract, self.autorotate, self.mode_grey_contrast,  self.autrotate_max, self.grey_threshold_min, self.grey_threshold_max, self.rey_blur_kernel)
                ocr_result=ocr_obj.process(list_imgs)
                print(ocr_result)
                for img_src, img_desc in data.items():
                    tmp_json={}
                    tmp_json["main_image_file"]=img_src
                    tmp_json["tiles"]=[]
                    print(img_src)
                    print(img_desc)
                    for tile in img_desc["tiles"]:
                        file=tile["file"]
                        print(file)                        
                        if file in ocr_result:
                            print("found")  
                            print("----------------------------")
                            vals=ocr_result[file]
                            #print(vals)
                            for val in vals:                                
                                text=val["text"]
                                conf=val["conf"]
                                if conf>0 and len(text.strip())>0:
                                    print(val)  
                                    #{'text': 'peut', 'level': 5, 'conf': 0.96, 'top': 3939, 'left': 494, 'height': 54, 'width': 130, 'box': {'x1': 494, 'x2': 624, 'y1': 3939, 'y2': 3993}, 'box_transformed': {'x1': 494, 'x2': 624, 'y1': 3939, 'y2': 3993}, 'par': 1, 'line': 20, 'line_absolute': 61, 'order': 1, 'file_name': '/opt/panafgeo_ict/src/panafgeo_franck_2026/out/panafgeo_mrac_004/imgs/chk_0000_mrac_05_0001/subtiles/000/437.png'}
                                    json_tile={}
                                    json_tile["tile_file"]=val["file_name"]
                                    json_tile["box"]=val["box"]
                                    json_tile["result"]=text
                                    json_tile["score"]=conf
                                    json_tile["line"]=val["line_absolute"]
                                    json_tile["order"]=val["order"]
                                    tmp_json["tiles"].append(json_tile)
                    json_result["images"].append(tmp_json) 
                    #sys.exit()

                                    
                            
            print("OCR_DONE")
           
            #print(ocr_result)            
            #json_tile["file"]=file
            
        print(json_result)
        #json_tile["text"]=
        json_result_file=os.path.join(self.new_dir_ocr, "ocr_result.json")
        print(json_result_file)
        with open(json_result_file, 'w') as f:
            json.dump(json_result, f, indent=4)          
            
    def process_img(self, p_segmentation_columns, p_segmentation_lines, p_remove_h_lines, p_remove_v_lines,p_remove_v_h_lines, p_remove_h_v_lines, ratio_pad_left=0.1, ratio_pad_right=0.5, force_min_x=None, force_max_x=None):
        print(p_segmentation_columns)
        print(p_segmentation_lines)
        print(p_remove_v_h_lines)
        new_dir=os.path.join(self.output_folder, self.project_name)
        os.makedirs(new_dir, exist_ok=True )
        self.new_dir_img=os.path.join(new_dir, "imgs")
        self.new_dir_ocr=os.path.join(new_dir, "ocr")
        new_dir_classification=os.path.join(new_dir, "classification")
        os.makedirs(self.new_dir_img, exist_ok=True )
        os.makedirs(self.new_dir_ocr, exist_ok=True )
        os.makedirs(new_dir_classification, exist_ok=True )
        imgs=[]
        for f in os.listdir(self.input_folder):
            tmp_file=os.path.join(self.input_folder, f)
            if os.path.isfile(tmp_file):
                print(tmp_file)
                imgs.append(tmp_file)
        imgs.sort()
        print(imgs)        
        main_i=0
        main_dict=OrderedDict()
        for f_img in imgs:
            dict_img={}
            img=cv2.imread(f_img)
            #segmentation_columns
            if p_remove_v_lines:
                #TODO 
                remove=RemoveLines(img)
                img=remove.process(p_is_vertical=True)
            elif p_remove_h_lines:
                remove=RemoveLines(img)
                img=remove.process(p_is_vertical=False)
            elif p_remove_v_h_lines:                
                remove=RemoveLines(img)
                tmp=remove.process(p_is_vertical=True)
                remove2=RemoveLines(tmp)
                img=remove2.process(p_is_vertical=False) 
            elif p_remove_h_v_lines:                
                remove=RemoveLines(img)
                tmp=remove.process(p_is_vertical=False)
                remove2=RemoveLines(tmp)
                img=remove2.process(p_is_vertical=True)    
            if p_segmentation_columns:            
                print("segmentation_columns")
                #img=cv2.imread(f_img)
                tmp_path=os.path.basename(f_img).split('.')
                tmp_path.pop()
                dir_filename=".".join(tmp_path)
                height, width, _ = img.shape
                print(f"{width=},{height=}")
                dict_img["full_image"]=OrderedDict()
                dict_img["tiles"]=[]
                dict_img["full_image"]["width"]=width
                dict_img["full_image"]["height"]=height
                threshold_width=math.ceil(width/100)
                #print(threshold_width)
                bboxes=self.split_columns(img, threshold_width, ratio_pad_left, ratio_pad_right, force_min_x, force_max_x)
                #print(bboxes)
                chunk_nr=str(main_i).zfill(4)
                tile_name="chk_"+chunk_nr
                new_dir_img_chunk=os.path.join(self.new_dir_img, tile_name+"_"+dir_filename)
                os.makedirs(new_dir_img_chunk, exist_ok=True )
                i=0
                max_i=len(bboxes)
                bboxes2=[]
                for i in range(0, max_i):
                    dict_tile=OrderedDict()                
                    dict_tile["box"]=OrderedDict()
                    box=bboxes[i]
                    #small=img[ box[1]:box[3], box[0]:box[2]]       
                    if (box[2]- box[0])>threshold_width:
                        if i==0:
                            box_pad=self.pad_box_x(box, 0, width, force_min_x=0)
                        elif i==max_i-1:
                            box_pad=self.pad_box_x(box, 0, width, force_max_x=width)
                        else:     
                            box_pad=self.pad_box_x(box, 0, width)
                        print(box_pad)
                        small_padded=img[ box_pad[1]:box_pad[3], box_pad[0]:box_pad[2]]
                        chunk_nr_2=str(i).zfill(3)
                        new_dir_img_col=os.path.join(new_dir_img_chunk, str(chunk_nr_2)+".png")
                        print(new_dir_img_col)
                        cv2.imwrite(new_dir_img_col, small_padded)
                        dict_tile["file"]=new_dir_img_col
                        dict_tile["box"]["x1"]=box_pad[0]
                        dict_tile["box"]["x2"]=box_pad[2]
                        dict_tile["box"]["y1"]=box_pad[1]
                        dict_tile["box"]["y2"]=box_pad[3]
                        dict_img["tiles"].append(dict_tile)
                #json_file=os.path.join(new_dir_img_chunk, "img_info.json")
                #print(json_file)
                #with open(json_file, 'w') as f:
                #    json.dump(dict_img, f)
                main_dict[f_img]=dict_img
            #SEGMENTATION_LINE 
            elif p_segmentation_lines: 
                pass
            #KEEP_ORIGINAL_IMAGE
            else:
                print("no segmentation")
                #img=cv2.imread(f_img)
                tmp_path=os.path.basename(f_img).split('.')
                tmp_path.pop()
                dir_filename=".".join(tmp_path)
                height, width, _ = img.shape
                print(f"{width=},{height=}")
                dict_img["full_image"]=OrderedDict()
                dict_img["tiles"]=[]
                dict_img["full_image"]["width"]=width
                dict_img["full_image"]["height"]=height
                box=[0,width, 0, height]
                chunk_nr=str(main_i).zfill(4)
                tile_name="chk_"+chunk_nr
                new_dir_img_chunk=os.path.join(self.new_dir_img, tile_name+"_"+dir_filename)
                os.makedirs(new_dir_img_chunk, exist_ok=True )
                i=0
                dict_tile=OrderedDict()                
                dict_tile["box"]=OrderedDict()                               
                chunk_nr_2=str(i).zfill(3)
                new_dir_img_col=os.path.join(new_dir_img_chunk, str(chunk_nr_2)+".png")
                print(new_dir_img_col)
                cv2.imwrite(new_dir_img_col, img)
                dict_tile["file"]=new_dir_img_col
                #check coordinate order different from segmented
                dict_tile["box"]["x1"]=box[0]
                dict_tile["box"]["x2"]=box[1]
                dict_tile["box"]["y1"]=box[2]
                dict_tile["box"]["y2"]=box[3]
                dict_img["tiles"].append(dict_tile)
                main_dict[f_img]=dict_img
            main_i=main_i+1    
        json_file=os.path.join(self.new_dir_img, "img_info.json")
        print(json_file)
        with open(json_file, 'w') as f:
            json.dump(main_dict, f, indent=4)        
        return json_file
    
if __name__ == "__main__": 
    try:
        parser = argparse.ArgumentParser()   
        parser.add_argument("--project_name",help= "project_name", required=True)  
        parser.add_argument("--input_folder",help= "input folder", required=True)   
        parser.add_argument("--output_folder",help= "input folder", required=True) 
        parser.add_argument("--ocr_mode",help= "OCR type", required=True) 
        parser.add_argument("--len_token",help= "Qwen tokens", default=4092) 
        parser.add_argument("--segmentation_columns",help= "auto segmentation columns flag", action='store_true')
        parser.add_argument("--segmentation_lines",help= "auto segmentation lines flag", action='store_true') 
        parser.add_argument("--remove_h_lines",help= "auto remove horizontal lines", action='store_true')
        parser.add_argument("--remove_v_lines",help= "auto remove vertical lines", action='store_true')
        parser.add_argument("--remove_v_h_lines",help= "auto remove vertical first and then horizontal lines" , action='store_true')
        parser.add_argument("--remove_h_v_lines",help= "auto remove horizontal first and then vertical lines" , action='store_true')
        parser.add_argument("--prompt",help= "prompt for OCR (currently QWEN7)", default="Transcribe the text of the document like you were an OCR engine. Keep spaces between words")
        parser.add_argument("--mode_grey_contrast",help= "Go to greyscale and enhance contrast" , action='store_true')
        parser.add_argument("--grey_threshold_min",help= "Min min value for noise reduction in greyscale (0-255), default is 220" , default="220")
        parser.add_argument("--grey_threshold_max",help= "Max min value for noise reduction in greyscale (0-255), default is 255" , default="255")
        parser.add_argument("--grey_blur_kernel",help= "Kernel size to blur in greyscale mode, default=5" , default="5")
        parser.add_argument("--autorotate",help= "auto remove horizontal first and then vertical lines" , action='store_true')
        parser.add_argument("--autorotate_max",help= "max angle in degrees for autorotate" , default="5")        
        args = parser.parse_args()
        if not os.path.exists(args.input_folder):
            raise CustomException("Error : Input folder not found")  
        if not os.path.exists(args.output_folder):
            raise CustomException("Error : output folder not found")
        if not args.ocr_mode in OCRParser.ocr_types:
            available_modes=','.join(OCRParser.ocr_types)
            raise CustomException("Error : OCR mode not supported. Possible values : "+available_modes)             
        grey_threshold_min=int(args.grey_threshold_min)
        grey_threshold_max=int(args.grey_threshold_max)
        grey_blur_kernel=int(args.grey_blur_kernel)
        autorotate_max=int(args.autorotate_max)
        len_token=int(args.len_token)
        ocr_parser= OCRParser(args.ocr_mode, args.autorotate, autorotate_max, args.mode_grey_contrast, grey_threshold_min, grey_threshold_max,grey_blur_kernel )
        ocr_parser.process(args.prompt, args.project_name, args.input_folder,args.output_folder, len_token, args.segmentation_columns, args.segmentation_lines, args.remove_h_lines, args.remove_v_lines, args.remove_v_h_lines, args.remove_h_v_lines )
    except Exception as e:
        print("EXCEPTION")
        print(e)
        print(traceback.print_exc())
        print(traceback.print_stack())
        print(traceback.format_exc())
        