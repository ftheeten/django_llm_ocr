import cv2
import pytesseract
import sys
import os
import math
from collections import OrderedDict
from pathlib import Path

from .images_pre_ops import ImagesPreOps


# python main.py --project_name=panafgeo_mrac_004 --input_folder=/opt/panafgeo_ict/src/panafgeo_franck_2026/src/panafgeo_mrac_0005/ --output_folder=/opt/panafgeo_ict/src/panafgeo_franck_2026/out   --ocr_mode=PYTESSERACT_WORD --autorotate --mode_grey_contrast

class OcrTesseract(ImagesPreOps):  
    
    modes=["WORD", "LINE"]
    
    def __init__(self, p_mode="LINE", p_autorotate=False, p_mode_grey_contrast=False, p_autorotate_max=5, p_grey_threshold_min=220, p_grey_threshold_max=255, p_grey_blur_kernel=5):
        self.mode=p_mode
        super().__init__( p_autorotate, p_mode_grey_contrast, p_autorotate_max, p_grey_threshold_min, p_grey_threshold_max, p_grey_blur_kernel)
        
        
    def process(self, p_images):
        returned=OrderedDict()
        
        i_page=0
        list_imgs=[]
        for img_path in p_images:
            print(img_path)
            base_name = os.path.basename(img_path)
            print(base_name)
            parent_folder=Path(img_path).parent.absolute()
            print(parent_folder)
            self.new_dir_tiles=os.path.join(parent_folder, "subtiles")
            tmp_name_ori_image=base_name.split(".")
            tmp_name_ori_image.pop()
            subtile_chunk_folder= os.path.join(self.new_dir_tiles, ".".join(tmp_name_ori_image)) 
            if not  os.path.exists(subtile_chunk_folder):
                os.makedirs(subtile_chunk_folder, exist_ok=True )   
            self.new_dir_transformed=os.path.join(parent_folder, "transformed")
            transformed_img_file=os.path.join(self.new_dir_transformed, ".".join(tmp_name_ori_image)+".jpg" )
            json_file=os.path.join(self.new_dir_transformed, "transform_info.json" )
            
              

            
            img=cv2.imread(img_path)
            h, w, _=img.shape  
            
            
            img, h2, w2, ratio_transform=self.transform_image(img, img_path, transformed_img_file, json_file)
            list_imgs.append(img)
            data = pytesseract.image_to_data(img, output_type='dict', config='-c preserve_interword_spaces=1,-psm 6')
            page=[]
            print(data)
            print(data["level"])
            print(data["width"])
            print(data["height"])
            print(data["par_num"])
            print(data["line_num"])
            print(data["word_num"])
            print(data["conf"])
            print(data["text"])
            current_par=0
            current_base_line=0
            current_line_total=0
            i_tile=0
            for i in range(0, len(data["text"])):
                text=data["text"][i]
                if len(text.strip())>0:
                    page_num=data["page_num"][i]
                    block=data["block_num"][i]
                    par=data["par_num"][i]
                    conf=data["conf"][i]
                    height=data["height"][i]
                    width=data["width"][i]
                    top=data["top"][i]
                    left=data["left"][i]
                    
                    if par>current_par:
                        current_par=par
                    line=data["line_num"][i]
                    if line != current_base_line:
                        current_base_line=line
                        current_line_total=current_line_total+1
                    line_absolute= current_line_total   
                    order=data["word_num"][i]
                    level=data["level"][i]
                    box={"x1": math.floor(left*ratio_transform), "x2": math.floor((left+width)*ratio_transform), "y1": math.floor(top*ratio_transform), "y2": math.floor((top+height)*ratio_transform)}
                    box_transformed={"x1": left, "x2": left+width, "y1": top, "y2": top+height}
                    tmp={
                        "text":text,
                        "page":page_num,
                        "block":block,
                        "line":line,
                        "line_absolute":line_absolute,
                        "level":level,
                        "par":par,
                        "conf":float(data["conf"][i])/100,
                        "top":int(top),
                        "left":int(left),
                        "height":int(height),
                        "width":int(width),
                        "box":box,
                        "box_transformed":box_transformed,                  
                        "order":order,
                    }        
                    if self.mode=="WORD":
                        cropped=img[box["y1"]:box["y2"],box["x1"]:box["x2"] ]
                        tile_file_name=str(i_tile).zfill(3)+".png"
                        tile_file_name=os.path.join(subtile_chunk_folder, tile_file_name)
                        print("write")
                        print(tile_file_name)
                        cv2.imwrite(tile_file_name, cropped)
                        tmp["file_name"]=tile_file_name
                    page.append(tmp)
                    i_tile=i_tile+1
            returned[img_path]=page
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(current_par)
            print(current_line_total)
            i_page=i_page+1
        if self.mode=="LINE":
            returned_tmp=OrderedDict()
            i=0
            for p_path, p_page in returned.items():
                print(p_path)
                #print(p_page)  
                print(subtile_chunk_folder)
                
                print("\n°°°°°°°°°°°°°°°°°°°°°°°°°°°°\n")
                page=self.group_by_lines(p_page, p_path, list_imgs[i], ratio_transform)
                returned_tmp[p_path]=page
                i=i+1
            returned=returned_tmp
        return returned
        
    def enlarge_box(self, p_box, p_test):
        if p_test["x1"]<p_box["x1"]:
            p_box["x1"]=p_test["x1"]
        if p_test["y1"]<p_box["y1"]:
            p_box["y1"]=p_test["y1"]
        if p_test["x2"]>p_box["x2"]:
            p_box["x2"]=p_test["x2"]
        if p_test["y2"]>p_box["y2"]:
            p_box["y2"]=p_test["y2"]
        return p_box
        
    def group_by_lines(self, p_list, p_img_path, p_img, p_ratio_transform):
        print(p_img)       
        page=[]        
        groups=OrderedDict()            
        for val in p_list:
            print(val)
            block=val["block"]
            par=val["par"]
            line=val["line"]
            order=val["order"]
            box=val["box"]
            box_transformed=val["box_transformed"]
            text=val["text"]
            
            if len(text.strip())>0:
                #print(val)
                if not block in groups:
                    groups[block]=OrderedDict()                    
                if not par in groups[block]:
                    groups[block][par]=OrderedDict()                  
                if not line in groups[block][par]:                
                    groups[block][par][line]=OrderedDict()
                groups[block][par][line][order]=val    
                
        
        #print(groups)
        line_absolute_nr=0
        par_nr=0
        i_tile=0
        for block, pars in groups.items():
            for par, lines in pars.items():
                tmp_line_list=[]                
                for line, words in lines.items():
                    line_dict={}
                    tmp_line_list=[]
                    line_box=None
                    line_box_transformed=None
                    conf_line=0
                    nb_chars=0
                    iw=0
                    for ikey, word in  words.items():                       
                        if iw==0:
                            line_box=word["box"]
                            line_box_transformed=word["box_transformed"]
                        else:
                            tmp_box=word["box"]
                            tmp_box_transformed=word["box_transformed"]
                            line_box=self.enlarge_box( line_box, tmp_box)
                            line_box_transformed=self.enlarge_box( line_box_transformed, tmp_box_transformed)
                        tmp_line_list.append(word["text"])
                        len_char=len(word["text"])
                        nb_chars=nb_chars+len_char
                        conf=word["conf"]
                        conf_weighted=len_char*conf
                        conf_line=conf_line+conf_weighted
                        iw=iw+1 
                    tmp_line_str=" ".join(tmp_line_list)
                    print(tmp_line_str)
                    print(line_box)
                    print(line_box_transformed)
                    line_dict={}
                    line_dict["text"]=tmp_line_str
                    average_conf=float(conf_line/nb_chars)
                    line_dict["conf"]=average_conf
                    line_dict["block"]=block
                    line_dict["par"]=par
                    line_dict["line"]=line
                    line_dict["line_absolute"]=line_absolute_nr
                    line_dict["order"]=0                    
                    bbox_to_write={}
                    bbox_to_write['x1']=math.floor(line_box_transformed['x1']/p_ratio_transform)
                    bbox_to_write['x2']=math.floor(line_box_transformed['x2']/p_ratio_transform)
                    bbox_to_write['y1']=math.floor(line_box_transformed['y1']/p_ratio_transform)
                    bbox_to_write['y2']=math.floor(line_box_transformed['y2']/p_ratio_transform)
                    line_dict["box"]=bbox_to_write                
                    line_absolute_nr=line_absolute_nr+1
                    
                    cropped=p_img[bbox_to_write["y1"]:bbox_to_write["y2"],bbox_to_write["x1"]:bbox_to_write["x2"] ]
                    tile_file_name=str(i_tile).zfill(3)+".png"
                    
                    print(p_img_path)
                    base_name = os.path.basename(p_img_path)
                    print(base_name)
                    parent_folder=Path(p_img_path).parent.absolute()
                    print(parent_folder)
                    self.new_dir_tiles=os.path.join(parent_folder, "subtiles")
                    tmp_name_ori_image=base_name.split(".")
                    tmp_name_ori_image.pop()
                    subtile_chunk_folder= os.path.join(self.new_dir_tiles, ".".join(tmp_name_ori_image)) 
                    if not  os.path.exists(subtile_chunk_folder):
                        os.makedirs(subtile_chunk_folder, exist_ok=True )            
                    tile_file_name=os.path.join(subtile_chunk_folder, tile_file_name)
                    print("write")
                    print(tile_file_name)
                    
                    cv2.imwrite(tile_file_name, cropped)                
                    line_dict["file_name"]=tile_file_name
                    i_tile=i_tile+1
                    page.append(line_dict)    
        par_nr=par_nr+1
        return page