import cv2
import os
from pathlib import Path
import json
from .detect_rotation import DetectRotation

class ImagesPreOps():
    
    def __init__(self, p_autorotate=False, p_mode_grey_contrast=False, p_autorotate_max=5, p_grey_threshold_min=220, p_grey_threshold_max=255, p_grey_blur_kernel=5):
        self.autorotate=p_autorotate
        self.mode_grey_contrast=p_mode_grey_contrast
        self.autorotate_max=p_autorotate_max
        self.grey_threshold_min=p_grey_threshold_min
        self.grey_threshold_max=p_grey_threshold_max
        self.grey_blur_kernel=p_grey_blur_kernel
      
    def create_parent_folder(self, p_file):
        parent_folder=Path(p_file).parent.absolute()
        if not  os.path.exists(parent_folder):
            os.makedirs(parent_folder, exist_ok=True )     
      
    def morph_shape(self, val):
        if val == 0:
            return cv.MORPH_RECT
        elif val == 1:
            return cv.MORPH_CROSS
        elif val == 2:
            return cv.MORPH_ELLIPSE
     
   
    
    def transform_image(self, p_img, p_img_path,  p_transform_file_img, p_transform_file_json):
        #img=cv2.imread(p_img)
        img=p_img.copy()
        h, w, _=img.shape  
        h2=h
        w2=w
        self.has_transform={}
        self.has_transform["original_file"]={"file":p_img_path, "size": {"width": w, "height": h}}
        self.has_transform["transforms"]=[]
        if self.autorotate:
            rotation=DetectRotation(img)
            angle_lines=rotation.detect_line_rotation()  
            tmp_transform={"type": "autorotate", "measured_angle":angle_lines , "correction_threshold_max":self.autorotate_max, "done":False }
            if angle_lines!=0 and abs(angle_lines) <=self.autorotate_max:
                print("ROTATE")
                print(angle_lines)                    
                origin_h=(0+h)/2
                origin_w=(0+w)/2
                img=rotation.perform_line_rotation(img, angle_lines,p_origin= (origin_w,origin_h))
                h2, w2, _=img.shape
                tmp_transform["done"]=True
            self.has_transform["transforms"].append(tmp_transform)
        if self.mode_grey_contrast:                
            ksize = (self.grey_blur_kernel, self.grey_blur_kernel)
            tmp_img = cv2.blur(img, ksize) 
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
            img = cv2.inRange(tmp_img, self.grey_threshold_min, self.grey_threshold_max)
            h2, w2=img.shape
            tmp_transform={"type": "blur_and_enhance_grey_contrast", "ksize":self.grey_blur_kernel , "bw_threshold_min":self.grey_threshold_min, "bw_threshold_max":self.grey_threshold_max, "done":True }
            self.has_transform["transforms"].append(tmp_transform)
        self.has_transform["transformed_file"]={"file":p_transform_file_img, "size": {"width": w2, "height": h2}}
        self.create_parent_folder(p_transform_file_img)
        cv2.imwrite(p_transform_file_img, img)
        print(p_transform_file_img)
        self.create_parent_folder(p_transform_file_json)
        with open(p_transform_file_json, 'w') as f:
            json.dump(self.has_transform, f, indent=4)            
        ratio=float(((h2/h)+(w2/w))/2)
        return img, h2, w2, ratio