from ultralytics import YOLO
import cv2
import numpy as np
import traceback

class SegmentationLines():

    PATH_MODEL_BOX="./riksarkivet_yolov9_region_1/model.pt"
    PATH_MODEL_LINES="./riksarkivet_yolov9_lines_within_region_1/model.pt"
    
    def __init__(self, p_cv2_img):
        self.img=p_cv2_img
        
        


    def process(self, offset_w_ratio=0, offset_h_ratio=0, pad_begin=False, pad_end=False):
        lines_coords=[]
        try:
            cv_image_copy=self.img.copy()
            bboxes, _=self.execute_yolo_model(SegmentationLines.PATH_MODEL_BOX, self.img)
            agg_bboxes=[] 
            for b in bboxes:
                x1, y1, x2, y2 =map(int, b)
                nested_bboxes=self.go_for_nested(SegmentationLines.PATH_MODEL_LINES, self.img, (x1, y1, x2, y2) )
                agg_bboxes.append(nested_bboxes)
            h, w, _=self.img.shape  
            if len(agg_bboxes)>0:
                pos_max_list=np.argmax([len(i) for i in agg_bboxes])
                main_structuring_element=agg_bboxes[pos_max_list]
                
                lines_coords=self.go_for_lines(main_structuring_element, 0, w, 0, h, offset_w_ratio, offset_h_ratio)
                lines_coords=sorted(lines_coords, key=lambda x: x[1] )
                if pad_begin:
                    min_h=self.get_min_h(lines_coords)
                    if min_h >0:
                        first_row=[0, 0, w, int(min_h+min(np.round(min_h*offset_h_ratio),h))]
                        lines_coords.insert(0, first_row)
                    
                if pad_end:
                    max_h=self.get_max_h(lines_coords)
                    if max_h <h:
                        last_row=[0, int(max(max_h-np.round(max_h*offset_h_ratio),0)), w, h]
                        lines_coords.append(last_row)
            else:
                lines_coords.append([0, 0, w, h])
        except Exception:
            print(traceback.format_exc())
        return lines_coords   
        
    def execute_yolo_model(self,p_path, p_img, offset_x=0, offset_y=0):
        bboxes=[]
        masks=[]
        model=YOLO(p_path)
        results= model.predict(p_img, device='cuda', save=False, show=False, verbose=False)    
        for r in results:        
            for box in r.boxes:            
                b = box.xyxy[0]
                x1_1, y1_1, x2_1, y2_1 =map(int, b)            
                bboxes.append([offset_x+x1_1, offset_y+y1_1, offset_x+x2_1, offset_y+y2_1])     
            masks.append(r.masks)
        return bboxes, masks
        
    def go_for_lines(self,p_bboxes, min_w, max_w, min_h, max_h, offset_w_ratio=0, offset_h_ratio=0):
        #print("CALL-LINES---------------")
        returned=[]
        size_w=max_w-min_w
        margin_w=int(np.round(abs(offset_w_ratio)*size_w))
        #print(f"{margin_w=}")
        if offset_w_ratio!=0:
            if offset_w_ratio<0:
                min_w= min_w+margin_w
                max_w=max(min_w, max_w-margin_w)
        for bbox in p_bboxes:        
            pmin_h= bbox[1]
            pmax_h= bbox[3]
            size_h=pmax_h-pmin_h
            #print("original_coords")
            #print([min_w, pmin_h, max_w, pmax_h])
            #print(f"{size_w=}")
            #print(f"{size_h=}")            
            if offset_h_ratio!=0:
                margin_h=int(np.round(abs(offset_h_ratio)*size_h))
                #print(f"{margin_h=}")
                if offset_h_ratio>0:
                    pmin_h=max(0, pmin_h-margin_h)
                    pmax_h= pmax_h+margin_h
                elif offset_h_ratio<0:
                    pmin_h=min(pmax_h, pmin_h+margin_h)
                    pmax_h=max(pmin_h, pmax_h-margin_h)
            #print("resized_coords")
            #print([min_w, pmin_h, max_w, pmax_h])
            returned.append([min_w, pmin_h, max_w, pmax_h])
        return returned
        
    def go_for_nested(self, p_path_model, p_img, p_bbox):
        #print(p_bbox)
        x1, y1, x2, y2 =p_bbox
        offset_x=x1
        offset_y=y1
        #print(x1)
        cropped=p_img[y1:y2, x1:x2]
        #print(cropped.shape)
        nestedbboxes,_=self.execute_yolo_model(p_path_model, cropped,offset_x, offset_y)
        return nestedbboxes

    def get_min_h(self,p_bboxes, default=0):
        returned=default
        acc=[]
        for bbox in p_bboxes:
            acc.append(bbox[1])
        return min(acc)
        #if len(p_bboxes)>0:
        #    returned=p_bboxes[0][1]
        #return returned
       
    def get_max_h(self,p_bboxes, default=0):
        returned=default
        """
        if len(p_bboxes)>0:
            returned=p_bboxes[len(p_bboxes)-1][3]
        return returned   
        """
        acc=[]
        for bbox in p_bboxes:
            acc.append(bbox[3])
        return max(acc)