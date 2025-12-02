import torch.multiprocessing as mp
import traceback
import cv2
import numpy as np
import fitz 
import datetime
import threading
import os
import sys
from segmentation_lines import SegmentationLines
from remove_lines import RemoveLines
from ocr_qwen_async import OcrQwenAsync
from detect_rotation import DetectRotation
from bisect import bisect_left
from pylatexenc.latex2text import LatexNodes2Text
import json
from pathlib import Path

SRC_PDF="PremierePartie1a21700_p1_10.pdf"
NO_SEGMENT=True
SEGMENT_ALL=True
OUTPUT_FOLDER_IMG="./output/images"
OUTPUT_FOLDER_TEXT="./output/text"
DELIMIT_TEXT=False
GO_GRAY=False
MAX_WIDTH=1250


def serialize_json(p_filename, p_data):
    with open(p_filename, 'w', encoding='utf-8') as f:
        json.dump(p_data,  f, ensure_ascii=False, indent=4)

def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        if len(myList)>1:
            return (None,myList[0], myList[1])
        else:
            return (None,myList[0], None)
    if pos == len(myList):
        if len(myList)>1:
            return (myList[-2],myList[-1],None)
    before = myList[pos - 1]
    after = myList[pos]
    
    if after - myNumber < myNumber - before:
        if (pos+1) <len(myList):
            last= myList[pos+1]
        else: 
            last=None
        returned= (before, after, last)
    else:
        if (pos-2) >=0:
            first=myList[pos-2]
        else:
            first=None
        returned= (first, before, after)
    return  returned   
    
def remove_elems(p_bboxes):
    returned={}
    for k, bbox in p_bboxes.items():
        if not bbox["to_remove"]:
            returned[k]=bbox
    return returned
  
def simplify_bboxes(p_bboxes):
    d_box={}
    i=0
    print(len(p_bboxes))
    for box in p_bboxes:
        d_box[i]={"box": box, "checked":False, "to_remove":False, "merge_with":[]}
        i=i+1
    tmp={}
    for i, bbox_obj in d_box.items():
        print(i)
        print(bbox_obj)
        y_min=bbox_obj["box"][1]
        y_max=bbox_obj["box"][3]
        print(y_min)
        print(y_max)
        if y_min not in tmp:
            tmp[y_min]={}
        if not y_max in tmp[y_min]:
            tmp[y_min][y_max]=[]
        tmp[y_min][y_max].append(i)
    print("tmp")
    print("===============>")
    print(tmp)
    for x_min, obj_max in tmp.items():
        if len(obj_max.keys())>1:
            #print(obj_max)
            od = dict(sorted(obj_max.items(), reverse=True))         
            print("duplicate")
            #print(x_min)
            #print(od)
            #print(list(od.keys()))
            v_keys=list(od.keys())
            first=v_keys.pop()
            #print(first)
            if len(od[first])>1:
                #print("several higher")
                for i in range(1, len(od[first])):
                    tmp_key=od[first][i]
                    d_box[tmp_key]["to_remove"]=True
            for y_min2, i_obj in od.items():
                print(y_min2)
                print(i_obj)
                for i in i_obj:
                   d_box[i]["to_remove"]=True 
            """print(obj_max)
            x_max=max(obj_max.keys())
            print(x_max)
            """
    d_box=remove_elems(d_box)
    print(d_box)
    print(len(d_box))
    return d_box
    

def segment_per_page(p_bboxes, w, p_segment=10, pad_up=False, pad_down=False, pad_h=0):
    returned=[]
    list_mins=[]
    list_maxs=[]
    for bbox in p_bboxes:
        list_mins.append(bbox[1])
        list_maxs.append(bbox[3])
    list_mins=sorted(list_mins)
    list_maxs=sorted(list_maxs)
    #print(list_mins)
    #print(list_maxs)
    min_h=list_mins[0]
    max_h=list_maxs[-1]
    #print(min_h)
    #print(max_h)
    tmp_h=max_h-min_h
    #print(tmp_h)
    size_segment=int(np.round(tmp_h/p_segment))    
    #print(size_segment)
    cumul_max_h=min_h
    bbox_h1=None
    bbox_h2=None
    for i in range(0,p_segment ):
        print("---------------")
        print(i)
        cumul_min_h=cumul_max_h
        cumul_max_h=cumul_max_h+size_segment
        #print(cumul_min_h)
        if i==p_segment-1:
            cumul_max_h=max_h
        #print(cumul_max_h)
        existing_min_h=take_closest(list_mins, cumul_min_h)
        print(existing_min_h)
        ref_min_h=existing_min_h[1]
        if ref_min_h>cumul_min_h and existing_min_h[0] is not None:
            ref_min_h=existing_min_h[0]
        #print(ref_min_h)
        existing_max_h=take_closest(list_maxs, cumul_max_h)
        print(existing_max_h)
        ref_max_h=existing_max_h[1]
        if ref_max_h<cumul_max_h and existing_max_h[1] is not None:
            ref_max_h=existing_max_h[2]
        #print(ref_max_h)
        if i==0:
            bbox_h1=ref_min_h
        else: 
            if bbox_h1 !=bbox_h2:
                bbox_h1=bbox_h2
            else:
                break
        #if i<p_segment-1:
        bbox_h2=ref_max_h
        if i==0 and pad_up is True and bbox_h1>0:
            returned.append([w[0], 0, w[1], bbox_h1 ])
        print(bbox_h1)
        print(bbox_h2)
        returned.append([w[0], bbox_h1 , w[1], bbox_h2])
        if i==p_segment-1 and pad_down is True and bbox_h2<pad_h:
            returned.append([w[0], bbox_h2,  w[1], bbox_h2 ])
    return returned

def process_page(p_page_image, p_page_index, p_file_name_img, lock ):
    global OUTPUT_FOLDER_IMG, OUTPUT_FOLDER_TEXT, MAX_WIDTH, DELIMIT_TEXT, NO_SEGMENT, SEGMENT_ALL,GO_GRAY
    try:
        lock.acquire()
        output_folder_img_pdf=OUTPUT_FOLDER_IMG+"/"+p_file_name_img
        output_folder_text_pdf=OUTPUT_FOLDER_TEXT+"/"+p_file_name_img
        Path(output_folder_text_pdf).mkdir(parents=True, exist_ok=True) 
        Path(output_folder_img_pdf).mkdir(parents=True, exist_ok=True) 
        
        file_name_img=p_file_name_img+"_"+(str(p_page_index).rjust(4, '0'))
        image=p_page_image
        h, w, _=image.shape
        h_original=h
        w_original=w
        r=1
        if MAX_WIDTH is not None:
            if MAX_WIDTH>0 and w>MAX_WIDTH:
                r=MAX_WIDTH/w
                image = cv2.resize(image, None, fx=r, fy=r, interpolation=cv2.INTER_LINEAR)
                h, w, _=image.shape
                    
        #image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        remover=RemoveLines(image)
        flag_remove, image=remover.process(p_threshold_line=0.10)
        results={}
        results["document"]=p_file_name_img
        results["results"]={}
        
        if (flag_remove and not SEGMENT_ALL) or NO_SEGMENT:
            results_tmp={}
            results_tmp["mode"]="full_image"
            results_tmp["segment"]=None
           
            try:
                print("no segmentation")
                seg=SegmentationLines(image)
                bboxes=seg.process(offset_w_ratio=0, offset_h_ratio=0)
                if DELIMIT_TEXT:
                    min_h=seg.get_min_h(bboxes)
                    max_h=seg.get_max_h(bboxes)
                    cropped=image[min_h:max_h, 0:w]           
                    results_tmp["bbox"]=[min_h* (1/r), max_h* (1/r), 0, w* (1/r)]
                else:  
                    cropped=image 
                    results_tmp["bbox"]=[0, h, 0, w]
                file_name_page=os.path.join(output_folder_img_pdf, file_name_img+".jpg")
                print(file_name_page)
                cv2.imwrite(file_name_page,cropped )
                ocr=OcrQwenAsync( p_max_tokens=8092)
                if GO_GRAY:
                    cropped=cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                    cropped=cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
                text_latex=ocr.process(cropped, 120)
                print(text_latex)
                text_raw=LatexNodes2Text().latex_to_text(text_latex)
                results_tmp["text_latex"]=text_latex
                results_tmp["text_raw"]=text_raw
                results_tmp["status"]="ok"
                results["results"][0]=results_tmp
            except TimeoutError as e:
                print("OCR Timeout:", e)
                results_tmp["text_latex"]=None
                results_tmp["text_raw"]=None
                results_tmp["status"]="timeout"
                results["results"][0]=results_tmp
                #sys.exit()
            except Exception:
                print(traceback.format_exc())
                results_tmp["text_latex"]=None
                results_tmp["text_raw"]=None
                results_tmp["status"]="exception"
                results["results"][0]=results_tmp
                sys.exit()
            output_json=os.path.join(output_folder_text_pdf, file_name_img+".json")
            print(output_json)
            serialize_json(output_json, results)
        else:
            print("segmentation")
            lock2 = threading.Lock()
            try:                
                image_source=image.copy()
                if GO_GRAY:
                    image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    image=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                bboxes=[]
                try:
                    lock2.acquire()
                    seg=SegmentationLines(image)
                    bboxes=seg.process(offset_w_ratio=0, offset_h_ratio=0)
                except Exception:
                    print("EXCEPTION IN SEGMENTATION PREPARATION")
                    print(traceback.format_exc())
                finally:
                    lock2.release()
                h, w, _=image.shape  
                print(bboxes)
                
                bboxes=segment_per_page(bboxes, (0, w), p_segment=10, pad_up=False)
                print(bboxes)
                i_seg=0
                
                for box in bboxes:
                    try:
                        results_tmp={}
                        lock2.acquire()
                        #print(box)
                        x1_1, y1_1, x2_1, y2_1=box
                        if y2_1>y1_1:
                            print((x1_1, y1_1, x2_1, y2_1))
                            
                            cropped=image[y1_1:y2_1, x1_1:x2_1]
                            cropped_original=image[y1_1:y2_1, x1_1:x2_1]
                            
                            
                            rotation=DetectRotation(cropped)
                            angle_lines=rotation.detect_line_rotation()
                            print(angle_lines)
                            #sys.exit()
                            if angle_lines!=0:
                                print("rotate")
                                origin_h=y1_1+y2_1/2
                                origin_w=x1_1+x2_1/2
                                rotated_full=rotation.perform_line_rotation(image, angle_lines,p_origin= (origin_w,origin_h))
                                rotated_full_original=rotation.perform_line_rotation(image_source, angle_lines,p_origin= (origin_w,origin_h))
                                rheight, rwidth = rotated_full_original.shape[:2]
                                pad_down=(y2_1-y1_1)/10
                                y2_1_b=min(int(round(y2_1+pad_down)),rheight)
                                cropped=rotated_full[y1_1:y2_1_b, 0:rwidth]
                                cropped_original=rotated_full_original[y1_1:y2_1_b, 0:rwidth]
                                seg2=SegmentationLines(cropped_original)
                                bboxes2=seg2.process(offset_w_ratio=0, offset_h_ratio=0)
                                if len(bboxes2)>0:
                                    min_h2=seg.get_min_h(bboxes2)
                                    max_h2=seg.get_max_h(bboxes2)
                                    cropped_original=cropped_original[min_h2:max_h2, 0:rwidth]                          
                            
                            
                            results_tmp["mode"]="segmented"
                            results_tmp["segment"]=i_seg
                            results_tmp["image_size_original"]=[0, h_original, 0, w_original]
                            results_tmp["image_size_acquisition"]=[0, h, 0, w]
                            results_tmp["bbox"]=[y1_1, y2_1, x1_1, x2_1]
                            file_name_page=os.path.join(output_folder_img_pdf, file_name_img+"_"+str(i_seg)+".jpg")
                            print(file_name_page)
                            cv2.imwrite(file_name_page,cropped_original )
                            ocr=OcrQwenAsync( p_max_tokens=2048)
                            text_latex=ocr.process(cropped_original, 120)
                            print(text_latex)
                            text_raw=LatexNodes2Text().latex_to_text(text_latex)
                            
                            
                            results_tmp["text_latex"]=text_latex
                            results_tmp["text_raw"]=text_raw
                            results_tmp["status"]="ok"
                            results["results"][i_seg]=results_tmp
                            
                        else:
                            results_tmp["mode"]="segmented"
                            results_tmp["segment"]=i_seg
                            results_tmp["image_size_original"]=[0, h_original, 0, w_original]
                            results_tmp["image_size_acquisition"]=[0, h, 0, w]
                            results_tmp["bbox"]=[y1_1, y2_1, x1_1, x2_1]
                            results_tmp["text_latex"]=None
                            results_tmp["text_raw"]=None
                            results_tmp["status"]="wrong_bbox"
                            results["results"][i_seg]=results_tmp
                        i_seg=i_seg+1
                    except TimeoutError as e:
                        print("OCR Timeout:", e)
                        results_tmp["image_size_original"]=[0, h_original, 0, w_original]
                        results_tmp["image_size_acquisition"]=[0, h, 0, w]
                        results_tmp["text_latex"]=None
                        results_tmp["text_raw"]=None
                        results_tmp["status"]="timeout"
                        results["results"][i_seg]=results_tmp
                        #sys.exit()
                    except Exception:
                        print(traceback.format_exc())
                        results_tmp["image_size_original"]=[0, h_original, 0, w_original]
                        results_tmp["image_size_acquisition"]=[0, h, 0, w]
                        results_tmp["text_latex"]=None
                        results_tmp["text_raw"]=None
                        results_tmp["status"]="exception"
                        results["results"][i_seg]=results_tmp
                        #sys.exit()
                    finally:
                        lock2.release()
                output_json=os.path.join(output_folder_text_pdf, file_name_img+".json")
                print(output_json)
                serialize_json(output_json, results)
            except Exception:
                print("EXCEPTION AT LOCK IN SEGMENTATION LEVEL")
                print(traceback.format_exc())

    except Exception:
        print("EXCEPTION AT LOCK LEVEL")
        print(traceback.format_exc())
    finally:
        lock.release()
        
def pix_to_image(pix):
    bytes = np.frombuffer(pix.samples, dtype=np.uint8)
    img = bytes.reshape(pix.height, pix.width, pix.n)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    doc = fitz.Document(SRC_PDF)
    file_name_pdf=Path(os.path.basename(SRC_PDF)).stem
    lock = threading.Lock()
    #lock = threading.Lock()
    for i in range(0,len(doc)):
        for img in doc.get_page_images(i):            
            try:
                print(f'page {i}')
                xref = img[0]
                #image = doc.extract_image(xref)
                pix = fitz.Pixmap(doc, xref)
                #print(pix.__class__)
                #print(image.__class__)
                #print(image.keys())
                image=pix_to_image(pix)
                #cv2.imshow("", image)
                #cv2.waitKey()
                process_page(image, i, file_name_pdf, lock)          
            except Exception:
                print("EXCEPTION_AT_PDF_LEVEL")
                print(traceback.format_exc())
                