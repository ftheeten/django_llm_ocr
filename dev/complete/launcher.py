from timeout import TimeoutException
from segmentation import Segmentation
from ocr import Ocr
from parser import Parser
import os
import datetime
import signal
from PIL import Image
import math
from pylatexenc.latex2text import LatexNodes2Text
import traceback

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


class Launcher():    
    
    def __init__(self, p_img, p_segment=False, p_classify_keywords=False,  p_timeout=120):
        try:
            self.timeout=p_timeout
            self.segment=p_segment
            self.classify_keywords=p_classify_keywords
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            self.print_time()
            #self.img=p_img
            self.img=Image.open(p_img).convert("RGB")
            signal.alarm(0) 
            return
        except TimeoutException:
            print("Inference timeout")
            self.print_time()   
        except Exception as e:
            print("exception1")
            print(traceback.format_exc())
            signal.alarm(0) 
        #finally:
        #    signal.alarm(0) 
            
    def process(self):
        try:
            print("launcher process")
            self.print_time()
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            ocr_results=[]
            if self.segment:
                segmentation=Segmentation(self.img)
                d_bboxes=segmentation.process()
                self.print_time()            
                print(d_bboxes)
                imgs=[]
                bboxes=[]
                #image = Image.open(self.img).convert("RGB")
                for d_bbox in d_bboxes:
                    bbox=d_bbox["bbox"]
                    pclass=d_bbox["class"]
                    if pclass.upper() not in ["LABEL_0","LABEL_3" ]:
                        tmp_img=self.img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        imgs.append(tmp_img)
                        bboxes.append(bbox)
                if len(imgs)>0:
                    ocr=Ocr(p_timeout=math.floor(self.timeout/2))                      
                    parser=Parser()                   
                    i=0
                    for img in imgs:
                        self.print_time()                    
                        text=ocr.process(img)
                        classification=None
                        if self.classify_keywords:
                            classification=self.f_classify_keywords(parser, text)
                        print(text)
                        ocr_results.append({"text": text, "bbox":bboxes[i], "classification":classification})     
                        i=i+1
            else:
                ocr=Ocr(p_timeout=math.floor(self.timeout/2)) 
                text=ocr.process(self.img)
                classification=None
                if self.classify_keywords:
                    parser=Parser() 
                    classification=self.f_classify_keywords(parser, text)
                    size=self.img.size
                ocr_results.append({"text": text, "bbox":[0, 0, size[0], size[1]], "classification":classification})
            self.print_time()
            print(ocr_results)
            print(len(ocr_results))
            signal.alarm(0) 
            return ocr_results
        except TimeoutException:
            print("Inference timeout")
            self.print_time()   
        except Exception as e:
            print("exception2")
            signal.alarm(0) 
            print(traceback.format_exc())
        #finally:
        #    signal.alarm(0) 
    
    def f_classify_keywords(self, p_parser, p_text):
        text=LatexNodes2Text().latex_to_text(p_text)
        print(text)
        classification=p_parser.process(text)
        print(classification)
        return classification
    
    def handler(self, signum, frame):
        raise TimeoutException()
        
    def print_time(self):
        now = datetime.datetime.now()
        print ("Current date and time : ")
        print (now.strftime("%Y-%m-%d %H:%M:%S"))