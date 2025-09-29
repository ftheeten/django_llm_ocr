from transformers import AutoImageProcessor, BeitForSemanticSegmentation
#from PIL import Image
import torch
import numpy as np
import json
import signal
import traceback

class Segmentation():
    processor=None
    model=None
    model_name = "nevernever69/dit-doclaynet-segmentation"
    
    def __init__(self, p_img, p_timeout=120):
        print("segmentation_init")
        self.timeout=p_timeout
        self.img=p_img
        try:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            if Segmentation.processor is None and  Segmentation.model is None:
                print("init static")
                Segmentation.processor= AutoImageProcessor.from_pretrained(Segmentation.model_name)
                Segmentation.model=BeitForSemanticSegmentation.from_pretrained(Segmentation.model_name)
                Segmentation.model = Segmentation.model.to("cuda") 
                print("init static done")    
            #self.img = Image.open(p_img).convert("RGB")
            self.img=p_img
            print("return init Seg")
        except TimeoutException:
            print("Inference segm timeout")  
            signal.alarm(0)             
        except Exception as e:
            print("exception sem 3")
            print(traceback.format_exc())
            signal.alarm(0)
            
    def process(self):
        try:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout)
            print("segmentation_process")
            results=[]
            inputs = Segmentation.processor(images=self.img, return_tensors="pt").to("cuda")
            Segmentation.model.eval()
            with torch.no_grad():
                outputs = Segmentation.model(**inputs)
                logits = outputs.logits
                upsampled = torch.nn.functional.interpolate(logits, size=self.img.size[::-1], mode="bilinear", align_corners=False)
                mask = upsampled.argmax(dim=1).squeeze().cpu().numpy()
                id2label = Segmentation.model.config.id2label
                results = []
                for class_id, class_name in id2label.items():
                    binary_mask = (mask == class_id)
                    if binary_mask.any():
                        ys, xs = np.where(binary_mask)
                        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                        score = float(np.mean(upsampled[0, class_id].cpu().numpy()[ys, xs]))
                        results.append({
                            "class": class_name,
                            "bbox": bbox,
                            "score": round(score, 3)
                        })
            signal.alarm(0)     
            return results
        except TimeoutException:
            print("Inference timeout")   
            signal.alarm(0)             
        except Exception as e:
            print("exception4")
            print(traceback.format_exc())
            signal.alarm(0)             

    def handler(self, signum, frame):
        raise TimeoutException()
                