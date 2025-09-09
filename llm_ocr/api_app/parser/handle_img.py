from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import base64
#from django.core.files.base import ContentFile
import numpy as np
import cv2

class HandleImg():
    def __init__(self, p_img_b64):
        self.img_b64=p_img_b64
        
    
    def parse(self):
        print(self.img_b64)
        if "," in self.img_b64:
            header, self.img_b64 = self.img_b64.split(",", 1)
        image_data = base64.b64decode(self.img_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
