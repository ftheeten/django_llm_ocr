import cv2
import numpy as np
import math

class DetectRotation():
    
    def __init__(self, p_small_image):
        self.small_image=p_small_image
        #self.bbox=p_bbox
        #self.full_image=p_full_image

    def detect_line_rotation(self):
        # Source - https://stackoverflow.com/a/59364345
        # Posted by Max Kaha
        # Retrieved 2025-11-27, License - CC BY-SA 4.0
        imgray = cv2.cvtColor(self.small_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append((area, cnt))
        crop = thresh
        edges = cv2.Canny(crop, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        sum=0
        cpt=0
        if lines is not None:
            for i in range(0, len(lines)):
                #rho = lines[i][0][0]
                theta = lines[i][0][1] 
                #theta_deg=math.degrees(theta)
                if theta!=0:
                    sum=sum+theta
                    cpt=cpt+1
        if cpt!=0:
            average= sum/cpt
            average_deg=math.degrees(average)-90
            return average_deg
        else:
            return 0

    def perform_line_rotation(self, p_image,p_angle_degree, p_origin=(0,0)):
        rotation_matrix=cv2.getRotationMatrix2D(p_origin, p_angle_degree, 1.0)
        height, width = p_image.shape[:2]
        rotated=cv2.warpAffine(p_image, rotation_matrix, (width, height))
        return rotated
        
        
            