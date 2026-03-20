"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2
import traceback

class RemoveLines():

    
    def __init__(self, p_file_cv2):
        self.file_cv2 = p_file_cv2    
        
    def process(self, p_is_vertical=True, p_kernel_w=2, p_kernel_h=2,  p_size_blur_kernel=50, p_threshold_line=0.10):
        self.is_vertical=p_is_vertical
        self.kernel_w=p_kernel_w
        self.kernel_h=p_kernel_h
        self.size_blur_kernel=p_size_blur_kernel
        self.threshold_line= p_threshold_line
        return self.blur_vline()
        
    def get_dominant_color(self, p_img, p_ksize):
        src=cv2.resize(p_img, (p_ksize,p_ksize), interpolation=cv2.INTER_LINEAR)
        blur = cv2.blur(p_img,(p_ksize,p_ksize))
        k = blur[int(p_ksize/2),int(p_ksize/2)]
        return k
    
    def fill_edges(self, src):
        im_floodfill = src.copy()
        h, w = im_floodfill.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = src | im_floodfill_inv
        return im_out
        
    def blur_vline(self): 
        try:
            # Check if image is loaded fine
            src=self.file_cv2.copy()
           
            if src is None:
                print ('Error opening image: ' + argv[0])
                return -1 
            if len(src.shape) != 2:
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            else:
                gray = src
            # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
            gray = cv2.bitwise_not(gray)
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, 15, -2)
            # Create the images that will use to extract the horizontal and vertical lines
            ref_direction = np.copy(bw)
            
            if self.is_vertical:
                cols = ref_direction.shape[1]
                dir_size = cols // 30
                structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dir_size)) 
            else:
                cols = ref_direction.shape[0]
                dir_size = cols // 30
                structure = cv2.getStructuringElement(cv2.MORPH_RECT, (dir_size, 1))
         
            

         
            # Create structure element for extracting vertical lines through morphology operations
            
            # Apply morphology operations
            ref_direction = cv2.erode(ref_direction, structure)
            ref_direction = cv2.dilate(ref_direction, structure) 

            # [smooth]
            # Inverse vertical image
            ref_direction = cv2.bitwise_not(ref_direction)
         

         
            # Step 1
            edges = cv2.adaptiveThreshold(ref_direction, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, 3, -2)
         
            # Step 2
            kernel = np.ones((self.kernel_w, self.kernel_h), np.uint8)
            edges = cv2.dilate(edges, kernel)
            
            edges=self.fill_edges(edges)
            #cv2.imshow("", edges)
            #cv2.waitKey()
            # Step 3
            smooth = np.copy(ref_direction)
         
            # Step 4
            smooth = cv2.blur(smooth, (2, 2))
         
            # Step 5 mask
            (rows, cols) = np.where(edges != 0)
            sum_line=np.sum(edges != 0)
            
            proportion_line=sum_line/(src.shape[0]*src.shape[1])
            if proportion_line>self.threshold_line:
                return src
            
            #step 6 resize and blur to get averaged RGB color
            (r_mean, g_mean,b_mean) =self.get_dominant_color(src, self.size_blur_kernel)
            #step 7 use the grey mask on the color image with the averaged color
            src[rows, cols] = (r_mean, g_mean,b_mean)
            height, width, _ = src.shape
            (bw, bh)=(int(width/2),int(height/2))

            #step 8 blur again to smooth
            blur2=cv2.blur(src,(bw, bh))
            #reapply the blurred mask on the original
            src[rows, cols]=blur2[rows, cols]
            #cv2.imshow("", src)
            #cv2.waitKey()
            #if len(src.shape) != 2:
            #    gray2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            #else:
            #    gray2 = src
            print("returned")
            print(src.__class__)
            return src #, gray2
        except Exception:
            print(traceback.format_exc()) 
            return self.file_cv2



  
