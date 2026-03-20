import cv2
import numpy as np
import math


class SegmentationColumns():

    def __init__(self, p_cv2_img):
        self.img=p_cv2_img
        self.acc_lines={}
        
        
    def process(self, p_ratio_merge_w=0.1, p_ratio_merge_h=0.2, p_min_line_length=100, p_margin_deg_cols=30, p_margin_deg_lines=20, p_hough_theshold=100,p_max_line_gap=80 ,  p_canny_threshold_1=50, p_canny_threshold_2=150, p_canny_aperture_size=3):
        self.ratio_merge_w=p_ratio_merge_w
        self.ratio_merge_h=p_ratio_merge_h
        self.margin_deg_cols=p_margin_deg_cols
        self.margin_deg_lines=p_margin_deg_lines
        self.hough_theshold=p_hough_theshold
        self.max_line_gap=p_max_line_gap
        self.min_line_length=p_min_line_length
        self.canny_threshold_1=p_canny_threshold_1
        self.canny_threshold_2=p_canny_threshold_2
        self.canny_aperture_size=p_canny_aperture_size
        bboxes=self.go()
        return bboxes
        
    def go(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        h,w = self.gray.shape
        self.edges = cv2.Canny(self.gray,self.canny_threshold_1,self.canny_threshold_2,apertureSize = self.canny_aperture_size)
        acc_columns=self.line_detection( "x", p_margin_deg=self.margin_deg_cols)
        acc_lines =self.line_detection("y", p_margin_deg=self.margin_deg_lines)
        acc_column_merge2=self.detect_merge_main_lines(acc_columns, w, self.ratio_merge_w, "x", "alternate", p_mode_begin="min", p_mode_end="max")
        acc_lines_merge2=self.detect_merge_main_lines(acc_lines, h, self.ratio_merge_h, "y", "min", p_mode_begin="min", p_mode_end="average")
        bboxes=self.prepare_bboxes("full_columns_recognized_lines", h, w, acc_column_merge2, acc_lines_merge2, take_borders_x=False,take_borders_y=True )
        return bboxes
        
        
    def line_detection(self, p_axis, p_margin_deg ):
        #global gray
        lines = cv2.HoughLinesP(image=self.edges,rho=1,theta=np.pi/180, threshold=self.hough_theshold,lines=np.array([]), minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        a,b,c = lines.shape
        acc={}  
        for i in range(a):
            pt1=(int(lines[i][0][0]), int(lines[i][0][1]))
            pt2=(int(lines[i][0][2]), int(lines[i][0][3]))            
            if p_axis=="x":
                deg_angle=self.calculate_angle_deg(pt1, pt2)
                if  deg_angle>(270-p_margin_deg) and deg_angle<(270+p_margin_deg):
                    tmp=pt1
                    pt1=pt2
                    pt2=tmp
                    deg_angle=self.calculate_angle_deg(pt1, pt2)
                if (deg_angle>(90-p_margin_deg) and deg_angle<(90+p_margin_deg)) : # or deg_angle>(270-margin_deg) and deg_angle<(270+margin_deg) :
                    dist = cv2.norm(pt1 , pt2, cv2.NORM_L2)
                    min_x=min(pt1[0], pt2[0])                    
                    if not min_x in acc:
                        acc[min_x]={}
                    if not dist in acc[min_x]:
                        acc[min_x][dist]=[]
                    acc[min_x][dist].append((pt1, pt2))
            elif p_axis=="y":
                deg_angle=self.calculate_angle_deg(pt1, pt2)
                if (deg_angle>(180-p_margin_deg) and deg_angle<(180+p_margin_deg)) or (deg_angle>(-180-p_margin_deg) and deg_angle<(-180+p_margin_deg)) :                      
                    tmp=pt1
                    pt1=pt2
                    pt2=tmp
                    deg_angle=self.calculate_angle_deg(pt1, pt2)                     
                if (deg_angle >=0 and (deg_angle <  p_margin_deg) or deg_angle > (360 -p_margin_deg )) or (deg_angle <0 and (deg_angle > -p_margin_deg)):                    
                    dist = cv2.norm(pt1 , pt2, cv2.NORM_L2)
                    min_y=min(pt1[1], pt2[1])
                    if not min_y in acc:
                        acc[min_y]={}
                    if not dist in acc[min_y]:
                        acc[min_y][dist]=[]
                    acc[min_y][dist].append((pt1, pt2))
        acc =  dict(sorted(acc.items(), key=lambda item: item[0]))     
        return acc
        
    def calculate_angle_deg(self, pt1, pt2):
        float_angle=math.atan2(int(pt1[1])-int(pt2[1]), int(pt1[0])-int(pt2[0]))
        deg_angle=float_angle*180/math.pi 
        return deg_angle
        
    def prepare_bboxes(self, p_mode,  height, width, p_columns, p_lines,  p_pad_x=0, p_pad_y=0, take_borders_x=False, take_borders_y=False):
        returned=[]
        
        min_h=0
        max_h=height
        min_w=0
        max_w=width
        if not take_borders_x:
            if len(p_columns)>0:
                first=p_columns[0]
                pt1=first[0]
                pt2=first[1]
                min_w=min(pt1[0], pt2[0])            
            if len(p_columns)>1:
                last=p_columns[-1]
                pt1=last[0]
                pt2=last[1]
                max_w=max(pt1[0], pt2[0])
                
        if not take_borders_y:
            if len(p_lines)>0:
                first=p_lines[0]
                pt1=first[0]
                pt2=first[1]
                min_h=min(pt1[1], pt2[1])           
            if len(p_lines)>1:
                last=p_lines[-1]
                pt1=last[0]
                pt2=last[1]
                max_h=max(pt1[1], pt2[1])   
        if p_mode=="full_lines_recognized_columns":
            #todo
            pass
        elif p_mode=="full_columns_recognized_lines":            
            previous_w=None
            i=0
            for col in p_columns:
                pt1=col[0]
                pt2=col[1]
                current_w=max(pt1[0],pt2[0])
                if i>0:
                    #returned.append([min_h,max_h, previous_w, current_w ])  
                    returned.append([previous_w, min_h, current_w, max_h])                      
                previous_w=current_w
                i=i+1
        elif p_mode=="cell":
            pass
        return returned
        
    def detect_merge_main_lines(self, acc, w_h_len, ratio_merge, axis="x", mode="alternate", p_mode_begin="min", p_mode_end="max"):
        returned=[]
        current_base=0
        current_base_max=0
        acc_merged={}
        step_merge=w_h_len*ratio_merge
        for min_x_or_y, dist_points in acc.items():
            #print(min_x_or_y)
            #print(dist_points)
            if current_base==0 or min_x_or_y > current_base_max:
                current_base=min_x_or_y
                current_base_max=current_base+step_merge
                acc_merged[current_base]={}
            if min_x_or_y <= current_base_max:
                acc_merged[current_base][min_x_or_y]=dist_points   
            
        alternate_border=False
        last=None
        init_pt_1=None
        init_pt_2=None
        last_pt_1=None
        last_pt_2=None
        i=0
        for base_cluster, clustered in  acc_merged.items():
            #print(base_cluster)
            #print(clustered)
            min_h_or_w=w_h_len
            max_h_or_w=0
            current_x_or_y=base_cluster
            for x_or_y, segments in clustered.items():
                #print(x_or_y)
                #print(segments)
                current_x_or_y=x_or_y
                
                for height, line in segments.items():
                    #print(height)
                    line=line[0]
                    #print(line)
                    pt1=line[0]
                    pt2=line[1]
                    if axis=="x":
                        h1=line[0][1]
                        h2=line[1][1]
                    elif axis=="y":
                        h1=line[0][0]
                        h2=line[1][0]
                    loc_min_h_or_w=min(h1,h2)
                    loc_max_h_or_w=min(h1,h2)
                    if loc_min_h_or_w<min_h_or_w:
                        min_h_or_w=loc_min_h_or_w
                    if loc_max_h_or_w>max_h_or_w:
                        max_h_or_w=loc_max_h_or_w    
            
            if axis=="x":            
                min_pt1=(base_cluster,max_h_or_w )
                min_pt2=(base_cluster,min_h_or_w )
            
                max_pt1=(current_x_or_y,max_h_or_w )
                max_pt2=(current_x_or_y,min_h_or_w )
            
                avg_pt1=((current_x_or_y+base_cluster)/2,max_h_or_w )
                avg_pt2=((current_x_or_y+base_cluster)/2, min_h_or_w)
            elif axis=="y":            
                min_pt1=(max_h_or_w,base_cluster )
                min_pt2=(min_h_or_w,base_cluster )
            
                max_pt1=(max_h_or_w,current_x_or_y )
                max_pt2=(min_h_or_w,current_x_or_y )
            
                avg_pt1=(max_h_or_w,(current_x_or_y+base_cluster) )
                avg_pt2=(min_h_or_w,(current_x_or_y+base_cluster)/2)
            
            if i==0:
                if p_mode_begin=="min":
                    init_pt_1= min_pt1
                    init_pt_2= min_pt2
                elif p_mode_begin=="max":
                    init_pt_1=max_pt1
                    init_pt_2= max_pt2
                elif p_mode_begin=="average":
                    init_pt_1=avg_pt1
                    init_pt_2= avg_pt2
            else:
                if p_mode_begin=="min":
                    last_pt_1= min_pt1
                    last_pt_2= min_pt2
                elif p_mode_begin=="max":
                    last_pt_1=max_pt1
                    last_pt_2= max_pt2
                elif p_mode_begin=="average":
                    last_pt_1=avg_pt1
                    last_pt_2= avg_pt2
            if mode=="alternate":
                if not alternate_border:
                    if axis=="x":
                        agg_pt1=(base_cluster,max_h_or_w )
                        agg_pt2=(base_cluster,min_h_or_w )
                    elif axis=="y":
                        agg_pt1=(max_h_or_w,base_cluster )
                        agg_pt2=(min_h_or_w,base_cluster )
                else:
                    if axis=="x":
                        agg_pt1=(current_x_or_y,max_h_or_w )
                        agg_pt2=(current_x_or_y,min_h_or_w )
                    elif axis=="y":
                        agg_pt1=(max_h_or_w,current_x_or_y )
                        agg_pt2=(min_h_or_w,current_x_or_y )
                alternate_border=not alternate_border
            elif mode=="min" or mode=="min_last_max":
                if axis=="x":
                    agg_pt1=(base_cluster,max_h_or_w )
                    agg_pt2=(base_cluster,min_h_or_w )
                    #if mode=="min_last_max":
                    #    last=((current_x_or_y,max_h_or_w ),(current_x_or_y,min_h_or_w ))
                elif axis=="y": 
                    agg_pt1=(max_h_or_w,base_cluster )
                    agg_pt2=(min_h_or_w,base_cluster )
                    #if mode=="min_last_max":
                    #    last=((max_h_or_w,current_x_or_y ),(min_h_or_w,current_x_or_y ))
            elif mode=="max":
                if axis=="x":
                    agg_pt1=(current_x_or_y,max_h_or_w )
                    agg_pt2=(current_x_or_y,min_h_or_w )
                elif axis=="y": 
                    agg_pt1=(max_h_or_w,current_x_or_y )
                    agg_pt2=(min_h_or_w,current_x_or_y )
            
                
            returned.append((agg_pt1, agg_pt2))
            i=i+1    
        if len(returned)>0:
            returned[0]=(init_pt_1, init_pt_2)
        if len(returned)>1:
            returned[-1]=(last_pt_1, last_pt_2)    
        return returned