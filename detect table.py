import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np

def preprocess(img,f: int):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    en = ImageEnhance.Sharpness(img).enhance(f)
    if img_gray.std() <30:
        en = ImageEnhance.Contrast(en).enhance(f)
    return np.array(en)

def group_h_line(h_lines,thin_thresh):
    new_h_lines = []
    
    while len(h_lines) >0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines =[]
        for line in h_lines:
            if thresh[1]-thin_thresh <= line[0][1] <= thresh[1]+thin_thresh:
                lines.append(line)
        lines_ex =[]
        for line in h_lines:
            if thresh[1]-thin_thresh > line[0][1] or line[0][1] > thresh[1]+thin_thresh:
                lines_ex.append(line)
        h_lines=lines_ex
        x =[]
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x)- 4*thin_thresh, max(x) + 4*thin_thresh
        new_h_lines.append([x_min,thresh[1],x_max,thresh[1]])

    return new_h_lines

def group_v_line(v_lines,thin_thresh):
    new_v_lines = []
    
    while len(v_lines) >0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines =[]
        for line in v_lines:
            if thresh[0]-thin_thresh <= line[0][0] <= thresh[0]+thin_thresh:
                lines.append(line)
        lines_ex =[]
        for line in v_lines:
            if thresh[0]-thin_thresh > line[0][0] or line[0][0] > thresh[0]+thin_thresh:
                lines_ex.append(line)
        v_lines=lines_ex
        y =[]
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y)- 4*thin_thresh, max(y)+4*thin_thresh
        new_v_lines.append([thresh[0],y_min,thresh[0],y_max])

    return new_v_lines

        
def seg_intersect(line1,line2):
    point=[]
    if line2[0][1]<=line1[0][1]:
        if line1[0][0]<=line2[0][0]<=line1[1][0]:
            return line2[0][0],line1[0][1]
    return  None, None
def get_bottom_right(right_point, bottom_point,points):
    for right in right_point:
        for bottom in bottom_point:
            if [right[0],bottom[1]] in points:  
                return right[0],bottom[1]
    return None,None

table_image = cv2.imread("table_img.png")
#"C:\Users\Hisokaxxzk\OneDrive\Máy tính\All\table_img.png"
table_image = preprocess(table_image,5)

gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
thresh, image_binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
image_binary = 255-image_binary

kernel_len = gray.shape[1]//120
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_len,1))
# print(horizontal_kernel)
img_hor = cv2.erode(image_binary, horizontal_kernel, iterations =3)
hor_line = cv2.dilate(img_hor,horizontal_kernel,iterations =3)
# print(hor_line)

h_lines = cv2.HoughLinesP(hor_line,1,np.pi/180,30)
# print(h_lines[0][0][1])
new_horizontal_lines = group_h_line(h_lines,kernel_len)
# print(new_horizontal_lines)

kernel_len = gray.shape[1]//120
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,kernel_len))
# print(horizontal_kernel)
img_vertical = cv2.erode(image_binary, vertical_kernel, iterations =3)
vertical_line = cv2.dilate(img_vertical,vertical_kernel,iterations =3)
# print(hor_line)

v_lines = cv2.HoughLinesP(vertical_line,1,np.pi/180,30)
# print(h_lines[0][0][1])
# print(new_horizontal_lines)
new_vertical_lines = group_v_line(v_lines,kernel_len)
# print(new_vertical_lines)

points =[]
for h_line in new_horizontal_lines:
    x1A,y1A,x2A,y2A = h_line
    for v_line in new_vertical_lines:
        x1B,y1B,x2B,y2B = v_line
        line1 = [np.array([x1A,y1A]),np.array([x2A,y2A])]
        line2 = [np.array([x1B,y1B]),np.array([x2B,y2B])]

        x,y = seg_intersect(line1,line2)
        if(x!=None):
            if x1A<=x<=x2A and y1B <=y <=y2B:
                points.append([int(x),int(y)])

# print(points)
cells=[]
for point in points:
    left,top = point
    right_points = sorted([p for p in points if p[0]>left and p[1]==top],key=lambda x: x[0])
    bottom_points = sorted([p for p in points if p[1]>top and p[0]==left],key=lambda x: x[1])
    right,bottom = get_bottom_right(right_points,bottom_points,points)
    if right and bottom:
        cv2.rectangle(table_image, (left,top), (right,bottom),(0,0,255),2)
        cells.append([left,top,right,bottom])

plt.imshow(table_image)
plt.show()