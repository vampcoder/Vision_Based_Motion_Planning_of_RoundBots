'''
In this Program, we are going to obtain transformation from pre-saved matrixes.
There are two camera input used in this code.
To calculate the transformation we need the points which is mapped to the points in the overhead image
Minimum four points are required to calculate the transformation matrix (Homography).
'''

import cv2
import sys
import numpy as np
from PIL import Image

def inputImage():           # for taking input
    
    first_image = 'input/Camera1/Lpicture070.jpg'
    second_image = 'input/Camera2/Rpicture069.jpg'
    '''
    first_image = 'C:\Users\hp pc\Documents\pythonFiles\open\Amol\img1.jpg'
    second_image = 'C:\Users\hp pc\Documents\pythonFiles\open\Amol\img2.jpg'
    '''
    return first_image, second_image

def endProg():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

def d(string):  # debugging
    print string
    
def print_coordinate(event,x,y,flags,param):
    font = cv2.FONT_HERSHEY_SIMPLEX
    points = np.float32()
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.putText(dst,str(x) + ' ' + str(y),(x-20,y-20),font,0.3,(0,0,255),1,cv2.LINE_AA)
		
def store_point_coordinate(event,x,y,flags,param):  # stores values in point
    font = cv2.FONT_HERSHEY_SIMPLEX
    if event == cv2.EVENT_FLAG_LBUTTON and len(pts1) < 4:
        cv2.circle(img,(x,y), 2, (255,255,255), -1)
        cv2.putText(img,str(x) + ' ' + str(y),(x-20,y-20),font,0.3,(255,255,255),1,cv2.LINE_AA)
        pts1.append([x,y])
		
def store_point_coordinate2(event,x,y,flags,param):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if event == cv2.EVENT_FLAG_LBUTTON and len(pts2) < 4:
        cv2.circle(img2,(x,y), 2, (255,255,255), -1)
        cv2.putText(img2,str(x) + ' ' + str(y),(x-20,y-20),font,0.3,(255,255,255),1,cv2.LINE_AA)
        pts2.append([x,y])

def resize(img, factor):                # resizing the image
    rows,cols,ch = img.shape
    trans = np.float32([[factor*1,0,0],[0,factor*1,0]])
    foreground = cv2.warpAffine(img,trans,(factor*cols,factor*rows))
    return foreground

def resizeOnCenter(img, factor):        # resizing image keeping it in center
    rows,cols,ch = img.shape
    tx = (factor-1)*rows*0.5
    ty = (factor-1)*cols*0.5
    trans = np.float32([[1,0,ty],[0,1,tx]])
    foreground = cv2.warpAffine(img,trans,(factor*cols,factor*rows))
    return foreground

def modifyValue(vec,img,factor,flag):   # modify coordinate wrt resized image
    rows,cols,ch = img.shape
    if flag == 1:
        for x in range(0, 4):
            vec[x][0] *= factor
            vec[x][1] *= factor
        return vec
    else:
        for x in range(0, 4):
            vec[x][0] += (factor-1)*cols*0.5
            vec[x][1] += (factor-1)*rows*0.5
        return vec

def plotPointOnImage(img,vec,size): 
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, size):
        cx = int(vec[x][0])
        cy = int(vec[x][1])
        cv2.circle(img,(cx,cy), 2, (255,255,255), -1)
        cv2.putText(img,str(cx) + ' ' + str(cy),(cx-20,cy-20),font,0.3,(255,255,255),1,cv2.LINE_AA)
    return img

def cropImage(img):                  # cropping image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop

def main():
    output_img = "./output/output33.jpg"            # storing output images

    first_image, second_image = inputImage()    # get input

    img = cv2.imread(first_image)
    print img.shape
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    img2 = cv2.imread(second_image)
    print img2.shape
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    rows,cols,ch = img.shape
    print img.shape
    print img2.shape
    
    img = resizeOnCenter(img,4)                 # resize image to 4 times
    rows,cols,ch = img.shape
    
    M = np.load('./output/calibData/perspective_Second_First.npy')       # using already saved transformation matrix

    dst = cv2.warpPerspective(img2,M,(cols,rows))

    background = dst
    foreground = img

    alpha = 0.5

    r = rows
    c = cols

    size = r,c, 3
    dist = np.zeros(size, dtype=np.uint8)

    # d('yy')
    mm = np.float32([[1,0,0],[0,1,0]])
    foreground = cv2.warpAffine(foreground,mm,(c,r))                # affine transformation matrix for scaling, translating, etc

    beta = 1.0 - alpha                                              # giving weights to both images
       
    cv2.addWeighted(background,alpha,foreground, beta, 0.0, dist)  # Stitching image 


    #cv2.imshow("Final",dist)               # in case of debugging with output
    cv2.imwrite("./output/FinalImage4.jpg",dist)

    h = np.load('./output/calibData/perspective_overhead.npy')   # using final overhead transformation matrix

    out = np.zeros((2000,2000,3),np.uint8)
    
    out = cv2.warpPerspective(dist,h,(2*cols,2*rows))

    cv2.namedWindow("FinalImage", cv2.WINDOW_AUTOSIZE) 

    #out = resize(out,0.5)              
    out = cropImage(out)                # cropping and resizing image 
    cv2.imshow("FinalImage",out)
    cv2.imwrite(output_img,out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
