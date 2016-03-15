'''
This program is for Live running Overhead video. The required transformation matrix is already stored.
Program will provide Live overhead video output.

'''

import cv2
import sys
import numpy as np
from PIL import Image

def endProg():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

def d(string):      # debugging
    print string
    
def print_coordinate(event,x,y,flags,param):
    font = cv2.FONT_HERSHEY_SIMPLEX
    points = np.float32()
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.putText(dst,str(x) + ' ' + str(y),(x-20,y-20),font,0.3,(0,0,255),1,cv2.LINE_AA)
		
def store_point_coordinate(event,x,y,flags,param):
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

def resize(img, factor):
    rows,cols,ch = img.shape
    trans = np.float32([[factor*1,0,0],[0,factor*1,0]])
    foreground = cv2.warpAffine(img,trans,(int(factor*cols),int(factor*rows)))
    return foreground

def resizeOnCenter(img, factor):
    rows,cols,ch = img.shape
    tx = (factor-1)*rows*0.5
    ty = (factor-1)*cols*0.5
    trans = np.float32([[1,0,ty],[0,1,tx]])
    foreground = cv2.warpAffine(img,trans,(factor*cols,factor*rows))
    return foreground

def modifyValue(vec,img,factor,flag):
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

def cropImage(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop

def main():
    first_image = './Amol/Produce_12.avi'
    second_image = './Amol/Produce_2.mp4'

    M = np.load('./output/calibData/perspective_Second_First.npy')
    h = np.load('./output/calibData/perspective_overhead.npy')


    cap1 = cv2.VideoCapture(0)               # camera objects
    cap2 = cv2.VideoCapture(2)

    #cap1.open(first_image)                  
    #cap2.open(second_image)

    print cap1.get(3),cap1.get(4)
    print cap2.get(3),cap2.get(4)

    try:
        while(cap1.isOpened() and cap2.isOpened()):
            ret, img = cap1.read()
            ret, img2 = cap2.read()
            
            img = resizeOnCenter(img,4)                 # resize image on center to 4 times
            rows,cols,ch = img.shape

            dst = cv2.warpPerspective(img2,M,(cols,rows))

            background = dst
            foreground = img

            alpha = 0.5

            r = rows
            c = cols

            size = r,c, 3
            dist = np.zeros(size, dtype=np.uint8)

            mm = np.float32([[1,0,0],[0,1,0]])
            foreground = cv2.warpAffine(foreground,mm,(c,r))

            beta = 1.0 - alpha
               
            cv2.addWeighted(background,alpha,foreground, beta, 0.0, dist)

            #out = np.zeros((2000,2000,3),np.uint8)

            out = cv2.warpPerspective(dist,h,(2*cols,2*rows))

            #out = resize(out,0.85)
            out = cropImage(out)
            cv2.imshow("FinalImage",out)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except:
        cap1.release()
        cap2.release()
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()


