'''
In this Program, we are going to obtain the transformation matrix for the transformation from camera's input
to the Overhead view.
There are two camera input used in this code.
To calculate the transformation we need the points which is mapped to the points in the overhead image
Minimum four points are required to calculate the transformation matrix (Homography).
We manually matches the four points to calculate the transformation matrix using mouse to give input.
'''

import cv2
import sys
import numpy as np
from PIL import Image

def inputImage():           # for taking input
    first_image = 'input/camera1/picture063.jpg'
    second_image = 'input/camera2/picture062.jpg'
    '''
    first_image = 'C:\Users\hp pc\Documents\pythonFiles\open\picture048.jpg'
    second_image = 'C:\Users\hp pc\Documents\pythonFiles\open\picture049.jpg'
    
    first_image = 'C:\Users\hp pc\Documents\pythonFiles\open\Amol\img1.jpg'
    second_image = 'C:\Users\hp pc\Documents\pythonFiles\open\Amol\img2.jpg'
    '''
    return first_image, second_image

def endProg():              # for testing
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

def d(string):              # debugging
    print string

def store_point_coordinate(event,x,y,flags,param):  # stores values in point
    font = cv2.FONT_HERSHEY_SIMPLEX
    if event == cv2.EVENT_FLAG_LBUTTON and len(pts1) < 4:
        cv2.circle(img,(x,y), 2, (255,255,255), -1)
        cv2.putText(img,str(x) + ' ' + str(y),(x-20,y-20),font,0.3,(255,255,255),1,cv2.LINE_AA)
        pts1.append([x,y])
        print x, y
		
def store_point_coordinate2(event,x,y,flags,param):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if event == cv2.EVENT_FLAG_LBUTTON and len(pts2) < 4:
        cv2.circle(img2,(x,y), 2, (255,255,255), -1)
        cv2.putText(img2,str(x) + ' ' + str(y),(x-20,y-20),font,0.3,(255,255,255),1,cv2.LINE_AA)
        pts2.append([x,y])
        print x, y

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

def plotPointOnImage(img,vec,size):     # print set of points on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, size):
        cx = int(vec[x][0])
        cy = int(vec[x][1])
        cv2.circle(img,(cx,cy), 2, (255,255,255), -1)
        cv2.putText(img,str(cx) + ' ' + str(cy),(cx-20,cy-20),font,0.3,(0,0,255),1,cv2.LINE_AA)
    return img

def cropImage(img):                  # cropping image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    print 'strinf ' + str(x),y
    crop = img[y:y+h,x:x+w]
    return crop

def moveImage(image1,x,y):
    row,col,ch = image1.shape
    mat = np.float32([[0,0,y],[0,0,x]])
    image1 = cv2.warpAffine(image1,mat,(col,row))
    return image1

def resizeAuto(img, transMatrix):   #resizing function
    r,c,ch = img.shape
    #point = [[0,0,1],[0,c,1],[r,c,1],[r,0,1]]
    point = [[0,0,r,r],[0,c,c,0],[1,1,1,1]]
    point = np.matrix(point)
    mat = np.matrix(transMatrix)
    point = mat * point
    
    xmax = point[0,:].max()
    xmin = point[0,:].min()
    ymax = point[1,:].max()
    ymin = point[1,:].min()
    print xmin, ymin
    print xmin, ymax
    print xmax, ymax
    print xmax, ymin


def main(img,img2,pts1,pts2):
    first_image, second_image = inputImage()    # get input

    img = cv2.imread(first_image)       # reading image
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    rows,cols,ch = img.shape
    print img.shape

    cv2.namedWindow('img1')
    cv2.setMouseCallback('img1',store_point_coordinate)         

    while(True):                        # storing 4 points into pts1, which is used for mapping
        cv2.imshow('img1',img)
        if(cv2.waitKey(30) == 27 or len(pts1) >= 4):
            break
        
    pts1 = np.float32(pts1)             # convert pts1 to float32


    img2 = cv2.imread(second_image)     # doing same thing for other image from other camera
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    cv2.namedWindow('img2')
    cv2.setMouseCallback('img2',store_point_coordinate2)

    while(True):
        cv2.imshow('img2',img2)
        if(cv2.waitKey(30) == 27 or len(pts2) >= 4):
            break

    pts2 = np.float32(pts2)

    pts1 = modifyValue(pts1,img,4,2)    # modifing points according to 4 times zoomed image
    img = resizeOnCenter(img,4)         # resize image to 4 times

    rows,cols,ch = img.shape
    
    M = cv2.getPerspectiveTransform(pts2,pts1)  # getting transformation matrix to get perspection from 2nd image to 1st image

    np.save('./output/calibData/perspective_Second_First',M)     # Saving matrix

    dst = cv2.warpPerspective(img2,M,(cols,rows))   # converting 2nd image to 1st image

    background = dst
    foreground = img

    alpha = 0.5

    r = rows
    c = cols

    size = r,c, 3
    dist = np.zeros(size, dtype=np.uint8)

    mm = np.float32([[1,0,0],[0,1,0]])              # affine transformation matrix for scaling, translating, etc
    foreground = cv2.warpAffine(foreground,mm,(c,r))

    while(True) :
        beta = 1.0 - alpha
        
        cv2.addWeighted(background,alpha,foreground, beta, 0.0, dist)   # weighted addition of two images
        cv2.imshow("Final",dist)
        
        if ord('l') == cv2.waitKey(0) and alpha < 1.0:
            alpha += 0.05
        elif ord('k') == cv2.waitKey(0) and alpha > 0.0:
            alpha -= 0.05

        if cv2.waitKey(0) == 27 :
            break


    dds = cropImage(dist)           # cropping image
#   cv2.imshow("Final",dds)
    cv2.imwrite("./output/FinalImage.jpg",dds)


    c_height = rows
    c_width = cols
    sx = 100                        # scaling factor used for mapping real dimension to image
    sy = 100

    c_height -= (sx*0.5)
    c_width -= (sy*0.5)

    UpValue = np.float32([[c_width,c_height],[sy+c_width,c_height], [sy+c_width,sx+c_height], [c_width,sx+c_height] ])  # mapping points to square coordinates
    h, mask = cv2.findHomography(pts1,UpValue,cv2.RANSAC, 1.0)      # converting the stitched image

    np.save('./output/calibData/perspective_overhead',h)     # saving the image overhead converion matrix

    out = np.zeros((2000,2000,3),np.uint8)

    out = cv2.warpPerspective(dist,h,(3*cols,3*rows))   # creating final overhead image
    out = cropImage(out)

    cv2.imshow("FinalImage",out)
    #cv2.imwrite("./output/FinalOverHeadImage.jpg",out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

pts1 = []
pts2 = []
img = np.zeros((2000,2000,3),np.uint8)
img2 = np.zeros((2000,2000,3),np.uint8)
main(img,img2,pts1,pts2)
