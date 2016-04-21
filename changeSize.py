import cv2
import numpy as np
import glob
import copy


'''
while True:
    im = cv2.imread('input/camera1/picture059.jpg')

    im2 = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow('image1', im)
    cv2.imshow('image2', im2)

    k = cv2.waitKey(0)

    if k == 27:
        break

cv2.destroyAllWindows();

'''


im = 'input/camera2/picture065.jpg'
img = cv2.imread(im)
#img = cv2.resize(im, (0,0), fx = 0.5, fy= 0.5)

cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cimg = cv2.medianBlur(cimg,17)
img3 = cv2.GaussianBlur(cimg, (5,5), 0)

#cv2.imshow('img1', img2)
#cv2.imshow('img', img3)

ret,thresh1 = cv2.threshold(cimg, 70, 255,cv2.THRESH_BINARY)
t2 = copy.copy(thresh1)
cv2.imshow('thresh', t2)

x, y  = thresh1.shape
arr = np.zeros((x, y, 3), np.uint8)
final_contours= []
image, contours, hierarchy = cv2.findContours(t2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    #cv2.drawContours(img, [cnt], -1, [0, 255,255])

    if cv2.contourArea(cnt) > 2500 and cv2.contourArea(cnt) < 20000 :
        cv2.drawContours(img, [cnt],-1, [0, 255, 255])
        cv2.fillConvexPoly(arr, cnt, [255, 255, 255])
        final_contours.append(cnt)

arr = cv2.bitwise_not(arr)
cv2.imshow('image1', img)
cv2.imshow('image2',arr)
cv2.imwrite('input/camera2/sample1.jpg', arr)
k = cv2.waitKey(0)

cv2.destroyAllWindows()
