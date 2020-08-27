import cv2 as cv
import numpy as np
import math
import time

cv.startWindowThread()
cap = cv.VideoCapture('video/Sentry_2.mkv')
img= cv.imread("Sample2.png")
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output2.mp4', fourcc, 15.0, (449,809),True)

scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv.resize(img, dim, interpolation = cv.INTER_AREA)
pts1 = np.float32([[178,141],[211,79],[390,91],[468,177]])
pts2 = np.float32([[105,404],[225,190],[347,403],[226,618]])#4 top vaale
M = cv.getPerspectiveTransform(pts1,pts2)

frames=1
while(frames<299):
    map_img= cv.imread("arena4.png")
    frames=frames+1
    ret, img = cap.read()
    scale_percent = 40 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    frame2=frame.copy()
    frame3=frame.copy()
    frame4=np.zeros(frame.shape)
    ''' frame : hb detection '''
    hsv=cv.cvtColor(frame2,cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv,np.array([0, 0,0]),np.array([255,200,255]))
    frame2=cv.bitwise_and(frame2,frame2,mask=mask)
    ekkaurmask=cv.inRange(frame2,np.array([0, 0,235]),np.array([255,255,255]))
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.dilate(ekkaurmask,kernel,iterations = 2)
    contours, hierarchy = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        i = max(contours, key = cv.contourArea)
        rect=cv.minAreaRect(i)
        angle=rect[2]
        # print( angle)
        angle=angle-90
        if angle<-179:
            angle=angle+180
        
        # int angle = 45;
        length = 20
        box=cv.boxPoints(rect)
        box=np.int0(box)
        centre=rect[0]
        # np.cos()
        P1=centre
        P2 =  ((P1[0] + length * np.cos(angle * 3.14 / 180.0)), (P1[1] + length * np.sin(angle * 3.14 / 180.0)))
        cv.drawContours(frame, [box],0,(255,255,255),2)
        # print(P1,P2)
        # cv.circle(frame,(int(centre[0]), int(centre[1])), 7, (0,0,0), -1)
        # cv.circle(frame,(int(centre[0]), int(centre[1])), 50, (77,93,100), -1)
        point1= np.float32([[centre[0]],[centre[1]],[1]])
        point2= np.float32([[P2[0]],[P2[1]],[1]])
        point1_new=np.matmul(M,point1)
        point1_new=point1_new/point1_new[2]
        # point1_new[1]=point1_new[1]
        point2_new=np.matmul(M,point2)
        point2_new=point2_new/point2_new[2]
        cv.arrowedLine(map_img,(int(point1_new[0]), int(point1_new[1]-30)),(int(point2_new[0]), int(point2_new[1]-30)),(0,0,255),2)
        cv.circle(map_img,(int(point1_new[0]), int(point1_new[1]-30)), 15, (255,255,255), 2)
        # cv.arrowedLine(frame4,(int(centre[0]), int(centre[1])),(int(P2[0]), int(P2[1])),(0,0,255),4)
        # print(frames)
        # print('image dtype ',map_img.dtype)
        # print(point_new)
        # map_img=cv.add(map_img,dest,dtype = cv.CV_8U)
        # cv.circle(map_img,(int(point_new[0]), int(point_new[1]-30)), 5, (0,0,255), -1)
        cv.imshow('b',map_img)
        out.write(map_img)
        # cv.imwrite('tracing2.png',map_img)
        # cv.imshow('l',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.waitKey(0)
out.release()