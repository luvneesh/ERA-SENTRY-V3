import cv2 as cv
import numpy as np
import math
import time

cv.startWindowThread()
cap = cv.VideoCapture('video/Sentry_1.mkv')
img= cv.imread("Sample1.png") #for dimentioning 

scale_percent = 40 
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv.resize(img, dim, interpolation = cv.INTER_AREA)

'''Mapping between images- to be selected during setup via GUI'''
pts1 = np.float32([[142,115], [169,63], [314,75], [368,141]]) #example mapping
pts2 = np.float32([[347,404], [225,617], [105,404], [226,190]])

M = cv.getPerspectiveTransform(pts1,pts2)
map_img= cv.imread("arena4.png")

'''optional - for ouput
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output1.mp4', fourcc, 15.0, (449,809),True)
'''
frames=1
while(frames<140):
    
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
    ''' frame : Health Bar Detection '''
    hsv=cv.cvtColor(frame2,cv.COLOR_BGR2HSV)
    mask=cv.inRange(hsv,np.array([0, 0,0]),np.array([255,200,255])) #rejection of overexposure
    frame2=cv.bitwise_and(frame2,frame2,mask=mask)
    ekkaurmask=cv.inRange(frame2,np.array([0, 0,235]),np.array([255,255,255])) #RED
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.dilate(ekkaurmask,kernel,iterations = 2)
    contours, hierarchy = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    '''Use of Health Bar for Pose Estimation'''
    if len(contours) != 0:
        i = max(contours, key = cv.contourArea) #HealthBar Contour
        rect=cv.minAreaRect(i)
        angle=rect[2]
        angle=angle-90
        if angle<-179:
            angle=angle+180
        length = 20
        box=cv.boxPoints(rect)
        box=np.int0(box)
        centre=rect[0]
        P1=centre
        P2 =  ((P1[0] + length * np.cos(angle * 3.14 / 180.0)), (P1[1] + length * np.sin(angle * 3.14 / 180.0)))
        '''Coordinates, Orientation Extracted'''

        cv.drawContours(frame, [box],0,(255,255,255),2)
        point1= np.float32([[centre[0]],[centre[1]],[1]])
        point2= np.float32([[P2[0]],[P2[1]],[1]])
        point1_new=np.matmul(M,point1)
        point1_new=point1_new/point1_new[2]
        point2_new=np.matmul(M,point2)
        point2_new=point2_new/point2_new[2]
        cv.arrowedLine(map_img,(int(point1_new[0]-40), int(point1_new[1]+165)),(int(point2_new[0]-40), int(point2_new[1]+165)),(0,0,255),2)
        cv.circle(map_img,(int(point1_new[0]-40), int(point1_new[1]+165)), 15, (255,255,255), 2)
        cv.arrowedLine(frame4,(int(centre[0]), int(centre[1])),(int(P2[0]), int(P2[1])),(0,0,255),4)
        print(frames)
        # cv.putText(frame, "angle : " + str(int(angle)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
   
    dest = cv.warpPerspective(frame,M,(449, 808))
    dest1=dest[0:600, 0:449]
    dest=cv.resize(dest1,(449,808), interpolation = cv.INTER_CUBIC)
     
    # print('image dtype ',map_img.dtype)
    out.write(map_img)
    # map_img=cv.add(map_img,dest,dtype = cv.CV_8U)
    cv.imshow('b',map_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.waitKey(0)