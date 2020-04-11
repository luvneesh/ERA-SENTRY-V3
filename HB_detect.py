import cv2 as cv
import numpy as np
import math
import time

cv.startWindowThread()
cap = cv.VideoCapture('video/Sentry_2.mkv')

frames=1
while(frames<268):
    
    frames=frames+1
    ret, img = cap.read()
    scale_percent = 40 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    frame2=frame.copy()
    frame3=frame.copy()
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
        print(P1,P2)
        # cv.circle(frame,(int(centre[0]), int(centre[1])), 7, (0,0,0), -1)
        # cv.circle(frame,(int(centre[0]), int(centre[1])), 50, (77,93,100), -1)
        cv.arrowedLine(frame,(int(centre[0]), int(centre[1])),(int(P2[0]), int(P2[1])),(0,0,255),2)
        # cv.putText(frame, "angle : " + str(int(angle)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    else :
        angle=0
        # print(P1,P2)
        centre=(0,0)
    cv.imshow('b',frame)
    ''' hb detection ends'''
    ''' bot detection'''
    
    # Mask=cv.inRange(frame3,np.array([0, 0, 0]),np.array([30,30,40]))
    # Mask=cv.bitwise_not(Mask)
    # kernel = np.ones((5,5),np.uint8)
    # # cv.imshow('b',Mask)
    # dilation = cv.dilate(Mask,kernel,iterations = 1)

    # erosion = cv.erode(dilation,kernel,iterations = 16)
    # # cv.imshow('b',erosion)
    # Mask=cv.bitwise_not(erosion)
    # # cv.imshow('b',Mask)
    # # cv.waitKey(0)
    # # cv.imshow('dil',dilation)
    # # mask2=mask1
    # # mask2=cv.inRange(hsv,np.array([170, 70, 50]),np.array([180, 255, 255]))
    # # mask=cv.bitwise_or(mask1,mask2)
    # contours, hierarchy = cv.findContours(Mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # frame4=frame3.copy()
    # x,y,w,h=0,0,0,0
    # for i in contours:
    #     x,y,w,h= cv.boundingRect(i)
    #     # paddding=100
    #     rect=cv.minAreaRect(i)
    #     # print(x,y,w,h)
    #     cv.rectangle(frame4,(x,y),(x+w,y+h),(77,93,100),1)
    #     # paddding=100
    #     # rect[0][0]-=paddding
    #     # rect[0][1]-=paddding
    #     # rect[1][0]+=paddding
    #     # rect[1][0]+=paddding
    #     box=cv.boxPoints(rect)
    #     # print(box)
    #     # paddding=100
    #     # [0][0]-=paddding
    #     # rect[0][1]-=paddding
    #     # rect[1][0]+=paddding
    #     # rect[1][0]+=paddding
    #     box=np.int0(box)
    #     centre1=rect[0]
    #     # cv.rectangle(frame3,box[0],box[2],(0,255,255))
    #     cv.drawContours(frame4, [box],0,(255,255,255),2)
    #     # cv.circle(frame3,(int(centre[0]), 200, 2, (77,93,100), 2)
    #     # print(centre)
    #     cv.circle(frame2,(int(centre1[0]), int(centre1[1])), int (w*0.95), (77,93,100), -1)
    #     cv.circle(frame2,(int(centre1[0]), int(centre1[1])), 7, (0,0,0), -1)
    #     cv.circle(frame2,(int(centre[0]), int(centre[1])), 7, (0,0,255), -1)
    # # cv.imshow('b',frame2)
    # # frame4 = cv.flip(frame4,0)
    

    # # Display tracker type on frame
    # # cv.putText(frame, tracker_type + " Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # # Display FPS on frame
    # # cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # # Display result
    # # cv.imshow("Tracking", frame)

    # # cv.imshow('Detection',frame4)
    # # out.write(frame4)
    # # cv.imshow('bcc',frame2)
    # # cv.imshow('bccc',frame1)
    # # cv.imshow('c',frame)
    '''
                    frame 2 is to be used for PT
                        frame 4 is to be used for hb detection, direction needs to be appended on frame 3
                    bot detection ends here 
    '''
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.waitKey(0)