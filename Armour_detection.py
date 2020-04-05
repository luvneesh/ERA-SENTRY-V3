# WORKING
import cv2 as cv
import numpy as np
import math
# cap=cv.VideoCapture(1)
#cap.set(cv.CAP_PROP_EXPOSURE,-6)

frame = cv.imread("pic.jpeg")
# tracker = cv.TrackerMIL_create()
rects=[]
# while True :
# ret,frame=cap.read()

# thresh=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
# # blurred=cv2.
# thresh=cv.threshold(thresh,220,255,cv.THRESH_BINARY)[1]
# thresh = cv.erode(thresh, None, iterations=2)
# thresh = cv.dilate(thresh, None, iterations=4)

# bright=cv.bitwise_and(frame,frame,mask=thresh)
hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
mask=cv.inRange(hsv,np.array([90,150,150]),np.array([150,255,255]))
final=cv.bitwise_and(frame,frame,mask=mask)
# yo=cv.cvtColor(final,cv.COLOR_BGR2GRAY)
# # yo = cv.erode(gray, None, iterations=1)
# yo = cv.dilate(yo, None, iterations=1)
# # cnts = np.array(cnts).reshape((-1,1,2)).astype(np.int32)
# cnts = cv.findContours(yo.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# #print(cnts)i
# # for i in range(15):
#     cv.drawContours(yo,cnts[i],-1,(0,255,0),3)
cv.imshow('b',final)
gray=cv.cvtColor(final,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(final, contours, -1, (0,255,0), 3)
hset=[]
# print(contours)
filters=np.ones(final.shape[:2],dtype="uint8")*255
cv.imshow('lol',filters)
for i in contours:
    x,y,w,h = cv.boundingRect(i)
    if w/h>1.8 and h>5:
        rect=cv.minAreaRect(i)
        box=cv.boxPoints(rect)
        box=np.int0(box)
        cv.drawContours(frame, [box],0,(0,0,255),2)
        centre=rect[0]
        
        cv.circle(frame,(int(centre[0]), int(centre[1])), 2, (0,0,255), 2)
    if w/h<0.5:
        cv.rectangle(final,(x,y),(x+w,y+h),(255,0,0),2)
        cv.drawContours(filters,[i],-1,1,2)
        # hset.append(h)
cv.imshow('d',final)
# cv.imshow('c',filters)
# print(hset)      
image=cv.bitwise_and(frame,frame,mask=cv.bitwise_not(filters))  
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
rectangles=[]
max_rect=[]
bgr=cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
max_area=0
ind_max=0
# for index, cont in enumerate(contours):
    
for j,i in enumerate(contours):
    rect= cv.minAreaRect(i)
    w, h=rect[1]
    rectangles.append(rect)
    area = h*w
    if(area > max_area):
        # max_rect.clear()
        ind_max=j
        max_area=area
        # max_rect.append(rect)

#     box = cv.boxPoints(rect)
#     box = np.int0(box)
#     cv.drawContours(frame, [box], 0, (0,0,255),2)
#     rectangles.append(rect)
#     angle_i=rect[2]
#     for rectangle in rectangles:
#         angle_j=rectangle[2]
#         if abs(angle_i - angle_j)<3:
#             # center= [sum(x) for x in zip(rect[0],rectangle[0])]
#             x1,y1=rect[0]
#             x2,y2=rectangle[0]
#             x=int ((x1+x2)/2)
#             y=int ((y1+y2)/2)
#             center_target=x,y
#             w, h=rectangle[1]
#             area = h*w
            
#             print(center_target)
#             cv.circle(frame,center_target,4,(0,255,0),5)
# cv.imshow('a',frame)
try:
    var = 0
    interest=[]
    # print(ind_max)
    interest.append(ind_max)
    if ind_max==0:
        interest.append(1)
    elif ind_max==(len(rectangles)-1):
        interest.append(len(rectangles)-2)
    else :
        
        rect1=rectangles[ind_max-1]
        rect2=rectangles[ind_max+1]
        rect0=rectangles[ind_max]
        angle_i=rect1[2]
        # if area==max_area:
        # #     continue
        # if(abs(max_area-area)<13):
        #     var=2
        # rectangle=rectangles[ind_max]
        # for rectangle in rectangles:
        angle_j=rect2[2]
        angle_k=rect0[2]
        diff1=abs(angle_k-angle_j)
        diff2=abs(angle_k-angle_i)
        if diff1<diff2:
            interest.append(ind_max+1)
        else:
            interest.append(ind_max-1)
        # if(difference)

    for j in interest:
                
        # w, h=rect[1]
        # area = h*w
        rect=rectangles[j]
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(frame, [box], 0, (255,255,0),2)
        # rectangles.append(rect)
    
    rect=rectangles[interest[0]]
    rectangle=rectangles[interest[1]]
    angle_i=rect[2]
    angle_j=rectangle[2]
    x1,y1=rect[0]
    x2,y2=rectangle[0]
    
    w1, h1=rect[1]
    w2, h2=rectangle[1]
    x=int ((x1+x2)/2)
    y=int ((y1+y2)/2)
    center_target=x,y        
    print(center_target)
    if abs(angle_i-angle_j)<10:
        # print("hii")
        cv.circle(frame,center_target,2,(0,255,0),2)
        # cv.rectangle(frame, (0, 0), (100, 100), (0, 255, 0), 2)
        if x1<x2:
            
            # print((x1, (y1-h1/2)), (x2, (y2+h2/2)))
            # print( "2")
            bbox=cv.rectangle(frame, (int(x1), int(y1-h1/2)), (int(x2), int(y2+h2/2)), (0,255,0),2)
        else:
            
            # print("1")
            # print((x2, (y2-h2/2)), (x1, (y1+h1/2)))
            bbox=cv.rectangle(frame, (int(x2), int(y2-h2/2)), (int(x1), int(y1+h1/2)), (0,255,0),2)
        rects.append(bbox.astype(int))
        # ok = tracker.init(image, bbox)
        
    # ok, newbox = tracker.update(frame)
    # print(ok)
    
    # if ok:
    #     p1 = (int(newbox[0]), int(newbox[1]))
    #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    #     cv.rectangle(frame, p1, p2, (0,0,255))
except:
    print('nahi mila')
cv.imwrite('lol.png',frame)
# gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray, 127, 255, 0)
# _, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cnts=sorted(contours,key=cv.contourArea,reverse=True)[:2]
# cv.drawContours(image,cnts,-1,0,-1)
# cv.imshow('b',image)
# cv.imshow('frame',yo)
# cv.imshow('mak',bright)
cv.waitKey(10)==27
    # break
    # cv.destroyAllWindows()
# cap.release()