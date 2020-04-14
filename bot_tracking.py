# WORKING
import cv2 as cv
import numpy as np
import math
# cap=cv.VideoCapture(1)
#cap.set(cv.CAP_PROP_EXPOSURE,-6)
cv.startWindowThread()
cap = cv.VideoCapture('video/Sentry_2.mkv')
# out = cv.VideoWriter('')
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('input2_1080.mp4', fourcc, 15.0, (1920,1080),True)
frames=1
while(frames<299):
    # img = cv.imread("lol3.png")
    # frames=frames+1
    ret, img = cap.read()
    # print (img.shape)
    scale_percent = 40 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # print(dim)
    # resize image
    frame=img.copy()
    frame2=frame.copy()
    frame3=frame2.copy()
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('1.avi',fourcc, 24.0, (frame2.shape[1],frame2.shape[0]))
    # tracker = cv.TrackerMIL_create()
    # cv.imshow('bc',frame3)
    # cv.waitKey(-1)
    rects=[]
    # 

    # while True :
    # ret,frame=cap.read()

    # thresh=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # # blurred=cv.
    # thresh=cv.threshold(thresh,220,255,cv.THRESH_BINARY)[1]
    # thresh = cv.erode(thresh, None, iterations=2)
    # thresh = cv.dilate(thresh, None, iterations=4)
    '''here 

    # # bright=cv.bitwise_and(frame,frame,mask=thresh)
    # ekkaurmask=cv.inRange(frame,np.array([0, 0,180]),np.array([255,255,255]))
    # frame1=cv.bitwise_and(frame,frame,mask=ekkaurmask)
    # hsv=cv.cvtColor(frame1,cv.COLOR_BGR2HSV)
    # mask1=cv.inRange(hsv,np.array([0, 70, 150]),np.array([10,255,255]))
    # mask2=mask1
    # # mask2=cv.inRange(hsv,np.array([170, 70, 50]),np.array([180, 255, 255]))
    # mask=cv.bitwise_or(mask1,mask2)
    # frame1=cv.bitwise_and(frame1,frame,mask=mask)
    # # yo=cv.cvtColor(final,cv.COLOR_BGR2GRAY)
    # # # yo = cv.erode(gray, None, iterations=1)
    # # yo = cv.dilate(yo, None, iterations=1)
    # # # cnts = np.array(cnts).reshape((-1,1,2)).astype(np.int32)
    # # cnts = cv.findContours(yo.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # # #print(cnts)i
    # # # for i in range(15):
    # #     cv.drawContours(yo,cnts[i],-1,(0,255,0),3)
    # '''
    # this is for elimination of hb. already done by applying consecutive masks
    # # cv.waitKey()
    # gray=cv.cvtColor(final,cv.COLOR_BGR2GRAY)
    # # ret, thresh = cv.threshold(gray, 100, 255, 0)
    # # cv.imshow('b',gray)
    # cv.waitKey()
    # contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # # cv.drawContours(final, contours, -1, (0,255,0), 3)
    # hset=[]
    # # cv.imshow('b',final)
    # cv.waitKey()
    # print(contours)
    # filters=np.ones(final.shape[:2],dtype="uint8")*255

    # frame1=frame
    # # cv.imshow('lol',filters)
    # # cv.waitKey()
    # for i in contours:
    #     x,y,w,h = cv.boundingRect(i)
    #     if True :
    #         rect=cv.minAreaRect(i)
    #         box=cv.boxPoints(rect)
    #         box=np.int0(box)
    #         cv.drawContours(final, [box],0,(0,255,255),2)
    #         centre=rect[0]
            
    #         # cv.circle(frame,(int(centre[0]), int(centre[1])), 2, (0,0,255), 2)
    #     if False:
    #         cv.rectangle(final,(x,y),(x+w,y+h),(255,0,0),2)
    #         cv.drawContours(filters,[i],-1,1,2)
    #         # hset.append(h)

    # # cv.imshow('c',filters)
    # # print(hset)      
    # image=frame1
    # # cv.imshow('d',image)
    # gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # ret, thresh = cv.threshold(gray, 127, 255, 0)
    # '''
    # contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # rectangles=[]
    # max_rect=[]
    # for i in contours:
    #     x,y,w,h = cv.boundingRect(i)
    #     rect=cv.minAreaRect(i)
    #     box=cv.boxPoints(rect)
    #     box=np.int0(box)
    #     cv.drawContours(frame2, [box],0,(255,255,255),3)
    #     centre=rect[0]


    # # bgr=cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
    # ''' ^^ ye kya chutiyaap hai bc ^^ '''

    # #lolololoolololololololololololololololololol
    # '''
    # this is for preffered armour plate, we dont need that here 
    # max_area=0
    # ind_max=0
    # # for index, cont in enumerate(contours):
    # cv.waitKey()    
    # for j,i in enumerate(contours):
    #     rect= cv.minAreaRect(i)
    #     w, h=rect[1]
    #     rectangles.append(rect)
    #     area = h*w
    #     if(area > max_area):
    #         # max_rect.clear()
    #         ind_max=j
    #         max_area=area
    #         # max_rect.append(rect)

    # #     box = cv.boxPoints(rect)
    # #     box = np.int0(box)
    # #     cv.drawContours(frame, [box], 0, (0,0,255),2)
    # #     rectangles.append(rect)
    # #     angle_i=rect[2]
    # #     for rectangle in rectangles:
    # #         angle_j=rectangle[2]
    # #         if abs(angle_i - angle_j)<3:
    # #             # center= [sum(x) for x in zip(rect[0],rectangle[0])]
    # #             x1,y1=rect[0]
    # #             x2,y2=rectangle[0]
    # #             x=int ((x1+x2)/2)
    # #             y=int ((y1+y2)/2)
    # #             center_target=x,y
    # #             w, h=rectangle[1]
    # #             area = h*w
                
    # #             print(center_target)
    # #             cv.circle(frame,center_target,4,(0,255,0),5)
    # # cv.imshow('a',frame)

    # try:
    #     var = 0
    #     interest=[]
    #     # print(ind_max)
    #     interest.append(ind_max)
    #     if ind_max==0:
    #         interest.append(1)
    #     elif ind_max==(len(rectangles)-1):
    #         interest.append(len(rectangles)-2)
    #     else :
            
    #         rect1=rectangles[ind_max-1]
    #         rect2=rectangles[ind_max+1]
    #         rect0=rectangles[ind_max]
    #         angle_i=rect1[2]
    #         # if area==max_area:
    #         # #     continue
    #         # if(abs(max_area-area)<13):
    #         #     var=2
    #         # rectangle=rectangles[ind_max]
    #         # for rectangle in rectangles:
    #         angle_j=rect2[2]
    #         angle_k=rect0[2]
    #         diff1=abs(angle_k-angle_j)
    #         diff2=abs(angle_k-angle_i)
    #         if diff1<diff2:
    #             interest.append(ind_max+1)
    #         else:
    #             interest.append(ind_max-1)
    #         # if(difference)

    #     for j in interest:
                    
    #         # w, h=rect[1]
    #         # area = h*w
    #         rect=rectangles[j]
    #         box = cv.boxPoints(rect)
    #         box = np.int0(box)
    #         cv.drawContours(frame, [box], 0, (255,255,0),2)
    #         # rectangles.append(rect)
        
    #     rect=rectangles[interest[0]]
    #     rectangle=rectangles[interest[1]]
    #     angle_i=rect[2]
    #     angle_j=rectangle[2]
    #     x1,y1=rect[0]
    #     x2,y2=rectangle[0]
        
    #     w1, h1=rect[1]
    #     w2, h2=rectangle[1]
    #     x=int ((x1+x2)/2)
    #     y=int ((y1+y2)/2)
    #     center_target=x,y        
    #     print(center_target)
    #     if abs(angle_i-angle_j)<10:
    #         # print("hii")
    #         cv.circle(frame,center_target,2,(0,255,0),2)
    #         # cv.rectangle(frame, (0, 0), (100, 100), (0, 255, 0), 2)
    #         if x1<x2:
                
    #             # print((x1, (y1-h1/2)), (x2, (y2+h2/2)))
    #             # print( "2")
    #             bbox=cv.rectangle(frame, (int(x1), int(y1-h1/2)), (int(x2), int(y2+h2/2)), (0,255,0),2)
    #         else:
                
    #             # print("1")
    #             # print((x2, (y2-h2/2)), (x1, (y1+h1/2)))
    #             bbox=cv.rectangle(frame, (int(x2), int(y2-h2/2)), (int(x1), int(y1+h1/2)), (0,255,0),2)
    #         rects.append(bbox.astype(int))
    #         # ok = tracker.init(image, bbox)
            
    #     # ok, newbox = tracker.update(frame)
    #     # print(ok)
        
    #     # if ok:
    #     #     p1 = (int(newbox[0]), int(newbox[1]))
    #     #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    #     #     cv.rectangle(frame, p1, p2, (0,0,255))
    # except:
    #     print('nahi mila')
    # till here
    '''
    '''
    # cv.imshow('lol',frame2)
    ''' bot detection'''

    Mask=cv.inRange(frame3,np.array([0, 0, 0]),np.array([30,30,40]))
    Mask=cv.bitwise_not(Mask)
    kernel = np.ones((5,5),np.uint8)
    # cv.imshow('b',Mask)
    dilation = cv.dilate(Mask,kernel,iterations = 1)

    erosion = cv.erode(dilation,kernel,iterations = 12)
    # cv.imshow('b',erosion)
    Mask=cv.bitwise_not(erosion)
    # cv.imshow('b',Mask)
    # cv.imshow('dil',dilation)
    # mask2=mask1
    # mask2=cv.inRange(hsv,np.array([170, 70, 50]),np.array([180, 255, 255]))
    # mask=cv.bitwise_or(mask1,mask2)
    contours, hierarchy = cv.findContours(Mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    frame4=frame3.copy()
    i= max(contours, key = cv.contourArea)
    x,y,w,h= cv.boundingRect(i)
    # paddding=100
    rect=cv.minAreaRect(i)
    # cv.rectangle(frame3,(x,y),(x+w,y+h),(77,93,100),-1)
    # paddding=100
    # rect[0][0]-=paddding
    # rect[0][1]-=paddding
    # rect[1][0]+=paddding
    # rect[1][0]+=paddding
    box=cv.boxPoints(rect)
    # print(box)
    # paddding=100
    # [0][0]-=paddding
    # rect[0][1]-=paddding
    # rect[1][0]+=paddding
    # rect[1][0]+=paddding
    box=np.int0(box)
    centre=rect[0]
    # cv.rectangle(frame3,box[0],box[2],(0,255,255))
    cv.drawContours(frame4, [box],0,(255,255,255),2)
    # cv.circle(frame3,(int(centre[0]), 200, 2, (77,93,100), 2)
    print(centre)
    cv.circle(frame3,(int(centre[0]), int(centre[1])), int (w*0.95), (77,93,100), -1)
    cv.circle(frame3,(int(centre[0]), int(centre[1])), 7, (0,0,0), -1)
    # print(box)
    cv.imshow('bc',frame3)
    out.write(frame4)
    cv.imshow('bcc',frame4)
    # cv.imshow('bccc',frame1)
    # cv.imshow('c',frame)
    '''
                    frame 3 is to be used for PT
                        frame 4 is to be used for hb detection, direction needs to be appended on frame 3
                    bot detection ends here 
    '''
        # Exit if ESC pressed
    k = cv.waitKey(1) & 0xff
    if k == 27 : break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
cv.waitKey(1)
    # break
    # cv.destroyAllWindows()
# cap.release()