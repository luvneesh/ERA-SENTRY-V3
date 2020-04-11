import numpy as np
import cv2

img= cv2.imread("lol5.png")
scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
pts1 = np.float32([[142,115], [169,63], [314,75], [368,141]]) 
pts2 = np.float32([[347,404], [225,617], [105,404], [226,190]])#4 top vaale
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(frame,M,(449, 808))
# print(M)
cv2.imshow('frame1',dst)
cv2.waitKey(0)
cv2.imwrite('pt1.jpg',dst)
img= cv2.imread("lol6.png")
scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
pts1 = np.float32([[178,141],[211,79],[390,91],[468,177]])
pts2 = np.float32([[105,404],[225,190],[347,403],[226,618]])#4 top vaale
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(frame,M,(449, 808))
# print(M)
cv2.imshow('frame2',dst)
if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()
else: 
    cv.waitKey(0)
cv2.imwrite('pt2.jpg',dst)

# dst=[(),(),(),()]
