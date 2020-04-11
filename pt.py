import numpy as np
import cv2
import cv2.aruco as aruco
import math

from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.image as img

# we will not use a built-in dictionary, but we could
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# define an empty custom dictionary with 
aruco_dict = aruco.custom_dictionary(44, 5, 1)
# add empty bytesList array to fill with 3 markers later
aruco_dict.bytesList = np.empty(shape = (44, 4, 4), dtype = np.uint8)

# add new marker(s)
mybits = np.array([[0,1,1,0,0],[1,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,1,1,0],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,1,1,1,1],[0,0,0,1,0]], dtype = np.uint8)
aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,1],[1,0,0,0,0],[0,1,1,1,0],[0,0,0,0,1],[1,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[4] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[5] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[1,1,1,1,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[6] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[7] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[8] = aruco.Dictionary_getByteListFromBits(mybits)#9

mybits = np.array([[0,1,1,1,0],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[9] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,0,1,0,0],[0,1,0,1,0],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[10] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[11] = aruco.Dictionary_getByteListFromBits(mybits)#b

mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[12] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[13] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[14] = aruco.Dictionary_getByteListFromBits(mybits)#e

mybits = np.array([[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[15] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,0],[1,0,0,0,0],[1,0,1,1,1],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[16] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[17] = aruco.Dictionary_getByteListFromBits(mybits)#h

mybits = np.array([[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[18] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[19] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[1,0,0,1,0],[1,1,1,0,0],[1,0,0,1,0],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[20] = aruco.Dictionary_getByteListFromBits(mybits)#k

mybits = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[21] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[22] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[23] = aruco.Dictionary_getByteListFromBits(mybits)#n

mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[24] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[25] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,1,0],[0,1,1,0,1]], dtype = np.uint8)
aruco_dict.bytesList[26] = aruco.Dictionary_getByteListFromBits(mybits)#q

mybits = np.array([[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,1,0],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[27] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,1,1,1],[1,0,0,0,0],[0,1,1,1,0],[0,0,0,0,1],[1,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[28] = aruco.Dictionary_getByteListFromBits(mybits)#s
mybits = np.array([[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[29] = aruco.Dictionary_getByteListFromBits(mybits)#t

mybits = np.array([[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[30] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[31] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[0,1,0,1,0]], dtype = np.uint8)
aruco_dict.bytesList[32] = aruco.Dictionary_getByteListFromBits(mybits)#w

mybits = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]], dtype = np.uint8)
aruco_dict.bytesList[33] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[34] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[35] = aruco.Dictionary_getByteListFromBits(mybits)#z

mybits = np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,1,1,0],[0,0,0,0,0],[0,0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[36] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,0,1,0],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[37] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[38] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[0,0,1,0,0],[0,1,1,1,1],[1,1,1,1,1],[0,0,0,1,1],[0,0,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[39] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,0,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,0,0,0],[1,1,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[40] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[41] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[42] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,1,1],[1,0,1,0,1],[1,1,1,1,1],[1,0,1,0,1],[1,1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[43] = aruco.Dictionary_getByteListFromBits(mybits)
# cap = cv2.VideoCapture('video/Sentry_1.mkv')
# ret, frame = cap.read()
img= cv2.imread("lol5.png")
scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
frame1=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_yel = np.array([23,41,133], dtype = "uint8")
upper_yel = np.array([60,255,255], dtype = "uint8")
frame1 = cv2.inRange(frame1, lower_yel, upper_yel)
frame1=255-frame1
frame1 = cv2.adaptiveThreshold(frame1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# cv2.imshow('1', frame1)
# cv2.waitKey(-1)
#lists of ids and the corners beloning to each id
corners, ids, rejectedImgPoints = aruco.detectMarkers(frame1, aruco_dict)
# draw markers on farme
frame1 = aruco.drawDetectedMarkers(frame, corners, (ids),  borderColor=(0, 255, 0))
# print(corners)
# cv2.waitKey(0)
meri_aruco_ki_list=[]
m=0
while m < (len(ids)):
    x = int((corners[m][0][0][0] + corners[m][0][1][0] + corners[m][0][2][0] + corners[m][0][3][0]) / 4)
    y = int((corners[m][0][0][1] + corners[m][0][1][1] + corners[m][0][2][1] + corners[m][0][3][1]) / 4)
    print(x,y,ids[m])
    m=m+1
if True:
    pts1 = np.float32([[142,115], [169,63], [314,75], [368,141]]) 
    pts2 = np.float32([[347,404], [225,617], [105,404], [226,190]])#4 top vaale
else:
    pts1 = np.float32([[751,286], [1556,444], [512,450], [445,362]]) 
    pts2 = np.float32([[225,403], [225,190], [104,453], [105,403]])

# dst=[(),(),(),()]
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(frame,M,(449, 808))
print(M)
cv2.imshow('frame',dst)
cv2.waitKey(0)
cv2.imwrite('pt.jpg',dst)