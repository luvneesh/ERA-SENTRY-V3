import numpy as np
import cv2
import cv2.aruco as aruco
import math

from tkinter import *
from tkinter import filedialog

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

frame = cv2.imread("pic.jpg")
frame1=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=22.0, tileGridSize=(10,30))
# frame1 = clahe.apply(frame1)

# frame1 = cv2.adaptiveThreshold(frame1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow('1', frame1)
#lists of ids and the corners beloning to each id
corners, ids, rejectedImgPoints = aruco.detectMarkers(frame1, aruco_dict)
# draw markers on farme
frame1 = aruco.drawDetectedMarkers(frame, corners, (ids),  borderColor=(0, 255, 0))
print(ids)
center=[0]*(2*len(ids))
m=0
while m < (len(ids)):
    x = int((corners[m][0][0][0] + corners[m][0][1][0] + corners[m][0][2][0] + corners[m][0][3][0]) / 4)
    y = int((corners[m][0][0][1] + corners[m][0][1][1] + corners[m][0][2][1] + corners[m][0][3][1]) / 4)
    print(x, y)
    center[2*m]=x
    center[2*m+1]=y
    m=m+1

cv2.imshow('frame',frame)
cv2.imwrite('detected_markers1.jpg',frame)

root=Tk()

frame = Frame(root, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH,expand=1)

img = ImageTk.PhotoImage(Image.open('detected_markers1.jpg'))
canvas.create_image(0,0,image=img,anchor="nw")
canvas.config(scrollregion=canvas.bbox(ALL))

#function to be called when mouse is clicked
pts1=[[0]*2]*4
pts2=[[0]*2]*4
pts=[0]*8
i=0
pt=[[29, 59], [58, 64], [29, 69], [109, 125], [109, 131], [109, 137], [86, 131], [132, 131], [86, 232], [92, 202], [98, 232], [232, 131], [232, 64], [232, 59], [232, 69],[261, 64], [203, 64], [203, 197], [232, 193], [232, 197], [232, 202], [262, 197], [364, 30], [370, 59], [376, 30], [354, 125], [354, 131], [354, 137], [376, 131], [332, 131], [436, 193], [407, 197], [436, 202]]
# print(center)
def mindis(point):
    global center
    a=0
    mindist=math.sqrt((center[0]-point.x)*(center[0]-point.x)+(center[1]-point.y)*(center[1]-point.y))
    for i in range(len(ids)):
        # print("hi1")
        dist=math.sqrt((center[2*i]-point.x)*(center[2*i]-point.x)+(center[2*i+1]-point.y)*(center[2*i+1]-point.y))
        # print(dist, mindist)
        if dist < mindist:
            # print("hi")
            mindist=dist
            a=i
    return a
def printcoords(event):
    #outputting x and y coords to console
    global i, pts, center, pts1
    print (event.x,event.y)
    # a=0
    # a=mindis(event)
    # pts[i]=center[2*a]
    # pts[i+1]=center[2*a+1]
    pts[i]=event.x
    pts[i+1]=event.y
    canvas.create_oval(pts[i]-2, pts[i+1]-2, pts[i]+2, pts[i+1]+2, fill="#476042")
    i=i+2
    if i==8:
        print(pts)
        pts1 = np.float32([[pts[0],pts[1]],[pts[2],pts[3]],[pts[4], pts[5]],[pts[6], pts[7]]])
        # print("hi")
        # print(pts1)
        # sys.exit()
        canvas.destroy()
        
canvas.bind("<Button 1>",printcoords)
root.mainloop()




root1=Tk()

frame1 = Frame(root1, bd=2, relief=SUNKEN)
frame1.grid_rowconfigure(0, weight=1)
frame1.grid_columnconfigure(0, weight=1)
xscroll1 = Scrollbar(frame1, orient=HORIZONTAL)
xscroll1.grid(row=1, column=0, sticky=E+W)
yscroll1 = Scrollbar(frame1)
yscroll1.grid(row=0, column=1, sticky=N+S)
canvas1 = Canvas(frame1, bd=0, xscrollcommand=xscroll1.set, yscrollcommand=yscroll1.set)
canvas1.grid(row=0, column=0, sticky=N+S+E+W)
xscroll1.config(command=canvas1.xview)
yscroll1.config(command=canvas1.yview)
frame1.pack(fill=BOTH,expand=1)

img1 = ImageTk.PhotoImage(Image.open('arena4.png'))
canvas1.create_image(0,0,image=img1,anchor="nw")
canvas1.config(scrollregion=canvas1.bbox(ALL))

#function to be called when mouse is clicked
pts=[0]*8
i=0
points=[[120, 758], [398, 658], [224, 658], [224, 618], [234, 618], [224, 404], [345, 454], [345, 404], [355, 404], [103, 404], [113, 404], [103, 454], [224, 190], [224, 230], [50, 170], [100, 160], [338, 100], [348, 50]]
# print(center)
def mindis1(point):
    global points
    a=0
    mindist=10000
    p=-1
    for pt in points:
        p=p+1
        dist=math.sqrt(int(math.pow(pt[0]-point.x, 2))+int(math.pow(pt[1]-point.y, 2)))
        if dist<mindist:
            mindist=dist
            a=p
    return a
            
def printcoords1(event):
    #outputting x and y coords to console
    global i, pts, points, pts2
    print (event.x,event.y)
    # a=0
    # a=mindis1(event)
    # pts[i]=points[a][0]
    # pts[i+1]=points[a][1]
    pts[i]=event.x
    pts[i+1]=event.y
    canvas1.create_oval(pts[i]-2, pts[i+1]-2, pts[i]+2, pts[i+1]+2, fill="#476042")
    i=i+2
    if i==8:
        print(pts)
        pts2 = np.float32([[pts[0],pts[1]],[pts[2],pts[3]],[pts[4], pts[5]],[pts[6], pts[7]]])
        # print(pts2)
        # sys.exit()
        canvas1.destroy()
        
canvas1.bind("<Button 1>",printcoords1)
root1.mainloop()

img = cv2.imread("pic.jpg")
print("pt")
print(pts1)
print(pts2)
M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(808, 448))

# plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(),plt.imshow(dst),plt.title('Output')
plt.show()