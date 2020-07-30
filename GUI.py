import tkinter
from tkinter import ttk

from tkinter import messagebox
import cv2
import PIL.Image, PIL.ImageTk
import time
import glob
import numpy as np
from martinez.contour import Contour
from martinez.point import Point
from martinez.polygon import Polygon
from martinez.boolean import OperationType, compute

def parseTXT(txtfile):
    file = open(txtfile, 'r') 
    return txtfile, file.readlines()

class App:
    def __init__(self, window, window_title, data_source=0, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.frame = []
        self.M = [[1,0,0],[0,1,0],[0,0,1]]
        self.maxWidth = 0
        self.maxHeight = 0

        self.markFlag = 0   # Mark intereset area
        self.posList = []   # List of marked positions
        self.pointList = [] # List of marked position (type Point)
        self.polyShape = Polygon([Contour([Point(-1,-1), Point(-2,-2), Point(-2,-1)],[],True)])

        self.markPerspectiveFlag = 0    # Mark perspective area
        self.perspectiveList = []       # List of marked positions for perspective correction

        self.frameID = 0    # Current frameID
        self.chosenTrack = 0# Chosen track ID
        self.Tracks = []    # list of Tracks in image
        self.skipFrames = 0 # frame change direction (1 for forward, -1 for backward)

        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.attributes("-toolwindow", 1)
        self.window.attributes("-alpha", 0.7)
        cv2.namedWindow('Vid', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Vid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Button that lets the user navigate the frames
        self.nav = tkinter.Label(self.window)
        self.nav.pack(side="top", expand=True, fill='both')
        self.btn_prev=tkinter.Button(self.nav, text="Prev", command=self.prevImage)
        self.btn_prev.pack(side="left", expand=True, fill='both')
        self.btn_snapshot=tkinter.Button(self.nav, text="Play/Stop",  command=self.play)
        self.btn_snapshot.pack(side="left", expand=True, fill='both')
        self.btn_next=tkinter.Button(self.nav, text="Next", command=self.nextImage)
        self.btn_next.pack(side="left", expand=True, fill='both')
        
        self.sep = ttk.Separator(self.window)
        self.sep.pack(side="top", fill="both", padx=4, pady=4)

        # Button that lets the user mark a polygon
        self.poly = tkinter.Label(self.window)
        self.poly.pack(side="bottom", expand=True, fill='both')
        self.btn_polygon=tkinter.Button(self.poly, text="Mark Poly", command=self.markPoly)
        self.btn_polygon.pack(side="left", expand=True, fill='both')
        self.btn_clrPly=tkinter.Button(self.poly, text="Clear Poly", command=self.clearPoly)
        self.btn_clrPly.pack(side="left", expand=True, fill='both')

        # Button that lets the user mark the perspective
        self.perspectiveLabel = tkinter.Label(self.window, text="top pane")
        self.perspectiveLabel.pack(side="bottom", expand=True, fill='both')
        self.btn_Perspolygon=tkinter.Button(self.perspectiveLabel, text="Mark Perspective", command=self.markPerspective)
        self.btn_Perspolygon.pack(side="left", expand=True, fill='both')
        self.btn_applyPers=tkinter.Button(self.perspectiveLabel, text="Apply", command=lambda: self.four_point_transform(self.perspectiveList))
        self.btn_applyPers.pack(side="left", expand=True, fill='both')
        self.btn_clrPersPly=tkinter.Button(self.perspectiveLabel, text="Clear Perspective", command=self.clearPerspective)
        self.btn_clrPersPly.pack(side="left", expand=True, fill='both')

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 30
        if (video_source == 0) and not (data_source == 0):
            self.data = glob.glob(data_source + "\\*.txt")
            self.updateImages()

        self.window.mainloop()

    def nextImage(self):
        self.frameID = self.frameID + 1

    def prevImage(self):
        self.frameID = self.frameID - 1

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.window.destroy()
    
    def clearPoly(self):
        self.posList = []
        self.pointList = []
        self.polyShape = Polygon([Contour([Point(-1,-1), Point(-2,-2), Point(-2,-1)],[],True)])

    def clearPerspective(self):
        self.perspectiveList = []
        self.maxWidth = 0

    def markPoly(self):
        self.markFlag = 1

    def markPerspective(self):
        self.markPerspectiveFlag = 1

    def play(self):
        # Get a frame from the video source
        if self.skipFrames == 0:
            self.skipFrames = 1
        else:
            self.skipFrames = 0

    def four_point_transform(self, rect):
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        self.maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        self.maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [self.maxWidth - 1, 0],
            [self.maxWidth - 1, self.maxHeight - 1],
            [0, self.maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it

        self.M = cv2.getPerspectiveTransform(np.array(rect, np.float32), dst)

    def draw_circle(self, event,x,y,flags,param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.markFlag = 0
            self.markPerspectiveFlag = 0

        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.markFlag == 0) and (self.markPerspectiveFlag == 0):
                bestDist = 10000
                chosen = 0
                for line in self.Tracks:
                    line = line[:-1].split(',')
                    bbox = line[0].split(' ')
                    middleXdist = (int(float(bbox[0])) + int(float(bbox[2]))) // 2 - x
                    middleYdist = (int(float(bbox[1])) + int(float(bbox[3]))) // 2 - y
                    dist = np.sqrt(middleXdist*middleXdist + middleYdist*middleYdist)

                    if dist < bestDist:
                        bestDist=dist
                        chosen = int(float(line[3]))
                self.chosenTrack = chosen
            
            if (self.markFlag == 1):
                self.posList.append([x,y])    
                self.pointList.append(Point(x,y))
                if (len(self.posList) > 2):
                    self.polyShape = Polygon([Contour(self.pointList,[],True)])

            if (self.markPerspectiveFlag == 1):
                self.perspectiveList.append([x,y])    

    def updateImages(self):

        cv2.setMouseCallback('Vid', self.draw_circle)
        
        self.frameID = self.frameID + self.skipFrames
        if (len(self.data) <= self.frameID):
            self.frameID = 0
        if (self.frameID < 0):
            self.frameID = len(self.data) - 1

        currentImage, self.Tracks = parseTXT(self.data[self.frameID])
        
        self.frame = cv2.imread(currentImage[:-3] + "jpg")

        # Draw Markings
        for pt in self.posList:               
            cv2.circle(self.frame,(pt[0],pt[1]),5,(0,0,255),1)

        for pt in self.perspectiveList:               
            cv2.circle(self.frame,(pt[0],pt[1]),5,(0,255,0),1)

        # draw marked area
        if len(self.posList) > 1:
                selectedColor = [0, 0, 0] * np.ones((len(self.frame), len(self.frame[0]), 3), np.uint8)
                imgSelectedColor = np.uint8(np.absolute(selectedColor))
                cv2.fillPoly(imgSelectedColor, [np.array(self.posList)], (0,0,255),8)
                self.frame = cv2.addWeighted(self.frame, 1, imgSelectedColor, 0.2, 0) 

        # draw perspective shape
        if len(self.perspectiveList) > 1:
                selectedColor = [0, 0, 0] * np.ones((len(self.frame), len(self.frame[0]), 3), np.uint8)
                imgSelectedColor = np.uint8(np.absolute(selectedColor))
                cv2.fillPoly(imgSelectedColor, [np.array(self.perspectiveList)], (0,255,0),8)
                self.frame = cv2.addWeighted(self.frame, 1, imgSelectedColor, 0.2, 0) 
    
        # Draw Tracks
        for line in self.Tracks:
            line = line[:-1].split(',')
            bbox = line[0].split(' ')
            startPoint = (int(float(bbox[0])), int(float(bbox[1])))
            endPoint = (int(float(bbox[2])), int(float(bbox[3])))
            boxPoly = Polygon([Contour([Point(int(float(bbox[0])), int(float(bbox[1]))), Point(int(float(bbox[0])), int(float(bbox[3]))), Point(int(float(bbox[2])), int(float(bbox[3]))), Point(int(float(bbox[2])), int(float(bbox[1])))],[],True)])
            
            # Sel bounding box color (Red if in Polygon)
            inter = compute(boxPoly, self.polyShape, OperationType.INTERSECTION)
            if inter==Polygon([]):
                color = (100*(int(float(line[2]))-1),255-100*(int(float(line[2]))-1),0)
            else:
                color = (0,0,255)


            if self.chosenTrack == int(float(line[3])):
                cv2.rectangle(self.frame, startPoint, endPoint, color, 3)
            else:
                cv2.rectangle(self.frame, startPoint, endPoint, color, 1)
            
            #a = cv2.perspectiveTransform(cv2.Point(startPoint), self.M)

            # draw track number
            cv2.putText(self.frame, line[3], (int(float(bbox[0])), int(float(bbox[1]))-10), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,255,0),1)
        if self.maxWidth == 0:
            cv2.imshow('Vid', self.frame)
        else:
            cv2.imshow('Vid', cv2.warpPerspective(self.frame, self.M, (self.frame.shape[0],self.frame.shape[1]) , flags=cv2.INTER_LINEAR))

        self.window.after(self.delay, self.updateImages)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, self.frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, self.frame)
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
# App(tkinter.Tk(), "Testing GUI", 0 ,"videoplayback.mp4") # for video
App(tkinter.Tk(), "Testing GUI", "more images" ,0) # for images