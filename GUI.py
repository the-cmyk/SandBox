import tkinter
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
        self.markFlag = 0   # Mark intereset area
        self.posList = []   # List of marked positions
        self.pointList = [] # List of marked position (type Point)
        self.frameID = 0    # Current frameID
        self.chosenTrack = 0# Chosen track ID
        self.Tracks = []    # list of Tracks in image
        self.skipFrames = 0
        self.polyShape = Polygon([Contour([Point(-1,-1), Point(-2,-2), Point(-2,-1)],[],True)])
        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.attributes("-toolwindow", 1)
        self.window.attributes("-alpha", 0.7)
        
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        self.poly = tkinter.Label(window, text="top pane")
        self.poly.pack(side="top", expand=True)
        # Button that lets the user take a snapshot
        self.btn_polygon=tkinter.Button(self.poly, text="Mark Poly", command=self.markPoly)
        self.btn_polygon.pack(side="left", expand=True)

        # Button that lets the user take a snapshot
        self.btn_clrPly=tkinter.Button(self.poly, text="Clear Poly", command=self.clearPoly)
        self.btn_clrPly.pack(side="left", expand=True)

        self.nav = tkinter.Label(window, text="bottom pane")
        self.nav.pack(side="bottom", expand=True)
        # Button that lets the user take a snapshot
        self.btn_prev=tkinter.Button(self.nav, text="Prev", command=self.prevImage)
        self.btn_prev.pack(side="left", expand=True)

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(self.nav, text="Play/Stop",  command=self.play)
        self.btn_snapshot.pack(side="left", expand=True)
        
        # Button that lets the user take a snapshot
        self.btn_next=tkinter.Button(self.nav, text="Next", command=self.nextImage)
        self.btn_next.pack(side="left", expand=True)

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

    def markPoly(self):
        self.markFlag = 1

    def play(self):
        # Get a frame from the video source
        if self.skipFrames == 0:
            self.skipFrames = 1
        else:
            self.skipFrames = 0

    def draw_circle(self, event,x,y,flags,param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.markFlag = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.markFlag == 0:
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


    def updateImages(self):
        cv2.namedWindow('Vid', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Vid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Vid', self.draw_circle)
        
        self.frameID = self.frameID + self.skipFrames
        if (len(self.data) <= self.frameID):
            self.frameID = 0
        if (self.frameID < 0):
            self.frameID = len(self.data) - 1

        currentImage, self.Tracks = parseTXT(self.data[self.frameID])
        
        frame = cv2.imread(currentImage[:-3] + "jpg")

        # Draw Poligon
        for pt in self.posList:               
            cv2.circle(frame,(pt[0],pt[1]),5,(0,0,255),1)

        if len(self.posList) > 1:
                selectedColor = [0, 0, 0] * np.ones((len(frame), len(frame[0]), 3), np.uint8)
                imgSelectedColor = np.uint8(np.absolute(selectedColor))
                cv2.fillPoly(imgSelectedColor, [np.array(self.posList)], (0,0,255),8)
                frame = cv2.addWeighted(frame, 1, imgSelectedColor, 0.4, 0) 
    
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
                cv2.rectangle(frame, startPoint, endPoint, color, 3)
            else:
                cv2.rectangle(frame, startPoint, endPoint, color, 1)
            
            # draw track
            cv2.putText(frame, line[3], (int(float(bbox[0])), int(float(bbox[1]))-10), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,255,0),1)

        cv2.imshow('Vid', frame)
        
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
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
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