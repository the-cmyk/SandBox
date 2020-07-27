import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import glob
import numpy as np

def parseTXT(txtfile):
    file = open(txtfile, 'r') 
    return txtfile, file.readlines()


class App:
    def __init__(self, window, window_title, data_source=0, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.markFlag = 0
        self.posList = []
        self.frameID = 0
        self.chosenTrack = 0
        self.Tracks = []
        self.window.attributes("-topmost", True)
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
        self.btn_snapshot=tkinter.Button(self.nav, text="Snapshot",  command=self.snapshot)
        self.btn_snapshot.pack(side="left", expand=True)
        
        # Button that lets the user take a snapshot
        self.btn_next=tkinter.Button(self.nav, text="Next", command=self.nextImage)
        self.btn_next.pack(side="left", expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 5
        if (video_source == 0) and not (data_source == 0):
            self.data = glob.glob(data_source + "\\*.txt")
            self.updateImages()
        else:
            self.update()

        self.window.mainloop()

    def nextImage(self):
        self.frameID = self.frameID + 1

    def prevImage(self):
        self.frameID = self.frameID - 1
    
    def clearPoly(self):
        self.posList = []

    def markPoly(self):
        self.markFlag = 1
        self.btn_polygon.config(relief="sunken")

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def draw_circle(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.markFlag == 0:
                bestDist = 10000
                chosen = 0
                for line in self.Tracks:
                    line = line[:-1].split(',')
                    bbox = line[0].split(' ')
                    middleXdist = (int(bbox[0]) + int(bbox[2]) // 2) - x
                    middleYdist = (int(bbox[1]) + int(bbox[3]) // 2) - y
                    dist = np.sqrt(middleXdist*middleXdist + middleYdist*middleYdist)
                    print("middleXdist " + str(middleXdist))
                    print("middleYdist " + str(middleYdist))
                    print("dist " + str(dist))
                    if dist < bestDist:
                        bestDist=dist
                        chosen = int(line[3])
                self.chosenTrack = chosen
            
            if (len(self.posList) < 4) and (self.markFlag == 1):
                self.posList.append([x,y])
            if (len(self.posList) == 4) and (self.markFlag == 1):
                self.markFlag = 0
                self.btn_polygon.config(relief="raised")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.namedWindow('Vid')
            cv2.setMouseCallback('Vid',self.draw_circle)
            for pt in self.posList:
                cv2.circle(frame,(pt[0],pt[1]),5,(255,0,0),2)
            if len(self.posList) > 1:
                cv2.line(frame,(self.posList[0][0],self.posList[0][1]), (self.posList[1][0], self.posList[1][1]),(0,0,255),1)
            if len(self.posList) > 3:
                cv2.line(frame,(self.posList[2][0],self.posList[2][1]), (self.posList[3][0], self.posList[3][1]),(0,0,255),1)
            cv2.imshow("Vid", frame)
            
        self.window.after(self.delay, self.update)

    def updateImages(self):
        cv2.namedWindow('Vid')
        cv2.setMouseCallback('Vid',self.draw_circle)
        
        if (len(self.data)<=self.frameID):
            self.frameID = 0
        if (self.frameID < 0):
            self.frameID = len(self.data) - 1

        currentImage, self.Tracks = parseTXT(self.data[self.frameID])
        
        frame = cv2.imread(currentImage[:-3] + "jpg")

        # Draw Poligon
        for pt in self.posList:
            cv2.circle(frame,(pt[0],pt[1]),5,(255,0,0),2)
        if len(self.posList) > 1:
            cv2.line(frame,(self.posList[0][0],self.posList[0][1]), (self.posList[1][0], self.posList[1][1]),(0,0,255), 1)
        if len(self.posList) > 3:
            cv2.line(frame,(self.posList[2][0],self.posList[2][1]), (self.posList[3][0], self.posList[3][1]),(0,0,255), 1)

        # Draw Tracks
        for line in self.Tracks:
            line = line[:-1].split(',')
            bbox = line[0].split(' ')
            startPoint = (int(bbox[0]), int(bbox[1]))
            endPoint = (int(bbox[0]) + int(bbox[2]), int(bbox[1])+int(bbox[3]))
            if self.chosenTrack == int(line[3]):
                cv2.rectangle(frame, startPoint, endPoint, (100*(int(line[2])-1),255-100*(int(line[2])-1),0), 3)
            else:
                cv2.rectangle(frame, startPoint, endPoint, (100*(int(line[2])-1),255-100*(int(line[2])-1),0), 1)
            
            # draw score
            cv2.putText(frame, "score: " + line[1], (int(bbox[0]), int(bbox[1])+30), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,0,255),1)

            # draw class
            cv2.putText(frame, "class: " + line[2], (int(bbox[0]), int(bbox[1])+20), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,0,255),1)
            
            # draw track
            cv2.putText(frame, "track: " + line[3], (int(bbox[0]), int(bbox[1])+10), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,0,255),1)

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
App(tkinter.Tk(), "Testing GUI", "images" ,0) # for images