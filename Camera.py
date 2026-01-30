import numpy as np
from picamera2 import Picamera2,Preview
import cv2
import time
from collections import deque

#camera class
class Camera:
    def __init__(self, 
                 resolution =(640,480), 
                 fps = 60,
                 roi_ratio = 0.75,
                 format = "RGB888"):
    
        # Frame duration (us)
        frame_duration = int(1e6 / fps)
        self.width, self.height = resolution
        self.frame_center = self.width // 2
        self.roi_ratio = roi_ratio
        #create object for Picamera2
        self.picam2 = Picamera2()

        #Configuration
        #main = image output in 640*480 resolution, in RGB fromat
        #controls = sensor timing of 16666 leads to 60fps
        config = self.picam2.create_preview_configuration(
            main = {"size": resolution, "format": format},
            controls = {"FrameDurationLimits": (frame_duration,frame_duration)}  #60fps
        )
        
        self.picam2.configure(config)

        # Position History Queue
        # Include six 0 digit to make sure the array is not empty
        # Keep only the latest 6 measurements, older ones are automatically discarded
        self.error_queue = deque([0]*6,maxlen = 6)

        self.picam2.start()

        # ------- Camera ------

        # Capture the full-resolution frame
        # Converts RGB -> BGR
        # Returns full immmage
    def read(self):
        frame = self.picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #Stop the camera
    def stop(self):
        self.picam2.stop()
        cv2.destroyAllWindows()


    # ------ Processing

    def preprocess(self, frame):
        # ROI + grayscale + blur + binary

        #Tuple describing the array dimensions
        #frame.shape -> (height, width, channels)
        #The height of the camera frame in pixels.
        h = frame.shape[0]
        
        # Find the starting row for the ROI
        roi_y = int(h*(1-self.roi_ratio))

        # Crops the bottom portion of the frame
        roi = frame[roi_y:int(0.75*h), :]

        # grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        #gray = cv2.equalizeHist(gray)

        #Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5,5),0)


        #Convert Grayscale image into binary
        #invert black and white
        #return binary
        
        _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

        return thresh, roi_y
    
    
    # ------ Core Logic ------

    def get_error(self, frame):
        thresh, roi_y = self.preprocess(frame)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        turn = None

        if not contours:
            return None, thresh, None, turn,0

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        """
        if area < 300:
            return None, thresh, None, turn
        """

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, thresh, None, turn, 0

        cx = int(M["m10"] / M["m00"])
        error = cx - self.frame_center

        self.error_queue.append(error)
        error_smoothed = int(np.mean(self.error_queue))

        # ---- Turn detection ----
        if area > 30000:
            if cx < self.frame_center * 0.75: #240
                turn = "LEFT"
            elif cx > self.frame_center * 1.25: #432
                turn = "RIGHT"

        return error_smoothed, thresh, cx, turn, area
    
    # ------ display ------
    def display(self, frame, cx = None):
        # Show original image with center and line centroid

        cv2.line(frame,
                    (self.frame_center, 0),
                    (self.frame_center, frame.shape[0]),
                    (0,255,0),2)
        if cx is not None:
            cv2.circle (frame,
                        (cx, frame.shape[0] - int(frame.shape[0] * self.roi_ratio//2)),
                        6, (0,0,255),-1)
        cv2.imshow("Camera View", frame)

    def display_draw(self, thresh):
        # Contour-cleaned white-line mask
        # 0 = background, 255 = object
        # output list of contours

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Initalize a blank black image
        mask = np.zeros_like(thresh)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask,[largest],-1,255,cv2.FILLED)

        cv2.imshow("Binary Line Mask",mask)
        return mask

