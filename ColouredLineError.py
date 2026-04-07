import cv2
import numpy as np
from Camera import Camera
import math

class ColouredLineError:

    def __init__(self, frame_width = 640):

        self.frame_center = frame_width // 2
        
        self.lower_hsv = np.array([105, 110, 120])
        self.upper_hsv = np.array([130, 160, 180])

    def preprocess(self, frame):
        h = frame.shape[0]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        thresh = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh
        
    def colored_error(self, frame):
        thresh = self.preprocess(frame)
        