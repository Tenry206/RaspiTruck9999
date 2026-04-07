import cv2
import numpy as np
from Camera import Camera
import math
from collections import deque

class toilet:

    def __init__(self, frame_width = 640):

        self.frame_center = frame_width // 2
        
        self.lower_yellow = np.array([40, 40, 30])
        self.upper_yellow = np.array([90, 120, 50])
        self.colorThresh = 8000
        self.error_queue = deque(maxlen = 2)

    def preprocess(self, frame):
        h = frame.shape[0]
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        thresh = cv2.inRange(self.hsv, self.lower_yellow, self.upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh
        
    def colored_error(self, frame):
        colorBool = False
        thresh = self.preprocess(frame)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        turn = None

        if not contours:
            return 0, False, 0

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return 0, False, 0

        cx = int(M["m10"] / M["m00"])
        error = cx - self.frame_center
        self.error_queue.append(error)
        error_smoothed = int(np.mean(self.error_queue))

        if area > self.colorThresh:
            colorBool = True


        return error_smoothed, colorBool, area
        