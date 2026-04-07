import cv2
import numpy as np
import math
from collections import deque

class toilet:

    def __init__(self, frame_width = 640):
        self.frame_center = frame_width // 2
        
        # Widened the ranges slightly to forgive room shadows / glare
        self.lower_yellow = np.array([15, 80, 80])#85,100,205
        self.upper_yellow = np.array([45, 255, 255])#jayden 105,255,255
        
        self.lower_red1 = np.array([0, 100, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # FIX 1: Matched the black line sensitivity! (Lowered from 3000 to 400)
        self.colorThresh = 400 
        self.error_queue = deque(maxlen = 2)

    def preprocess(self, frame):
        h = frame.shape[0]
        roi_y = int(h * (1 - 0.75)) 
        roi = frame[roi_y:int(0.75*h), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_combined = cv2.bitwise_or(mask_yellow, mask_red)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh, hsv
        
    def colored_error(self, frame):
        thresh, hsv_roi = self.preprocess(frame)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # FIX 2: Print ALL raw color data BEFORE the threshold kills it!
        if area > 50: 
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            mean_hsv = cv2.mean(hsv_roi, mask=mask)
            print(f"RAW COLOR -> Area: {area:.0f} | Avg HSV: [H:{int(mean_hsv[0])}, S:{int(mean_hsv[1])}, V:{int(mean_hsv[2])}]")

        if area > self.colorThresh:
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None, False

            cx = int(M["m10"] / M["m00"])
            error = cx - self.frame_center
            
            self.error_queue.append(error)
            error_smoothed = int(np.mean(self.error_queue))

            return error_smoothed, True
            
        return None, False