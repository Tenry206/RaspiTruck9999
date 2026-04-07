import cv2
import numpy as np
import math
from collections import deque

class toilet:

    def __init__(self, frame_width = 640):
        self.frame_center = frame_width // 2
        
        # 1. Yellow HSV Range
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        
        # 2. Red HSV Range (Red requires two ranges because it wraps around 0 and 180)
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # Lowered threshold so it catches the thin color lines easier
        self.colorThresh = 3000 
        self.error_queue = deque(maxlen = 2)

    def preprocess(self, frame):
        # MUST match the exact ROI cropping from Camera.py!
        h = frame.shape[0]
        roi_y = int(h * (1 - 0.75)) 
        roi = frame[roi_y:int(0.75*h), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create separate masks for the colors
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        
        # Combine both halves of Red
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combine Red and Yellow into one master "Priority Mask"
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

        if area > self.colorThresh:
            
            # --- EXACT HSV DEBUG PRINT ---
            # This mathematically averages the color inside the line so you can tune perfectly!
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            mean_hsv = cv2.mean(hsv_roi, mask=mask)
            print(f"DEBUG COLOR -> Area: {area:.0f} | Avg HSV: [H:{int(mean_hsv[0])}, S:{int(mean_hsv[1])}, V:{int(mean_hsv[2])}]")
            # -----------------------------

            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None, False

            cx = int(M["m10"] / M["m00"])
            error = cx - self.frame_center
            
            self.error_queue.append(error)
            error_smoothed = int(np.mean(self.error_queue))

            return error_smoothed, True
            
        return None, False