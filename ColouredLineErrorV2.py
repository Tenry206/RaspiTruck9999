import cv2
import numpy as np
import math
from collections import deque

class toilet:

    def __init__(self, frame_width = 640):
        self.frame_center = frame_width // 2
        
        # --- THE RED/BLUE SWAP FIX ---
        # Because the camera channels are swapped:
        # 1. Physical YELLOW looks CYAN (Hue ~80 to 105)
        self.lower_yellow = np.array([80, 180, 180])
        self.upper_yellow = np.array([120, 220, 220])
        
        # 2. Physical RED looks BLUE (Hue ~105 to 140)
        self.lower_red = np.array([40,40, 30])#105, 100, 50
        self.upper_red = np.array([90, 120, 50])#140, 255, 255

        # Matched the black line sensitivity
        self.colorThresh = 400 
        self.error_queue = deque(maxlen = 2)

    def preprocess(self, frame):
        h = frame.shape[0]
        roi_y = int(h * (1 - 0.75)) 
        roi = frame[roi_y:int(0.75*h), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create separate masks for the swapped colors
        mask_yellow_cyan = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red_blue = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        # Combine them into one priority mask
        mask_combined = cv2.bitwise_or(mask_yellow_cyan, mask_red_blue)

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

        # Print ALL raw color data BEFORE the threshold kills it
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