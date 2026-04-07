import cv2
import numpy as np
from collections import deque

class toilet:

    def __init__(self, frame_width = 640):
        self.frame_center = frame_width // 2
        
        # --- THE RED/BLUE SWAP FIX ---
        self.lower_yellow = np.array([80, 180, 180])
        self.upper_yellow = np.array([120, 220, 220])
        
        self.lower_red = np.array([100, 140, 160])
        self.upper_red = np.array([140, 200, 200])

        # Lowered threshold to pick up the faintest traces of color
        self.colorThresh = 150 
        self.error_queue = deque(maxlen = 3) # Increased smoothing slightly
        
        # --- HARD STATE LOCK COUNTER ---
        # Increased to 10 iterations (0.2 seconds) to easily coast over the black junction
        self.color_persistence = 0
        self.last_known_error = 0

    def preprocess(self, frame):
        h = frame.shape[0]
        # To make it FASTER, we only look at the very bottom of the screen (the immediate track)
        roi_y = int(h * (1 - 0.75)) 
        roi = frame[roi_y:int(0.75*h), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_yellow_cyan = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red_blue = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        mask_combined = cv2.bitwise_or(mask_yellow_cyan, mask_red_blue)

        # REDUCED Morphology: This makes the code run significantly faster!
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

        return thresh, hsv
        
    def colored_error(self, frame):
        thresh, hsv_roi = self.preprocess(frame)

        # 1. NEW STRATEGY: Count all colored pixels instantly!
        # This is 10x faster than cv2.findContours
        area = cv2.countNonZero(thresh)

        if area > 50: 
            # A lighter debug print that won't slow down the terminal as much
            print(f"RAW COLOR PIXELS -> Count: {area}")

        # 2. THE HARD STATE LOCK
        if area > self.colorThresh:
            # We see a solid color line! Lock the mode for 10 frames
            self.color_persistence = 8

            # Calculate Center of Mass of the ENTIRE mask
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - self.frame_center
                
                self.error_queue.append(error)
                error_smoothed = int(np.mean(self.error_queue))

                self.last_known_error = error_smoothed
                return error_smoothed, True

        # 3. WE ARE IN THE JUNCTION GAP
        elif self.color_persistence > 0:
            # The color vanished, but we are locked into Color Mode!
            # Coast blindly using the last known angle until we cross the gap
            self.color_persistence -= 1
            print(f"COASTING OVER GAP... Persistence left: {self.color_persistence}")
            return self.last_known_error, True
                
        # 4. DEFAULT TO BLACK LINE
        return None, False