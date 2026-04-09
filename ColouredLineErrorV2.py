import cv2
import numpy as np
from collections import deque

class toilet:

    def __init__(self, frame_width=640):
        self.frame_center = frame_width // 2
        
        # --- THE RED/BLUE SWAP FIX ---
        self.lower_yellow = np.array([80, 100, 100]) 
        self.upper_yellow = np.array([120, 255, 255])
        
        self.lower_red = np.array([100, 100, 100])
        self.upper_red = np.array([140, 255, 255]) 

        self.colorThresh = 150 
        self.error_queue = deque(maxlen=3) 
        
        # --- STATE LOCK VARIABLES ---
        self.color_persistence = 0
        self.last_known_error = 0
        self.active_color = None # Tracks whether we are currently on Red or Yellow

    def preprocess(self, frame):
        h = frame.shape[0]
        roi_y = int(h * (1 - 0.75)) 
        roi = frame[roi_y:int(0.75*h), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # SEPARATE the masks!
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh_y = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        thresh_r = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        return thresh_y, thresh_r, hsv
        
    def colored_error(self, frame, black_area=0):
        thresh_y, thresh_r, hsv_roi = self.preprocess(frame)

        # Count pixels for both colors separately
        area_y = cv2.countNonZero(thresh_y)
        area_r = cv2.countNonZero(thresh_r)

        # 1. ACTIVE TRACKING
        # Check if either color is strong enough to track
        if area_r > self.colorThresh or area_y > self.colorThresh:
            
            # Find the dominant color
            if area_r > area_y:
                active_mask = thresh_r
                self.active_color = 'Red'
                self.color_persistence = 10 # HIGHER persistence for Red
            else:
                active_mask = thresh_y
                self.active_color = 'Yellow'
                self.color_persistence = 5  # LOWER persistence for Yellow

            # Calculate Center of Mass for the dominant color
            M = cv2.moments(active_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - self.frame_center
                
                self.error_queue.append(error)
                error_smoothed = int(np.mean(self.error_queue))

                self.last_known_error = error_smoothed
                return error_smoothed, True, self.active_color

        # 2. COASTING (We are in the gap)
        elif self.color_persistence > 0:
            
            # --- EARLY ABORT MECHANISM ---
            # If the black area is large (> 1000), it means we have crossed the gap 
            # and landed safely on the black track. Cancel persistence instantly!
            if black_area > 1000:
                #print(f"BLACK TRACK DETECTED! Aborting {self.active_color} coasting.")
                self.color_persistence = 0
                self.active_color = None
                return None, False, None

            # Otherwise, keep coasting
            self.color_persistence -= 1
            #print(f"COASTING ({self.active_color})... Persistence left: {self.color_persistence}")
            return self.last_known_error, True, self.active_color
                
        # 3. DEFAULT (No color, no persistence)
        self.active_color = None
        return None, False, None