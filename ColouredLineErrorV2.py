import cv2
import numpy as np
from collections import deque

class toilet:

    def __init__(self, frame_width = 640):
        self.frame_center = frame_width // 2
        
        # --- THE HSV CEILING FIX ---
        # Changed 220 to 255. Never cap Saturation or Value on bright tracks!
        self.lower_yellow = np.array([80, 100, 100]) 
        self.upper_yellow = np.array([120, 255, 255])
        
        self.lower_red = np.array([100, 140, 160])
        self.upper_red = np.array([140, 200, 200]) 

        # Sensitivity threshold
        self.colorThresh = 150 
        self.error_queue = deque(maxlen = 3) 
        
        # --- HARD STATE LOCK COUNTER ---
        self.color_persistence = 0
        self.last_known_error = 0

    def preprocess(self, frame):
        h = frame.shape[0]
        roi_y = int(h * (1 - 0.75)) 
        roi = frame[roi_y:int(0.75*h), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_yellow_cyan = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_red_blue = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        mask_combined = cv2.bitwise_or(mask_yellow_cyan, mask_red_blue)

        # Fast morphology just to clear out tiny static
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

        return thresh, hsv
        
    def colored_error(self, frame):
        thresh, hsv_roi = self.preprocess(frame)

        # 1. HIGH SPEED COUNT (Replaces cv2.findContours)
        area = cv2.countNonZero(thresh)

        # 2. RESTORED HSV DEBUG PRINT 
        # We use the raw 'thresh' mask to find the mathematical average color
        if area > 50: 
            mean_hsv = cv2.mean(hsv_roi, mask=thresh)
            print(f"RAW COLOR -> Pixels: {area} | Avg HSV: [H:{int(mean_hsv[0])}, S:{int(mean_hsv[1])}, V:{int(mean_hsv[2])}]")

        # 3. THE HARD STATE LOCK (Active Tracking)
        if area > self.colorThresh:
            self.color_persistence = 10 

            # Calculate Center of Mass of the ENTIRE mask
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - self.frame_center
                
                self.error_queue.append(error)
                error_smoothed = int(np.mean(self.error_queue))

                self.last_known_error = error_smoothed
                return error_smoothed, True

        # 4. WE ARE IN THE JUNCTION GAP (Coasting)
        elif self.color_persistence > 0:
            self.color_persistence -= 1
            print(f"COASTING OVER GAP... Persistence left: {self.color_persistence}")
            return self.last_known_error, True
                
        # 5. DEFAULT TO BLACK LINE
        return None, False