import numpy as np
import cv2

class SymbolDetector:
    def __init__(self):
        # 1. Initialize variables ONCE when the class is created
        print("Loading Symbol Templates...")
        self.templates = {
            'button': cv2.imread('symbols/button.png', 0),
            'fingerprint': cv2.imread('symbols/fingerprint.png', 0),
            'qr': cv2.imread('symbols/qr.png', 0),
            # Add the rest of your symbols here
        }
        
        self.orb = cv2.ORB_create(nfeatures=4000, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        self.RATIO = 0.8
        self.MIN_GOOD = 12
        self.MIN_INLIER = 15
        self.RANSAC_THRESH = 5.0
        
        # Pre-compute the keypoints for the templates
        self.templatesF = {}
        for name, img in self.templates.items():
            if img is not None:
                tK, tD = self.orb.detectAndCompute(img, None)
                self.templatesF[name] = {"img": img, "kps": tK, "des": tD}
            else:
                print(f"Warning: Failed to load {name}.png")

    def detect(self, frame):
        # 2. This function ONLY does the math for the current frame
        # NO WHILE LOOP HERE
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vK, vD = self.orb.detectAndCompute(frame_gray, None)
        
        if vD is None:
            return None # No features found in frame
            
        best_name = None
        best_inliers = 0

        for name, tpl in self.templatesF.items():
            tD = tpl["des"]
            tK = tpl["kps"]
            
            if tD is None or len(tD) < 2:
                continue

            knn = self.matcher.knnMatch(tD, vD, k=2)

            good = []
            for pair in knn:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.RATIO * n.distance:
                        good.append(m)

            matchesNum = len(good)
            inliers = 0
            
            if matchesNum >= self.MIN_GOOD:
                src_pts = np.float32([tK[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([vK[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                # Use try-except block just in case RANSAC fails mathematically
                try:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_THRESH)
                    if mask is not None:
                        inliers = int(mask.ravel().astype(bool).sum())
                except Exception as e:
                    pass

            # Update best match if this template has more inliers
            if inliers > best_inliers and inliers >= self.MIN_INLIER:
                best_inliers = inliers
                best_name = name

        return best_name