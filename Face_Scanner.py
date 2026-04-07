import cv2
import os
import urllib.request

class FaceScanner:
    def __init__(self):
        print("Initializing OpenCV Haar Cascade Face & Feature Detection...")
        
        # We scale the image down by 2 (640->320) to save CPU. 
        # We must multiply the coordinates by 2 later to draw them correctly!
        self.scale = 2 

        # The AI models we need for faces, eyes, and mouths (smiles)
        cascades = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'mouth': 'haarcascade_smile.xml'
        }
        
        self.models = {}
        
        # Download and load all models
        for key, xml_name in cascades.items():
            if not os.path.exists(xml_name):
                print(f"Downloading {xml_name}...")
                url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_name}"
                urllib.request.urlretrieve(url, xml_name)
            
            self.models[key] = cv2.CascadeClassifier(xml_name)

    def scan_for_face(self, frame):
        """
        Scans for faces, and then sub-scans for eyes and mouths.
        Returns: (Boolean face_found, List of face_data dictionaries)
        """
        # 1. Downscale to save massive CPU power
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Find the faces first
        faces = self.models['face'].detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        results = []
        
        for (x, y, w, h) in faces:
            # Create a dictionary to hold all the coordinates (Scaled up by x2)
            face_data = {
                'box': (x * self.scale, y * self.scale, w * self.scale, h * self.scale),
                'eyes': [],
                'mouths': []
            }
            
            # --- ROI SUB-SCANNING ---
            # Crop the image down to JUST the face to find the eyes/mouth
            roi_gray = gray[y:y+h, x:x+w]
            
            # Find Eyes inside the face
            eyes = self.models['eye'].detectMultiScale(roi_gray, 1.1, 15, minSize=(10, 10))
            for (ex, ey, ew, eh) in eyes:
                face_data['eyes'].append((
                    (x + ex) * self.scale, 
                    (y + ey) * self.scale, 
                    ew * self.scale, 
                    eh * self.scale
                ))
                
            # Find Mouth (Smile) inside the lower half of the face
            roi_lower_face = gray[y + h//2 : y + h, x : x + w]
            mouths = self.models['mouth'].detectMultiScale(roi_lower_face, 1.5, 15, minSize=(15, 15))
            for (mx, my, mw, mh) in mouths:
                face_data['mouths'].append((
                    (x + mx) * self.scale, 
                    (y + (h//2) + my) * self.scale, # Add the lower half offset back
                    mw * self.scale, 
                    mh * self.scale
                ))

            results.append(face_data)
            
        if len(results) > 0:
            return True, results
            
        return False, []