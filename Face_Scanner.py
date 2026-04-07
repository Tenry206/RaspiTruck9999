import cv2
import os
import urllib.request

class FaceScanner:
    def __init__(self):
        print("Initializing OpenCV Haar Cascade Face Detection...")
        
        # The specific AI model file we need
        xml_name = 'haarcascade_frontalface_default.xml'
        
        # If the file isn't in your robot folder yet, download it automatically!
        if not os.path.exists(xml_name):
            print(f"Downloading {xml_name} to project folder...")
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_name}"
            urllib.request.urlretrieve(url, xml_name)
            print("Download complete!")
            
        # Load the AI model directly from your local project folder
        self.face_cascade = cv2.CascadeClassifier(xml_name)
        
        if self.face_cascade.empty():
            print("ERROR: Could not load the Haar Cascade xml file!")

    def scan_for_face(self, frame):
        """
        Takes a BGR camera frame, converts to grayscale, 
        and returns True if a face is detected.
        """
        # 1. Downscale to save CPU power
        small_frame = cv2.resize(frame, (320, 240))
        
        # 2. Haar Cascades require Grayscale images!
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 3. Scan the image for faces
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # 4. If the array has anything in it, we found a face
        if len(faces) > 0:
            return True
            
        return False