import cv2
import os
import urllib.request

class FaceScanner:
    def __init__(self):
        print("Initializing OpenCV LBPH Face Recognizer...")
        
        # 1. Load the Haar Cascade for Face Detection
        xml_name = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(xml_name):
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_name}"
            urllib.request.urlretrieve(url, xml_name)
        self.face_cascade = cv2.CascadeClassifier(xml_name)
        
        # 2. Load the LBPH Face Recognizer (The Brain)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists('trainer.yml'):
            self.recognizer.read('trainer.yml')
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("WARNING: trainer.yml not found. Recognition will not work!")

        # 3. --- SET YOUR NAMES HERE ---
        # ID 0 is empty. ID 1 is the first name, ID 2 is the second name, etc.
        self.names = ['None', 'Tenry', 'Partner Name'] 

    def scan_for_face(self, frame):
        # We need grayscale for LBPH Recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        results = []
        
        for (x, y, w, h) in faces:
            name = "Unknown"
            confidence_text = ""
            
            # If the brain is loaded, try to recognize the cropped face
            if self.model_loaded:
                id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                
                # In LBPH, 0 confidence is a perfect match. 
                # Anything under 80 is usually a good, recognizable match.
                if confidence < 80:
                    try:
                        name = self.names[id]
                    except IndexError:
                        name = f"ID: {id}"
                    confidence_text = f" {round(100 - confidence)}%"
                else:
                    name = "Unknown Person"
                    confidence_text = f" {round(100 - confidence)}%"
            
            results.append({
                'box': (x, y, w, h),
                'name': name,
                'confidence': confidence_text
            })
            
        if len(results) > 0:
            return True, results
            
        return False, []