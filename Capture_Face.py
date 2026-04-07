import cv2
import os
import time
from Camera_backup import Camera 

def main():
    # Make sure the dataset folder exists
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        
    print("--- Face Dataset Generator ---")
    
    # 1. Enter your specific ID number
    face_id = input("Enter an integer ID for this person (e.g., 1 for Tenry, 2 for Randy): ")
    
    print("\nInitializing camera...")
    cam = Camera(resolution=(640, 480), fps=30)
    
    # Use the OpenCV Face Detector we downloaded yesterday to crop the faces
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    print("\n[INFO] Look at the camera and wait. Capturing 50 faces...")
    time.sleep(2)
    
    count = 0
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face so you know it sees you
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Crop the face out of the frame and save it to the dataset folder
                count += 1
                file_path = f"dataset/User.{face_id}.{count}.jpg"
                cv2.imwrite(file_path, gray[y:y+h, x:x+w])
                
                print(f"Captured Image {count}/50")
            
            cv2.imshow('Capturing Faces', frame)
            
            # Press 'q' to quit early, or auto-stop when we reach 50 images
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif count >= 50:
                break
                
            time.sleep(0.1) # Slight delay between photos to get different angles
            
    finally:
        print(f"\n[SUCCESS] Successfully captured {count} faces for ID {face_id}!")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()