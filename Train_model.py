import cv2
import numpy as np
import os
from PIL import Image 

def getImagesAndLabels(path):
    print("Reading images from dataset folder...")
    # Get the paths to all the files in the dataset folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    faceSamples = []
    ids = []
    
    # Loop through every image
    for imagePath in imagePaths:
        # Convert it to a grayscale numpy array
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        
        # Extract the ID from the filename (e.g., User.1.25.jpg -> extracts the '1')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # We assume the whole cropped image is the face
        faceSamples.append(img_numpy)
        ids.append(id)
        
    return faceSamples, ids

def main():
    print("--- OpenCV LBPH Face Trainer ---")
    path = 'dataset'
    
    # Initialize the LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Get the faces and IDs
    faces, ids = getImagesAndLabels(path)
    
    print(f"[INFO] Training model with {len(np.unique(ids))} unique ID(s)... Please wait.")
    
    # Train the model!
    recognizer.train(faces, np.array(ids))
    
    # Save the compiled model into your main project folder
    recognizer.write('trainer.yml')
    
    print(f"\n[SUCCESS] Model trained and saved as 'trainer.yml'!")

if __name__ == "__main__":
    main()