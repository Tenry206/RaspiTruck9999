import cv2
import os
import time
import numpy as np
from Camera import Camera

# 1. Setup Folder Structure
# Create a main folder, then subfolders for each class
DATASET_DIR = "training_data"
# Change this name for each symbol you are teaching (e.g., 'QR_Code', 'Arrow', 'Star')
CURRENT_CLASS = "QR_Code" 

save_path = os.path.join(DATASET_DIR, CURRENT_CLASS)
if not os.path.exists(save_path):
    os.makedirs(save_path)

def main():
    print(f"Initializing Collection for class: {CURRENT_CLASS}")
    cam = Camera(resolution=(640, 480), fps=30)
    count = 0
    
    print("\nControls:")
    print("  's' - Capture a burst of 20 images")
    print("  'q' - Quit and close")

    try:
        while True:
            frame = cam.read()
            
            # 2. Replicate your Shape.py preprocessing
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            blurred = cv2.GaussianBlur(saturation, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. Find the Symbol ROI
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            display_frame = frame.copy()
            

            cv2.imshow("Dataset Collector - Press 'S' for Burst", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 4. Burst Capture Logic
            if key == ord('s'):
                
                print(f"Capturing burst for {CURRENT_CLASS}...")
                for i in range(20):
                    # Capture fresh frame for slight variations in movement/lighting
                    temp_frame = cam.read()
                    # We use the same bounding box for the burst speed

                    
                    timestamp = int(time.time() * 1000)
                    file_name = f"{save_path}/{CURRENT_CLASS}_{timestamp}_{i}.jpg"
                    cv2.imwrite(file_name, temp_frame)
                    time.sleep(0.05) # Small delay to allow for natural movement
                    
                    count += 20
                    print(f"Total images for {CURRENT_CLASS}: {count}")
                else:
                    print("No object detected to capture!")

            elif key == ord('q'):
                break

    finally:
        cam.stop()
        print(f"Finished! {count} images saved in {save_path}")

if __name__ == "__main__":
    main()