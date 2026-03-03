import cv2
import numpy as np
from Camera import Camera
import math



def detect_shape(cnt):
    A = cv2.contourArea(cnt)
    if A <500:
        return 'noise', 0.0, A, 0.0, 0.0, 0.0
    
    P = cv2.arcLength(cnt, True)
    if P == 0:
        return 'noise', 0.0, A,P,0.0, 0.0
    
    C = 4*np.pi * A / (P**2)

    # Polygon approximation
    esp = 0.03*P
    approx = cv2.approxPolyDP(cnt, esp, True)
    verts = len(approx)
    
    # Bounding box -> aspect ratio
    _,_,w,h = cv2.boundingRect(approx)
    ar = w/float(h)

    if verts == 4 and 0.6<C<0.8:
        if  A<13000:
            return 'Trapisium', ar, A, P, C, verts
        elif A>1300:
            return 'Diamond' ,ar, A, P, C, verts
        
    elif verts == 6 and 0.76<C<0.9 :
        return 'Semicircle', ar, A, P, C, verts
    elif verts == 8 :
        if 0.8<C<1.0:
            return 'Octagon' ,ar, A, P, C, verts
        elif 0.5<C<0.8:
            return '3/4 Circle', ar, A, P, C, verts
    elif verts == 10 and 0.28<C<0.30:
        return 'Star' ,ar, A, P, C, verts
    elif verts == 12 and 0.5<C<0.7:
        return 'Cross' ,ar, A, P, C, verts
    
    
    return 'Noise', ar, A, P, C, verts

def main():
    # 1. Initialize your custom Camera class
    print("Initializing Camera...")
    cam = Camera(resolution=(640, 480), fps=30)
    
    try:
        while True:
            # 2. Grab the full, uncropped frame
            frame = cam.read()
            
            # 3. Process the full frame for shapes
            # Convert to HSV color space instead of Grayscale
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Extract ONLY the Saturation channel (Index 1)
            # Colorful things are white, grayscale things (background) become black
            saturation = hsv[:, :, 1]
            blurred = cv2.GaussianBlur(saturation, (5, 5), 0)           
            
            # NOTE: If your shapes are black on a white background, use THRESH_BINARY_INV
            # If they are white/light on a dark background, use THRESH_BINARY
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow("Threshold Mask", thresh)

            # 4. Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                shape_label, ar, A, P, C, verts = detect_shape(cnt)
                
                if shape_label != 'noise':
                    # Draw the bounding polygon
                    eps = 0.03 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
                    
                    print(f"{shape_label}: AR={ar:.2f}, Area={A:.0f}, Perim={P:.0f}, Circ={C:.2f}, Verts={verts:.2f}")
            
            # 5. Display the live feed
            cv2.imshow("Shape Detection", frame)
            
            # 6. Exit condition (Press 'q' to quit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # 7. Safely shut down the camera using your stop() method
        print("Stopping camera...")
        cam.stop()

if __name__ == "__main__":
    main()
