import cv2
import numpy as np
from Camera import Camera
import math

color_ranges = {
    'Green':  [(30, 80, 150), (60, 120, 170)],
    'Blue':   [(10, 235, 180), (20, 255, 200)],
    'Orange': [(90, 180, 200), (110, 200, 220)],
    'Red':    [(110, 130, 180), (125, 150, 200)]
}

def detect_shape(cnt):
    A = cv2.contourArea(cnt)
    if A <500:
        return 'Noise', 0.0, A, 0.0, 0.0, 0.0
    
    P = cv2.arcLength(cnt, True)
    if P == 0:
        return 'Noise', 0.0, A,P,0.0, 0.0
    
    C = 4*np.pi * A / (P**2)

    # Polygon approximation
    esp = 0.03*P
    approx = cv2.approxPolyDP(cnt, esp, True)
    verts = len(approx)
    
    # Bounding box -> aspect ratio
    _,_,w,h = cv2.boundingRect(approx)
    ar = w/float(h)

    if verts == 4 :
        if  A<11500 and 0.6<C<0.7:
            return 'Trapezium', ar, A, P, C, verts
        elif 0.685<C<0.8:
            return 'Diamond' ,ar, A, P, C, verts
        
    elif verts == 6:
        if 0.76<C<0.9 :
            return 'Semicircle', ar, A, P, C, verts
        elif 0.2<C<0.26:
            return 'Arrow', ar,A, P, C, verts
    elif verts == 7 and 0.2<C<0.26 :
        return 'Arrow', ar, A, P, C, verts
    elif verts == 8 :
        if 0.8<C<1.0:
            return 'Octagon' ,ar, A, P, C, verts
        elif 0.5<C<0.8:
            return '3/4 Circle', ar, A, P, C, verts
        elif 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
    elif verts == 10 and 0.28<C<0.30:
        return 'Star' ,ar, A, P, C, verts
    elif verts == 12 and 0.5<C<0.7:
        return 'Cross' ,ar, A, P, C, verts
    
    
    return 'Noise', ar, A, P, C, verts

def process_shapes(frame):
    """Processes a frame to detect shapes and colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)           
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Re-include your Centroid Linking logic here if needed for QR codes

    initial_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in initial_contours:
        if cv2.contourArea(cnt) > 300:
            M = cv2.moments(cnt)
            if M['m00']!= 0:
                centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            if math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1]) < 80:
                cv2.line(thresh, centers[i], centers[j], 255, thickness=6)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        shape_label, ar, area, perim, circ, verts = detect_shape(cnt)
        direction = "None"
        detected_color = "Unknown"

        if shape_label != 'Noise':
            # Color detection logic
            mask = np.zeros(saturation.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(hsv, mask=mask)
            h_avg = int(mean_val[0])
            
            
            for color_name, (lower, upper) in color_ranges.items():
                if lower[0] <= h_avg <= upper[0]:
                    detected_color = color_name
                    break
            
            if shape_label == "Arrow" and detected_color == "Unknown":
                shape_label = 'Noise'

            if shape_label == "Arrow":
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00']) 
                    cy = int(M['m01']/M['m00']) 
                    max_dist = 0
                    tip_x, tip_y = cx, cy

                    for pt in cnt:
                        px, py = pt[0]
                        dist = (px-cx)**2 + (py-cy)**2
                        if dist>max_dist:
                            max_dist = dist
                            tip_x, tip_y = px,py
                    dx = tip_x - cx
                    dy = tip_y - cy
                    if abs(dx) > abs(dy):
                        direction = 'Right' if dx>0 else "Left"
                    else:direction = "Down" if dy > 0 else "Up"


        if shape_label != 'Noise' or (shape_label == 'Noise' and area > 1000):
            results.append({
                'label': shape_label,
                'color': detected_color,
                'direction': direction,
                'contour': cnt,
                'area': area
            })
    return results, thresh
'''
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


            
            initial_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            #Distance between seperate shape
            MAX_DISTANCE = 100
            
            cv2.imshow("Linked Threshold Mask", thresh)
            
            # 4. Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                shape_label, ar, A, P, C, verts = detect_shape(cnt)
                
                mask = np.zeros(saturation.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1,255,-1)
                mean_val = cv2.mean(hsv, mask=mask)
                h_avg, s_avg, v_avg = int(mean_val[0]), int(mean_val[1]), int(mean_val[2])
                
                detected_color = "Unknown"
                for color_name, (lower, upper) in color_ranges.items():
                    if lower[0] <= h_avg <= upper[0]:
                        detected_color = color_name
                        break

                if shape_label != 'Noise':
                    # Draw the bounding polygon
                    eps = 0.03 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
                
                if shape_label == 'Arrow':
                    if detected_color == 'Red':
                        print("Down Arrow")
                    elif detected_color == 'Green':
                        print("Up Arrow")
                    elif detected_color == 'Blue':
                        print("Right Arrow")
                    elif detected_color == 'Orange':
                        print("Left Arrow")
                    else:
                        print("Noise")
                else:
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
'''