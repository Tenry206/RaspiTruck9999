import cv2
import numpy as np
from Camera import Camera
import math

color_ranges = {
    'Green':  [(30, 80, 105), (60, 130, 125)],
    'Blue':   [(10, 235, 180), (20, 255, 200)],
    'Orange': [(100, 165, 170), (120, 180, 200)],
    'Red':    [(105, 110, 120), (130, 160, 180)]
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
        if 10000< A<17000 and 0.67<C<0.70:
            return 'Trapezium', ar, A, P, C, verts
        elif 10000<A<18500 and 0.68<C<0.8:
            return 'Diamond' ,ar, A, P, C, verts   
    elif verts == 6 and 9000<A <12000 and 0.74<C<0.8:
        if 0.76<C<0.9 :
            return 'Semicircle', ar, A, P, C, verts
        elif 0.2<C<0.26:
            return 'Arrow', ar,A, P, C, verts
    elif verts == 7:
        if 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
        elif 0.5<C<0.8:
            return '3/4 Circle', ar, A, P, C, verts
        elif 0.76<C<0.9:
            return 'Semicircle',ar, A, P, C, verts
    elif verts == 8 :
        if 0.8<C<1.0:
            return 'Octagon' ,ar, A, P, C, verts
        elif 0.5<C<0.8:
            return '3/4 Circle', ar, A, P, C, verts
        elif 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
    elif verts == 10 and 0.25<C<0.30 and A<8000:
        return 'Star' ,ar, A, P, C, verts
    elif verts == 12 and 0.5<C<0.7:
        return 'Cross' ,ar, A, P, C, verts
    
    
    return 'Noise', ar, A, P, C, verts

def process_shapes(frame):
    """Processes a frame to detect shapes and colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    blur_sat = cv2.GaussianBlur(saturation, (5, 5), 0)
    blur_gray = cv2. GaussianBlur(gray, (5,5),0)
    
    # if background too bright just ignore
    '''
    if np.max(blurred) < 15:  # You may need to tune '50' between 30 and 80
        return [], np.zeros_like(blurred)
    '''


    _, mask_color = cv2.threshold(blur_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_dark = cv2.threshold(blur_gray, 150, 255, cv2.THRESH_BINARY_INV)

    thresh = cv2.bitwise_or(mask_color, mask_dark)
    '''
    #roi
    thresh[:, :120] = 0  
    thresh[:, 500:] = 0  
    # Optionally, ignore the very top of the frame too
    thresh[:50, :] = 0   
    '''
    # --- UPGRADED STRATEGY: ADAPTIVE THIN LINE DETECTION ---
    
    # 1. Adaptive Thresholding: Perfect for finding thin lines on varying backgrounds!
    # It compares a pixel to its neighbors (21x21 block), pulling out the thin box.
    thin_line_mask = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # 2. Thicken the thin line slightly so the 4 corners connect into a solid square
    line_kernel = np.ones((11, 11), np.uint8)
    thin_line_mask = cv2.morphologyEx(thin_line_mask, cv2.MORPH_CLOSE, line_kernel)

    # --- NEW: TEMPORARY DEBUG WINDOWS ---
    # WARNING: Only run these via `python Shape.py` standalone!
    #cv2.imshow("DEBUG 1: Thin Line Mask", thin_line_mask)
    
    contour_debug_frame = frame.copy()
    # ------------------------------------

    # 3. Find Contours (Using RETR_TREE to find the box)
    box_contours, hierarchy = cv2.findContours(thin_line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw EVERY contour in green so you can visually check if the box is captured!
    cv2.drawContours(contour_debug_frame, box_contours, -1, (0, 255, 0), 1)
    #cv2.imshow("DEBUG 2: All Contours", contour_debug_frame)

    roi_mask = np.zeros_like(gray)
    box_found = False

    if hierarchy is not None:
        for i, c in enumerate(box_contours):
            area = cv2.contourArea(c)
            
            # Look for a medium-to-large square (the black bounding box)
            if 10000 < area < 30000: 
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                
                _, _, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                corners = len(approx)
                
                # If we found the Thin Black Box (approx 4 corners, square-ish)
                if corners >= 4 and 0.5 < ar < 1.5:
                    print(f">>> SUCCESS! Thin Box Found! Area: {area:.0f} AR:{ar:.2f}")
                    
                    # 4. Fill the exact shape of the box to act as our window
                    cv2.drawContours(roi_mask, [c], -1, 255, -1)
                    box_found = True
                    
                    # Draw a thick blue box on the main camera feed
                    cv2.drawContours(frame, [c], -1, (255, 0, 0), 4)
                    break 

    if box_found:
        # 5. Shrink the mask slightly inward so the black border itself 
        # doesn't interfere with your color detection inside
        shrink_kernel = np.ones((8, 8), np.uint8)
        roi_mask = cv2.erode(roi_mask, shrink_kernel, iterations=1)
        
        # 6. Apply the mask to your original color threshold
        thresh = cv2.bitwise_and(mask_color, roi_mask)
    else:
        # If no box is found, erase everything.
        thresh[:, :] = 0 
        
    # --- END UPGRADED STRATEGY ---
    # ------------------------------------------------

    # 5. Shape Glue
    kernel = np.ones((5,5), np.uint8)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)
    # Re-include your Centroid Linking logic here if needed for QR codes

    initial_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    

    MAX_DISTANCE = 100
    centers = []

    for cnt in initial_contours:
        if cv2.contourArea(cnt) >300:
            M = cv2.moments(cnt)
            if M['m00']!= 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append((cx,cy))
                
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            # Calculate straight-line distance between shape centers
            if math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1]) < MAX_DISTANCE:
                # Draw a thick white line to fuse the shapes
                cv2.line(thresh, centers[i], centers[j], 255, thickness=6)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("Linked Threshold Mask", thresh)

    results = []
    for cnt in contours:
        shape_label, ar, area, perim, circ, verts = detect_shape(cnt)
        direction = "None"
        detected_color = "Unknown"

        h_avg, s_avg, v_avg = 0, 0, 0
        
        if shape_label != 'Noise':
            # Color detection logic
            mask = np.zeros(saturation.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(hsv, mask=mask)
            h_avg = int(mean_val[0])
            s_avg = int(mean_val[1])
            v_avg = int(mean_val[2])
            
            
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
                'area': area,
                'hsv': (h_avg, s_avg, v_avg)
            })
    return results, thresh

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
            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Extract ONLY the Saturation channel (Index 1)
            # Colorful things are white, grayscale things (background) become black
            #saturation = hsv[:, :, 1]
            #blurred = cv2.GaussianBlur(saturation, (5, 5), 0)           
            
            # NOTE: If your shapes are black on a white background, use THRESH_BINARY_INV
            # If they are white/light on a dark background, use THRESH_BINARY
            #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            
            #initial_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            #Distance between seperate shape
            '''
            MAX_DISTANCE = 100
            centers = []

            for cnt in initial_contours:
                if cv2.contourArea(cnt) >300:
                    M = cv2.moments(cnt)
                    if M['m00']!= 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        centers.append((cx,cy))
                        
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    # Calculate straight-line distance between shape centers
                    if math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1]) < MAX_DISTANCE:
                        # Draw a thick white line to fuse the shapes
                        cv2.line(thresh, centers[i], centers[j], 255, thickness=6)
            '''
            #cv2.imshow("Linked Threshold Mask", thresh)
            
            # 4. Find contours
            #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_shapes, thresh_mask = process_shapes(frame)
            '''
            for shape in detected_shapes:
                
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
amsk
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
            print(h_avg, s_avg, v_avg)
            # 5. Display the live feed
            '''

            for shape in detected_shapes:
                cnt = shape['contour']
                label = shape['label']
                color = shape['color']
                direction = shape['direction']
                area = shape['area']
                h, s, v = shape['hsv']
                
                #--- temporary test
                # 1. Recalculate corners and AR for our debug print
                eps = 0.03 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                verts = len(approx)
                
                _, _, bw, bh = cv2.boundingRect(approx)
                ar = bw / float(bh) if bh > 0 else 0
                
                # 2. Draw the outline (RED if unrecognized Noise, GREEN if successful Shape)
                outline_color = (0, 0, 255) if label == 'Noise' else (0, 255, 0)
                cv2.drawContours(frame, [approx], 0, outline_color, 3)
                
                # 3. Print stats DIRECTLY onto the camera window!
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    # Draw Text: Area, Corners (Verts), and AR hovering over the shape
                    cv2.putText(frame, f"Area: {area:.0f}", (cx - 50, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Corn: {verts}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"AR: {ar:.2f}", (cx - 50, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Hue: {h}", (cx - 50, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"{label}", (cx - 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Print to terminal as well
                print(f"TUNING -> {label}: Area={area:.0f}, Corners={verts}, AR={ar:.2f}, Hue={h}")
                
            # 4. Show the temporary tuning window
            cv2.imshow("Shape Tuning Window", frame)
            '''
                # Only draw and print if it's a real shape
                if label != 'Noise':
                    # Draw the green outline
                    eps = 0.03 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
                    
                    # Print to terminal
                    if label == 'Arrow':
                        print(f"Detected {color} Arrow pointing {direction}")
                        print(f"HSV: H={h} S={s} V={v}")
                    else:
                        P = cv2.arcLength(cnt, True)
                        C = 4 * np.pi * area / (P**2) if P > 0 else 0
                        verts = len(approx)
                        print(f"{label}: Area={area:.0f}, Circ={C:.2f}, Verts={verts}")
                        
                
            #cv2.imshow("Shape Detection", frame)
            '''
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
