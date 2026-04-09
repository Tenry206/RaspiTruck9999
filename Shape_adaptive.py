import cv2
import numpy as np
from Camera import Camera
import math


SHAPE_AREA_RANGE = (2500, 36000)

def build_shape_candidate_mask(blur_gray, blur_sat):
    """Return a loose adaptive mask; final classification is handled later."""
    adaptive_dark = cv2.adaptiveThreshold(
        blur_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12
    )
    adaptive_color = cv2.adaptiveThreshold(
        blur_sat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -3
    )

    candidate_mask = cv2.bitwise_or(adaptive_dark, adaptive_color)

    # Opening first helps suppress thin line noise before reconnecting shape regions.
    open_kernel = np.ones((5, 5), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, open_kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(blur_gray)

    min_area, max_area = SHAPE_AREA_RANGE
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        _, _, w, h = cv2.boundingRect(cnt)
        if w < 35 or h < 35:
            continue

        bbox_area = w * h
        if bbox_area == 0:
            continue

        extent = area / float(bbox_area)
        if extent < 0.25:
            continue

        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask


def is_line_like_contour(cnt):
    """Reject long thin contours that are more likely to be track lines than symbols."""
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    short_side = min(w, h)
    long_side = max(w, h)

    if short_side <= 0:
        return True

    rect_area = w * h
    area = cv2.contourArea(cnt)
    rect_fill = area / rect_area if rect_area > 0 else 0.0
    elongation = long_side / short_side

    return elongation > 4.5 and rect_fill < 0.65


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

    if verts in [4,5,6] and (20000 < A < 36000) and (0.95 < ar < 1.25):
        return 'warning', ar, A, P, C, verts
        
    # 2. QR CODE (Target: 6-8 corners, Area ~11k-17k, AR ~1.01-1.10)
    if verts in [6, 7, 8] and (9000 < A < 20000) and (0.90 < ar < 1.4) and (0.09 <C<0.33):
        return 'qr', ar, A, P, C, verts
        
    # 3. BUTTON (Target: 9-10 corners, Area ~17k-24k, AR ~1.19-1.33)
    if verts in [8, 9, 10] and (14000 < A < 28000) and (1.10 < ar < 1.45):
        return 'button', ar, A, P, C, verts
    
    if verts in [5, 6] and (12000 < A < 13500) and (0.5 < ar < 0.90):
        return 'button', ar, A, P, C, verts
    
    if verts in [6, 7] and (8000 < A < 11000) and (0.5 < ar < 0.1):
        return 'button', ar, A, P, C, verts
    
    if verts in [5,8,10,12] and (8000 < A < 13000) and (0.90 < ar < 1.20) and (0.17 < C < 0.30):
        return 'fingerprint', ar, A, P, C, verts
        
    # Part B (The Bottom Arch): 6 corners, Area ~2300, AR ~1.87, Circ ~0.15
    if verts == 6 and (1500 < A < 3500) and (1.38 < ar < 2.20) and (0.10 < C < 0.25):
        return 'fingerprint', ar, A, P, C, verts

    
    if verts == 4 :
        if 10000< A<17000 and 0.67<C<0.70:
            return 'Trapezium', ar, A, P, C, verts
        elif 10000<A<18500 and 0.68<C<0.8:
            return 'Diamond' ,ar, A, P, C, verts   
    elif verts == 6 :
        if 0.76<C<0.9 and 9000<A <12000 :
            return 'Semicircle', ar, A, P, C, verts
        elif 0.9<ar<1.2 and 4000<A<6000:
            return 'Arrow', ar,A, P, C, verts
    elif verts == 7:
        if A<7000 and 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
        elif 0.5<C<0.8:
            return '3/4 Circle', ar, A, P, C, verts
        elif 0.74<C<0.9:
            return 'Semicircle',ar, A, P, C, verts
        elif 8000<A<13500 and ar<1.09 and 0.12<C<0.18:
            return 'recycle', ar, A, P, C, verts
    elif verts == 8 :
        if 0.8<C<1.0:
            return 'Octagon' ,ar, A, P, C, verts
        elif 0.5<C<0.8:
            return '3/4 Circle', ar, A, P, C, verts
        elif A<7000 and 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
        elif 8000<A<13500 and 0.12<C<0.18:
            return 'recycle', ar, A, P, C, verts
    elif verts == 9:
        if 8000<A<13500 and 0.12<C<0.18:
            return 'recycle', ar, A, P, C, verts
        elif 8000<A<13500 and 0.05<C<0.1:
            return 'recycle', ar, A, P, C, verts
        elif A<7000 and 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
    elif verts == 10:
        if 0.25<C<0.30 and A<8000:
            return 'Star' ,ar, A, P, C, verts
        elif 0.9<ar<1.2 and 8000<A<13900 and 0.12<C<0.18:
            return 'recycle', ar, A, P, C, verts
    elif verts == 11  and 0.9<ar<1.2 and 8000<A<13900 and 0.12<C<0.18:
        return 'recycle', ar, A, P, C, verts
    elif verts == 12 and 0.5<C<0.7:
        return 'Cross' ,ar, A, P, C, verts
    
    
    return 'Noise', ar, A, P, C, verts

def process_shapes(frame):
    """Processes a frame to detect shapes and colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    blur_sat = cv2.GaussianBlur(saturation, (5, 5), 0)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # if background too bright just ignore
    '''
    if np.max(blurred) < 15:  # You may need to tune '50' between 30 and 80
        return [], np.zeros_like(blurred)
    '''
    _, mask_color = cv2.threshold(blur_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    '''
    #roi
    thresh[:, :120] = 0  
    thresh[:, 500:] = 0  
    # Optionally, ignore the very top of the frame too
    thresh[:50, :] = 0   
    '''
    adaptive_candidates = build_shape_candidate_mask(blur_gray, blur_sat)
    thresh = cv2.bitwise_and(mask_color, adaptive_candidates)

    # 5. Shape Glue
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Re-include your Centroid Linking logic here if needed for QR codes

    initial_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    

    MAX_DISTANCE = 150
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
        if is_line_like_contour(cnt):
            continue

        shape_label, ar, area, perim, circ, verts = detect_shape(cnt)
        direction = "None"
        #detected_color = "Unknown"

        #h_avg, s_avg, v_avg = 0, 0, 0
        
        if shape_label != 'Noise':
            # Color detection logic
            #mask = np.zeros(saturation.shape, np.uint8)
            #cv2.drawContours(mask, [cnt], -1, 255, -1)
            #mean_val = cv2.mean(hsv, mask=mask)
            #h_avg = int(mean_val[0])
            #s_avg = int(mean_val[1])
            #v_avg = int(mean_val[2])
            
            '''
            for color_name, (lower, upper) in color_ranges.items():
                if lower[0] <= h_avg <= upper[0]:
                    detected_color = color_name
                    break
            
            if shape_label == "Arrow" and detected_color == "Unknown":
                shape_label = 'Noise'
            '''
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
                    #else:direction = "Down" if dy > 0 else "Up"

        if shape_label != 'Noise':
            results.append({
                'label': shape_label,
                'color': None,#detected_color
                'direction': direction,
                'contour': cnt,
                'area': area,
                'circ': circ,   
                'verts': verts,
                'hsv': (0, 0, 0)
                #'hsv': (h_avg, s_avg, v_avg)
            })
    return results, thresh

def main():
    # 1. Initialize your custom Camera class
    print("Initializing Camera...")
    cam = Camera(resolution=(640, 480), fps=30)
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            original_frame = frame.copy()
            detected_shapes, thresh_mask = process_shapes(frame)

            for shape in detected_shapes:
                cnt = shape['contour']
                label = shape['label']
                color = shape['color']
                direction = shape['direction']
                area = shape['area']
                h, s, v = shape['hsv']

                # --- NEW: Extract C and Verts from the dictionary! ---
                C = shape['circ']
                verts = shape['verts']

                #--- temporary test
                # 1. Recalculate corners and AR for our debug print
                eps = 0.03 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                
                
                bx, by, bw, bh = cv2.boundingRect(approx)
                ar = bw / float(bh) if bh > 0 else 0
                
                # 2. Draw the outline (RED if unrecognized Noise, GREEN if successful Shape)
                outline_color = (0, 0, 255) if label == 'Noise' else (0, 255, 0)
                cv2.drawContours(frame, [approx], 0, outline_color, 3)
                
                # Draw a purple rectangle around the shape!
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
                
                # 3. Print stats DIRECTLY onto the camera window!
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    # Draw Text: Area, Corners (Verts), and AR hovering over the shape
                    cv2.putText(frame, f"Area: {area:.0f}", (cx - 50, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Corn: {verts}", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"AR: {ar:.2f}", (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Circ: {C:.2f}", (cx - 50, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"{label}", (cx - 50, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
                print(f"TUNING -> {label}: Area={area:.0f}, Corners={verts}, AR={ar:.2f}, Circ={C:.2f}")
            contour_debug = frame.copy()
            all_contours, _ = cv2.findContours(thresh_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_debug, all_contours, -1, (255, 0, 0), 2)

            raw_view = cv2.resize(original_frame, (640, 480))
            contour_view = cv2.resize(frame, (640, 480))
            debug_view = np.hstack((raw_view, contour_view))
            
            cv2.imshow("Shape Drawn Contours", contour_debug)
            cv2.imshow("Shape Threshold Mask", thresh_mask)

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
