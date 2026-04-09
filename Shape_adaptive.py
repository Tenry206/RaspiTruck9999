import cv2
import numpy as np
from Camera import Camera
import math


SHAPE_AREA_RANGE = (2500, 33000)


def is_shape_candidate(cnt):
    """Broad filter so broken contours can survive until final shape classification."""
    area = cv2.contourArea(cnt)
    if area < SHAPE_AREA_RANGE[0] or area > SHAPE_AREA_RANGE[1]:
        return False

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False

    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    verts = len(approx)
    if verts < 4 or verts > 14:
        return False

    _, _, w, h = cv2.boundingRect(cnt)
    if w < 35 or h < 35:
        return False

    bbox_area = w * h
    if bbox_area == 0:
        return False

    extent = area / float(bbox_area)
    circularity = 4 * np.pi * area / (peri ** 2)
    ar = w / float(h)

    if extent < 0.18:
        return False
    if circularity < 0.12:
        return False
    if not (0.45 <= ar <= 1.8):
        return False

    return True


def build_shape_candidate_mask(blur_gray, blur_sat):
    """Return a mask of blobs whose geometry looks close to our known shapes."""
    adaptive_dark = cv2.adaptiveThreshold(
        blur_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        61,
        6
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

    # Opening first helps suppress the thin track line before contour filtering.
    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((7, 7), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, open_kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(blur_gray)

    min_area, max_area = SHAPE_AREA_RANGE
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        if not is_shape_candidate(cnt):
            continue

        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask


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

    if verts in [4, 5, 6] and (20000 < A < 35000) and (0.95 < ar < 1.25):
        return 'warning', ar, A, P, C, verts

    if verts in [6, 7, 8] and (9000 < A < 20000) and (0.90 < ar < 1.4) and (0.09 < C < 0.33):
        return 'qr', ar, A, P, C, verts

    if verts in [9, 10] and (14000 < A < 28000) and (1.10 < ar < 1.45):
        return 'button', ar, A, P, C, verts

    if verts in [8, 10, 12] and (8000 < A < 13000) and (0.90 < ar < 1.20) and (0.17 < C < 0.30):
        return 'fingerprint', ar, A, P, C, verts

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
        if A < 7000 and 0.2<C<0.26:
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
        elif A < 7000 and 0.2<C<0.26:
            return 'Arrow', ar, A, P, C, verts
        elif 8000<A<13500 and 0.12<C<0.18:
            return 'recycle', ar, A, P, C, verts
    elif verts == 9:
        if 8000<A<13500 and 0.12<C<0.18:
            return 'recycle', ar, A, P, C, verts
        elif A < 7000 and 0.2<C<0.26:
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


        if shape_label != 'Noise' or (shape_label == 'Noise' and area > 1000):
            results.append({
                'label': shape_label,
                'color': None,#detected_color
                'direction': direction,
                'contour': cnt,
                'area': area,
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
            
            original_frame = frame.copy()
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
                
            # 4. Show the raw frame beside the contour overlay for tuning
            raw_view = cv2.resize(original_frame, (640, 480))
            contour_view = cv2.resize(frame, (640, 480))
            debug_view = np.hstack((raw_view, contour_view))
            cv2.imshow("Shape Debug: Original | Contours", debug_view)
            cv2.imshow("Shape Threshold Mask", thresh_mask)
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
