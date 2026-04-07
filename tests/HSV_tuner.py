import cv2
import numpy as np
from Camera import Camera # Uses your existing camera setup

def nothing(x):
    pass

def main():
    print("Starting HSV Tuner with High-Contrast Contours...")
    cam = Camera(resolution=(640, 480), fps=30)
    
    # Create a window with Trackbars
    cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV Tuner', 400, 300)
    
    # Default Trackbar positions (Start wide open)
    cv2.createTrackbar('Hue Min', 'HSV Tuner', 0, 179, nothing)
    cv2.createTrackbar('Sat Min', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('Val Min', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('Hue Max', 'HSV Tuner', 179, 179, nothing)
    cv2.createTrackbar('Sat Max', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('Val Max', 'HSV Tuner', 255, 255, nothing)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # We usually only want to tune the bottom half of the frame for the line
            h, w = frame.shape[:2]
            roi = frame[int(h/2):h, :].copy() # Added .copy() so we can draw on it safely
            
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Read current trackbar positions
            h_min = cv2.getTrackbarPos('Hue Min', 'HSV Tuner')
            s_min = cv2.getTrackbarPos('Sat Min', 'HSV Tuner')
            v_min = cv2.getTrackbarPos('Val Min', 'HSV Tuner')
            h_max = cv2.getTrackbarPos('Hue Max', 'HSV Tuner')
            s_max = cv2.getTrackbarPos('Sat Max', 'HSV Tuner')
            v_max = cv2.getTrackbarPos('Val Max', 'HSV Tuner')

            # Create arrays for upper and lower limits
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            # Create the raw mask
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up the mask to remove tiny dust particles
            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Bitwise-AND mask and original image to show the color
            result = cv2.bitwise_and(roi, roi, mask=mask_clean)
            
            # Find contours on the cleaned mask
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw a thick BLUE outline around any contour large enough to be a line
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500: # Ignore tiny flecks of background noise
                    # Draw a thick Blue contour on BOTH windows
                    cv2.drawContours(roi, [cnt], -1, (255, 0, 0), 3)
                    cv2.drawContours(result, [cnt], -1, (255, 0, 0), 3)
                    
                    # Draw a highly visible Black centroid dot
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(roi, (cx, cy), 6, (0, 0, 0), -1)
                        cv2.circle(result, (cx, cy), 6, (0, 0, 0), -1)

            # Display the windows
            cv2.imshow("Original ROI (With Blue Contours)", roi)
            cv2.imshow("Mask (White = Detected)", mask_clean)
            cv2.imshow("Color Result", result)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"--- FINAL VALUES ---")
                print(f"Lower HSV: [{h_min}, {s_min}, {v_min}]")
                print(f"Upper HSV: [{h_max}, {s_max}, {v_max}]")
                break

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()