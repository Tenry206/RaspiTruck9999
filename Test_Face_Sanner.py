import cv2
import time
from Camera_backup import Camera 
from Face_Scanner import FaceScanner

def main():
    print("--- Starting Standalone Face Scanner Test ---")
    
    print("Warming up camera...")
    cam = Camera(resolution=(640, 480), fps=30)
    scanner = FaceScanner()
    
    print("Ready! Point the camera at your face. Press 'q' to quit.")
    
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            
            start_time = time.time()
            
            # Now it returns BOTH the True/False flag AND the coordinate data
            is_face_found, face_data_list = scanner.scan_for_face(frame)
            
            process_time = (time.time() - start_time) * 1000
            
            if is_face_found:
                print(f"? FACE DETECTED! (Math took: {process_time:.1f} ms)")
                
                # Loop through every face found and draw the exact boxes
                for data in face_data_list:
                    
                    # 1. Draw Face (Green)
                    fx, fy, fw, fh = data['box']
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)
                    cv2.putText(frame, "FACE", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 2. Draw Eyes (Blue)
                    for (ex, ey, ew, eh) in data['eyes']:
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                        
                    # 3. Draw Mouth (Red)
                    for (mx, my, mw, mh) in data['mouths']:
                        cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                        
            else:
                print(f"? Scanning... (Math took: {process_time:.1f} ms)")
                cv2.putText(frame, "Looking for face...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Face Scanner Test Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\nTest Interrupted.")
    finally:
        print("Shutting down camera safely...")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()