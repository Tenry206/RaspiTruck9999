import cv2
import time
from Camera_backup import Camera 
from Face_Scanner import FaceScanner

def main():
    print("--- Starting Standalone Face Scanner Test ---")
    
    # 1. Initialize your robot's camera
    print("Warming up camera...")
    cam = Camera(resolution=(640, 480), fps=30)
    
    # 2. Initialize your new MediaPipe scanner
    scanner = FaceScanner()
    
    print("Ready! Point the camera at your face. Press 'q' to quit.")
    
    try:
        while True:
            # Grab a frame from the Picamera2
            frame = cam.read()
            if frame is None:
                continue
            
            # 3. Test the scanner logic!
            # We wrap it in a timer to see exactly how fast it runs on your Pi 4
            start_time = time.time()
            is_face_found = scanner.scan_for_face(frame)
            process_time = (time.time() - start_time) * 1000
            
            # 4. Visual & Terminal Feedback
            if is_face_found:
                print(f"? FACE DETECTED! (Math took: {process_time:.1f} ms)")
                # Draw a thick green border so you can easily see it worked
                cv2.rectangle(frame, (0, 0), (640, 480), (0, 255, 0), 15)
                cv2.putText(frame, "FACE FOUND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                print(f"? Scanning... (Math took: {process_time:.1f} ms)")
                cv2.putText(frame, "Looking for face...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show the live feed
            cv2.imshow("Face Scanner Test Feed", frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Sleep for 0.1s to simulate the 10 FPS tick rate of your main FSM thread
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\nTest Interrupted.")
    finally:
        print("Shutting down camera safely...")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()