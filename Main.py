from Camera import Camera
import cv2

cam = Camera(resolution = (640,480),fps = 60)

try:
    while True:
        #Read a frame
        frame = cam.read()

        #Get PID error and binary mask
        error, thresh, cx = cam.get_error(frame)

        if error is not None:
            print("PID Error:", error)
        
        cam.display(frame, cx)

        cam.display_draw(thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.stop()