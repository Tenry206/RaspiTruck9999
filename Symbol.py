import numpy as np
import cv2
from Camera import Camera

cam = Camera(resolution=(640,480), fps=60)

while True:
    
    frame = cam.read()

    orb = cv2.ORB_create()

    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)

    frame2 = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    cam.display(frame2)

    if cv2.waitKey(1) == 27:
        break