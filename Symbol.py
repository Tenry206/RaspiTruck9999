import numpy as np
import cv2
from Camera import Camera

cam = Camera(resolution=(640,480), fps=60)

qr = cv2.imread('symbols/qr.png')
qr = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)

while True:

    frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    tK, tD = orb.detectAndCompute(qr, None) # tK for template keypoints and d for descriptor
    vK, vD = orb.detectAndCompute(frame, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(tD, vD)

    frame2 = cv2.drawMatches(qr, tK, frame, vK, matches[:20], None)

    cam.display(frame2)
    

    if cv2.waitKey(1) == 27:
        break