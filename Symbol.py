import numpy as np
import cv2
from Camera import Camera

cam = Camera(resolution=(640,480), fps=60)

qr = cv2.imread('symbols/button.png')
qr = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=2000, fastThreshold=15)
tK, tD = orb.detectAndCompute(qr, None) # tK for template keypoints and d for descriptor
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:

    frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    vK, vD = orb.detectAndCompute(frame, None)

    if tD is None or vD is None or len(tD) == 0 or len(vD) == 0:
        cam.display(frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    matches = matcher.match(tD, vD)

    frame2 = cv2.drawMatches(qr, tK, frame, vK, matches[:20], None)

    matches = sorted(matches, key=lambda val: val.distance)

    matchesNum = len(matches)

    print(f"{str(matchesNum)}")
    cam.display(frame2)
    

    if cv2.waitKey(1) == 27:
        break