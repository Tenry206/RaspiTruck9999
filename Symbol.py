import numpy as np
import cv2
from Camera import Camera

cam = Camera(resolution=(640,480), fps=60)

qr = cv2.imread('symbols/qr.png')
qr = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=2000, fastThreshold=15)
tK, tD = orb.detectAndCompute(qr, None) # tK for template keypoints and d for descriptor
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

RATIO = 0.8

MIN_GOOD = 12
MIN_INLIER = 15
RANSAC_THRESH = 5.0

while True:

    frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    vK, vD = orb.detectAndCompute(frame, None)

    if tD is None or vD is None or len(tD) == 0 or len(vD) == 0:
        cam.display(frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    
    knn = matcher.knnMatch(tD, vD, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair

        if m.distance < RATIO * n.distance:
            good.append(m)

    good = sorted(good, key=lambda m: m.distance)


    #matches = sorted(matches, key=lambda val: val.distance)

    matchesNum = len(good)

    inliers = 0
    H = None
    inlier_matches = []

    if matchesNum >= MIN_GOOD:
        src_pts = np.float32([tK[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([vK[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)

    if mask is not None and H is not None:
        mask = mask.ravel().astype(bool)
        inliers = int(mask.sum())
        inlier_matches = [good[i] for i in range(len(good)) if mask[i]]

    print(f"good={matchesNum}, inliers={inliers}")

    detected = (H is not None) and (inliers >= MIN_INLIER)

    show = inlier_matches if detected else good
    show = sorted(show, key=lambda m: m.distance)

    vis = cv2.drawMatches(qr, tK, frame, vK, show[:30], None, flags=2)

    frame2 = cv2.drawMatches(qr, tK, frame, vK, good[:20], None)

    if detected:
        h, w = qr.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        proj = cv2.perspectiveTransform(corners, H)

    cam.display(vis)
    

    if cv2.waitKey(1) == 27:
        break