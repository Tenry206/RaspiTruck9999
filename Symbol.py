import numpy as np
import cv2

def build_templatesF(templates, orb):
    templatesF = {}
    for name, img in templates.items():
        if img is None:
            continue
        k, d = orb.detectAndCompute(img, None)
        templatesF[name] = {"kps": k, "des": d}
    return templatesF

def symbol_detect(frame_gray, templatesF, orb, matcher,
                  RATIO=0.8, MIN_GOOD=12, MIN_INLIER=10, RANSAC_THRESH=5.0):

    vK, vD = orb.detectAndCompute(frame_gray, None)
    if vD is None or len(vD) < 2:
        return None

    best_name = None
    best_good = 0
    best_inliers = 0
    best_H = None

    for name, tpl in templatesF.items():
        tK, tD = tpl["kps"], tpl["des"]
        if tD is None or len(tD) < 2:
            continue

        knn = matcher.knnMatch(tD, vD, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < RATIO * n.distance:
                good.append(m)

        if len(good) < MIN_GOOD:
            continue

        src_pts = np.float32([tK[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([vK[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
        if H is None or mask is None:
            continue

        inliers = int(mask.ravel().sum())

        if (inliers > best_inliers) or (inliers == best_inliers and len(good) > best_good):
            best_name = name
            best_good = len(good)
            best_inliers = inliers
            best_H = H

    detected = (best_H is not None) and (best_inliers >= MIN_INLIER) and (best_good >= MIN_GOOD)
    return best_name if detected else None