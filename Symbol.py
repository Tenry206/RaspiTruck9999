import numpy as np
import cv2

def symbol_detect(frame):

    templates = {
        'button': cv2.imread('symbols/button.png', 0),
        'fingerprint': cv2.imread('symbols/fingerprint.png', 0),
        'qr': cv2.imread('symbols/qr.png', 0),
        'recycle': cv2.imread('symbols/recycle.png', 0),
        'warning': cv2.imread('symbols/warning.png', 0),
    }

    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=12)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


    RATIO = 0.8

    MIN_GOOD = 12
    MIN_INLIER = 15
    RANSAC_THRESH = 5.0

    templatesF = {}
    for name, img in templates.items():
        if img is None:
            print("Epic fail")
            continue
        tK, tD = orb.detectAndCompute(img, None)
        templatesF[name] = {"img": img, "kps": tK, "des": tD}


    while True:

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vK, vD = orb.detectAndCompute(frame_gray, None)

        if tD is None or vD is None or len(tD) == 0 or len(vD) == 0:
            #cam.display(frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        best_name = None
        best_tpl = None
        best_good = 0
        best_inliers = 0
        best_H = None
        best_inlier_matches = []
        best_tK = None

        for name, tpl in templatesF.items():
            tK = tpl["kps"]
            tD = tpl["des"]
            img = tpl["img"]

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

            good = sorted(good, key=lambda m: m.distance)


            #matches = sorted(matches, key=lambda val: val.distance)

            matchesNum = len(good)

            inliers = 0
            H = None
            mask = None
            inlier_matches = []

            if matchesNum >= MIN_GOOD:
                src_pts = np.float32([tK[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([vK[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)

            if mask is not None and H is not None:
                mask = mask.ravel().astype(bool)
                inliers = int(mask.sum())
                inlier_matches = [good[i] for i in range(len(good)) if mask[i]]

            if (inliers > best_inliers) or (inliers == best_inliers and matchesNum > best_good):
                best_name = name
                best_tpl = img
                best_tK = tK
                best_good = matchesNum
                best_inliers = inliers
                best_H = H
                best_inlier_matches = inlier_matches

        detected = (H is not None) and (inliers >= MIN_INLIER)

        if best_name is None:
            if cv2.waitKey(1) == 27:
                break
            continue

        show = inlier_matches if detected else good
        show = sorted(show, key=lambda m: m.distance)

        vis = cv2.drawMatches(best_tpl, tK, frame, vK, show[:30], None, flags=2)

        #cam.display(vis)

        if cv2.waitKey(1) == 27:
            break

    return best_name