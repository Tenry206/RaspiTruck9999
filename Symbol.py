import cv2
from Camera import Camera
import numpy as np


cam = Camera(resolution=(640,480), fps=60)
frame = cam.read()

templates = {
    'arrow_down': cv2.imread('symbols/arrow_down.png',0),
    'arrow_left': cv2.imread('symbols/arrow_left.png',0),
    'arrow_right': cv2.imread('symbols/arrow_right.png',0),
    'arrow_up': cv2.imread('symbols/arrow_up.png',0),
    'button': cv2.imread('symbols/button.png',0),
    'fingerprint': cv2.imread('symbols/fingerprint.png',0),
    'qr': cv2.imread('symbols/qr.png',0),
    'recycle': cv2.imread('symbols/recycle.png',0),
    'warning': cv2.imread('symbols/warning.png',0),
}
w , h = template.shape[::-1]

frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

if max_val > 0.7:
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    print("Symbol detected!")