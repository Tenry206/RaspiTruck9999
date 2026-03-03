import cv2
from Camera import Camera
import numpy as np


cam = Camera(resolution=(640,480), fps=60)
frame = cam.read()

frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

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

threshold = 0.75


for symbol_name, template in templates.items():
    result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

        
print(f"Detected: {symbol_name}")