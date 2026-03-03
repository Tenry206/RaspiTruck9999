import cv2
from Camera import Camera
import numpy as np


cam = Camera(resolution=(640,480), fps=60)



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

w = 20
h = 20

threshold = 0.75

while True:
    frame = cam.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    for symbol_name, template in templates.items():

        w, h = template.shape[::-1]
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            print(f"Symbol name: {symbol_name}")

        cam.display(frame)
    
    


        