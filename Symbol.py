import cv2
from Camera import Camera
import numpy as np


cam = Camera(resolution=(640,480), fps=60)
frame = cam.read()

template = cv2.imread('', 0)
w , h = template.shape[::-1]

frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)