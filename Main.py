from gpiozero import OutputDevice, PWMOutputDevice as pwm
from time import sleep
from Camera import Camera
import cv2
import numpy as np

# ------ Motor A GPIO setup ------
ENA = pwm(17)
IN1 = OutputDevice(22)
IN2 = OutputDevice(27)

# ------ Motor B GPIO setup ------
ENB = pwm(13)
IN3 = OutputDevice(5)
IN4 = OutputDevice(6)

# ------ PID parameters ------
Kp = 0.6
Ki = 0.0
Kd = 0.2

base_speed = 0.4  # duty cycle 0-1
last_error = 0
integral = 0
dt = 0.02  # control loop 50 Hz

# ------ Camera setup ------
cam = Camera(resolution=(640,480), fps=60)

try:
    while True:
        # ---- Capture frame and get error ----
        frame = cam.read()
        error, thresh, cx = cam.get_error(frame)
        if error is None:
            error = 0  # fallback if line is lost

        # ---- PID calculation ----
        integral += error * dt
        derivative = (error - last_error) / dt
        pid_output = Kp * error + Ki * integral + Kd * derivative
        last_error = error

        # ---- Compute motor speeds ----
        left_speed = base_speed + pid_output / 1000   # scale PID to 0-1
        right_speed = base_speed - pid_output / 1000

        # Clamp 0-1
        left_speed = max(min(left_speed, 1), 0)
        right_speed = max(min(right_speed, 1), 0)

        # ---- Set motor directions ----
        if left_speed > 0:
            IN1.on()
            IN2.off()
        else:
            IN1.off()
            IN2.on()

        if right_speed > 0:
            IN3.on()
            IN4.off()
        else:
            IN3.off()
            IN4.on()

        # ---- Set PWM duty cycle ----
        ENA.value = left_speed
        ENB.value = right_speed

        # ---- Display for debugging ----
        cam.display(frame, cx)
        cam.display_draw(thresh)

        sleep(dt)

        # ---- Exit ----
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop motors and camera
    ENA.off()
    ENB.off()
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()
    cam.stop()
