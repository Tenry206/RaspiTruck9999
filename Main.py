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
    

# ------ Motor with negative pwm control ------


def set_motor(ENA, IN1, IN2, speed):
    speed = max(min(speed, 1), -1) 

    if left_speed < 0.3 and left_speed > 0:
        adjusted_left_speed = 0.3
    elif left_speed > -0.3 and left_speed < 0:
        adjusted_left_speed = -0.3
    else:
        adjusted_left_speed = left_speed
    if right_speed < 0.3 and right_speed > 0:
        adjusted_right_speed = 0.3
    elif right_speed > -0.3 and right_speed < 0:
        adjusted_right_speed = -0.3
    else:
        adjusted_right_speed = right_speed

    if speed >= 0:
        IN1.on()
        IN2.off()
        ENA.value = speed
    else:
        IN1.off()
        IN2.on()
        ENA.value = abs(speed)


# ------ PID parameters ------


Kp = 2.56 #4.7  #tu 2.60  ku = 4.7
Ki = 2.51
Kd = 0.157

base_speed = 0.4  # duty cycle 0-1
last_error = 0
integral = 0
dt = 0.02  # control loop 50 Hz


# ------ Camera setup ------


cam = Camera(resolution=(640,480), fps=60)

lost_counter = 0
search_speed = 0.7

try:
    while True:
        # ---- Capture frame and get error ----
        frame = cam.read()
        error, thresh, cx, turn, area = cam.get_error(frame)
        print(area)
        # ------ Sharp 90 turn ------


        if turn == "LEFT":
            
            set_motor(ENA, IN1, IN2, -0.4)
            set_motor(ENB, IN3, IN4, 0.9)
            sleep(0.5)
            continue
        elif turn == "RIGHT":
            
            set_motor(ENA, IN1, IN2, 0.9)
            set_motor(ENB, IN3, IN4, -0.4)
            sleep(0.5)
            continue


        # ------ Handle line lost ------


        if error is None or area < 300:
            lost_counter += 1
            #print(f"Line lost! Counter: {lost_counter}")
            
            # Use previous error for PID
            error = last_error
            
            # Slow down when lost
            base_speed_lost = search_speed
            
            # If lost for a while, do a gentle spin to search
            if lost_counter > 20:
                if last_error >= 0:
                    set_motor(ENA, IN1, IN2, search_speed)
                    set_motor(ENB, IN3, IN4, -search_speed)
                else:
                    set_motor(ENA, IN1, IN2, -search_speed)
                    set_motor(ENB, IN3, IN4, search_speed)
                sleep(dt)
                continue
        else:
            lost_counter = 0
            base_speed_lost = base_speed
        
        error /= 2.5

        # ------ PID calculation ------
        integral += error * dt
        derivative = (error - last_error) / dt
        pid_output = Kp * error + Ki * integral + Kd * derivative
        last_error = error

        # ------ Compute motor speeds ------
        left_speed = base_speed + pid_output / 1000   # scale PID to 0-1
        right_speed = base_speed - pid_output / 1000

    
        # ------ Set motor directions ------
        set_motor(ENA, IN1, IN2, left_speed)
        set_motor(ENB, IN3, IN4, right_speed)

        # ------ Display for debugging ------
        cam.display(frame, cx)
        cam.display_draw(thresh)

        sleep(dt)

        # ------ Exit ------
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
