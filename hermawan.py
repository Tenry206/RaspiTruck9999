from gpiozero import OutputDevice, PWMOutputDevice as pwm
from time import time,sleep
from Camera_backup import Camera
import cv2
import threading
from shared_state import SharedState
from Test_Shape import process_shapes
from ColouredLineErrorV2 import toilet
import numpy as np

from Face_Scanner import FaceScanner

state = SharedState()

if not hasattr(state, 'current_shape'):
    state.current_shape = ""


# ------ Motor A GPIO setup ------


ENA = pwm(17)
IN1 = OutputDevice(22)
IN2 = OutputDevice(27)


# ------ Motor B GPIO setup ------


ENB = pwm(13)
IN3 = OutputDevice(5)
IN4 = OutputDevice(6)
    

# ------ Motor with Negative PWM Control ------


def set_motor(motor_pwm, in1, in2, speed):

    # Clamp speed between -1 to 1
    speed = max(min(speed, 1), -1) 

    # 0.4 PWM Deadzone 
    if 0 < speed < 0.23:
        adjusted_speed = 0.23

    elif -0.23 < speed < 0:
        adjusted_speed = -0.23

    else:
        adjusted_speed = speed

    # Wheel spin direction
    if adjusted_speed >= 0:
        in1.on()
        in2.off()
        motor_pwm.value = adjusted_speed

    else:
        in1.off()
        in2.on()
        motor_pwm.value = abs(adjusted_speed)


# ------ Stop function ------


def stop():

    ENA.off()
    ENB.off()

    IN1.off()
    IN2.off()

    IN3.off()
    IN4.off()


# ------ General Move Function ------


def Move(left, right):

    # Stop if both speeds are zero
    if left == 0 and right == 0:
        stop()
        return

    # Left motor backward
    if left < 0:
        ENB.value = abs(left)
        IN3.off()
        IN4.on()

    else:
        ENB.value = left
        IN3.on()
        IN4.off()

    # Right motor backward
    if right < 0:
        ENA.value = abs(right)
        IN1.off()
        IN2.on()

    else:
        ENA.value = right
        IN1.on()
        IN2.off()


# ------ Turn function ------


def Turn(angle, speed = 0.5 , clockwise = True):
     
    # Time taken for 1 complete rotation in seconds
    T_360 = 1.5

    # Rotation turn time calculation
    turn_time = (angle / 360) * T_360/speed

    if clockwise: 
        Move(speed, -speed)
    else:
        Move(-speed, speed)
    
    sleep(turn_time)
    Move(0,0)


# ------ Camera Thread ------ 


def thread_camera():
    global cam
    print("Camera Thread Started...")
    
    while state.running:
        frame = cam.read()
        if frame is not None:
            state.update_frame(frame)


# ------ Line Following Thread ------


def thread_line_follow():
    # Added coloredLine to global so the thread can access your toilet() instance
    global cam, coloredLine

    Kp = 5
    Ki = 0
    Kd = 0.36

    base_speed = 0.23 #0.7
    last_error = 0
    integral = 0
    dt = 0.02

    lost_counter = 0
    search_speed = 0.7

    fps_start_time = time()
    fps_frame_count = 0

    while state.running:
        

        if state.get_override() == 'FACE_SCAN':
            sleep(0.1)
            continue

        # 2. GRAB FRAME FROM SHARED MEMORY (NOT FROM THE CAMERA!)
        frame = state.get_frame()
        if frame is None:
            sleep(0.01)
            continue
        
        fps_frame_count +=1
        current_time = time()

        if (current_time - fps_start_time) >= 1.0:
            fps = fps_frame_count / (current_time - fps_start_time)
            #print(f"--- Sequential Loop Speed: {fps:.1f} FPS ---")
            fps_frame_count = 0
            fps_start_time = current_time

        # ------ Get Errors ------
        error, thresh, cx, turn, area = cam.get_error(frame)
        error_color, colorBool, activeColor = coloredLine.colored_error(frame)

        # ==========================================
        # --- NEW DEBUGGING BLOCK ---
        # ==========================================
        # 1. Debug the Black Line (We have the area and 'cx' centroid)
        if error is not None:
            # Convert frame to HSV to read the color
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Sample the exact pixel at the center of the line (y=400 is deep in your ROI)
            safe_cx = max(0, min(639, int(cx))) # Prevent out-of-bounds crash
            h, s, v = hsv_frame[400, safe_cx]

        active_error = None
        
        # Priority 1: Check for Colored Line
        if colorBool and error_color is not None:
            active_error = error_color / 2.5
            lost_counter = 0
            
        # Priority 2: Fallback to Black Track
        elif error is not None and area >= 300:
            active_error = error / 2.5
            lost_counter = 0

        # ==========================================
        # 2. HANDLE LINE LOST
        # ==========================================
        # Only panic if BOTH the color line and black line are missing
        if active_error is None:
            lost_counter += 1
            #print("I'm so lost - Searching...")
            
            if lost_counter > 0: 
                if last_error >= 0:
                    state.set_steering(search_speed, -search_speed+0.175)
                else:
                    state.set_steering(-search_speed+0.175, search_speed)
                
                sleep(dt)
                continue # Skip PID calculation while spinning

        # ==========================================
        # 3. UNIFIED PID CALCULATION
        # ==========================================
        # This math now works flawlessly for whichever line is "active"
        integral += active_error * dt
        derivative = (active_error - last_error) / dt

        pid_output = Kp * active_error + Ki * integral + Kd * derivative
        last_error = active_error

        # ------ Compute motor speeds ------
        left_speed = base_speed + pid_output / 1000   
        right_speed = base_speed - pid_output / 1000

        # ------ Set motor directions ------
        state.set_steering(left_speed, right_speed)

        # 50 FPS Limit
        sleep(0.02)

def thread_vision():

    global orb, matcher, templatesF, vision_roi
    
    symbol_cooldown = 0
    while state.running:
        start = time()

        if state.get_override() == 'FACE_SCAN':
            sleep(0.1)
            continue

        frame = state.get_frame()

        if frame is None:
            sleep(0.01)
            continue
            
        if symbol_cooldown > 0:
            symbol_cooldown -=1
            sleep(0.01)
            continue

        h, w = frame.shape[:2]
        vision_roi = frame[int(h * 0.0):int(h * 0.9),int(w * 0.2):int(w * 0.8), :]
        resized_roi = cv2.resize(vision_roi, None, fx = 0.775, fy = 0.775)
        frame_gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
        detected_shapes, shape_thresh = process_shapes(vision_roi)

        state.shape_mask = shape_thresh

        for shape in detected_shapes:

            if shape['label']=='Arrow':
                #print(f"Detected {shape['color']} Arrow pointing {shape['direction']}")
                if shape['direction'] == 'Left':
                    state.set_override('SPIN_LEFT')
                elif shape['direction'] == 'Right':
                    state.set_override('SPIN_RIGHT')
                break  

            elif shape['label'] == 'recycle':
                print("Recycle detected")
                state.set_override('spongebob')

            elif shape['label'] == 'warning' or shape['label'] == 'button':
                state.set_override('STOP')

            elif shape['label'] == 'qr' or shape['label'] == 'fingerprint':
                print(shape['label'])
                state.set_override('FACE_SCAN')

        # 10 FPS
        sleep(0.01)

        fps = 1.0 / (time() - start)
        #print(fps)

def thread_motor():
    while state.running:
        override = state.get_override()

        if override == 'STOP':
            stop()
            sleep(1)
            state.set_override('NONE')
        elif override == 'FACE_SCAN':
            stop()
        elif override == 'spongebob':
            stop()
            Turn(450, speed = 1, clockwise = True) 
            state.set_override('NONE')
        
#        elif override == 'squidward':
#            Turn(30, speed = 1, clockwise = True)
#            Turn(30, speed = 1, clockwise = False)
#            Turn(30, speed = 1, clockwise = True)
#            state.set_override('NONE')

        elif override == 'SPIN_LEFT':
            Turn(80, speed = 1, clockwise = False)
            print('left')
            state.set_override('NONE')

        elif override == 'SPIN_RIGHT':
            Turn(80, speed = 1, clockwise = True)
            print('right')
            state.set_override('NONE')
        else:
            left,right = state.get_steering()
            set_motor(ENA, IN1, IN2, left)
            set_motor(ENB, IN3, IN4, right) 

        # 20 FPS
        sleep(0.05)


# ------ Initialization ------


print("Initializing System ...")

cam = Camera(resolution=(640,480), fps=60)
coloredLine = toilet()

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

face_scanner = FaceScanner()
print("Starting Threads ...")

threads = [
    threading.Thread(target=thread_camera),
    threading.Thread(target=thread_line_follow),
    threading.Thread(target=thread_vision),
    threading.Thread(target=thread_motor)
]

for t in threads:
    t.start()

try:
    while state.running:
        display_frame = state.get_frame()

        if display_frame is not None:
            override = state.get_override()
            # ===============================================
            # FACIAL RECOGNITION MODE
            # ===============================================
            if override == 'FACE_SCAN':
                
                # 1. Run the Scanner Math
                is_face_found, face_data_list = face_scanner.scan_for_face(display_frame)
                
                # 2. Draw the boxes and text
                if is_face_found:
                    for data in face_data_list:
                        fx, fy, fw, fh = data['box']
                        cv2.rectangle(display_frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)
                        
                        # Display the recognized Name and Confidence!
                        text = f"{data['name']} {data['confidence']}"
                        cv2.putText(display_frame, text, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 3. Add the UI Prompts
                cv2.putText(display_frame, f"Detected: {state.current_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(display_frame, "PRESS 'C' TO RESUME DRIVING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 4. Open the dedicated window
                cv2.imshow("Facial Recognition Checkpoint", display_frame)
                
                # 5. Listen for the 'C' key!
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    print("Continue authorized by user. Resuming path...")
                    cv2.destroyWindow("Facial Recognition Checkpoint")
                    state.set_override('NONE') # This wakes all threads back up!
                elif key == ord('q'):
                    state.running = False
            else:        
                #cv2.imshow("Robot View", display_frame)
                #cv2.imshow("Vision ROI", vision_roi)

                if hasattr(state, 'shape_mask') and state.shape_mask is not None:
                    cv2.imshow("Linked Threshold Mask", state.shape_mask)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit triggered ...")
                    state.running = False

except KeyboardInterrupt:

    print("Interrupted by User.")

finally:

    # Stop motors and camera
    print("Shutting downsafely ...")

    state.running = False

    for t in threads:
        t.join()

    stop()
    cam.stop()
    cv2.destroyAllWindows()
