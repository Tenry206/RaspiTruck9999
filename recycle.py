from gpiozero import OutputDevice, PWMOutputDevice as pwm
from time import time,sleep
from Camera_backup import Camera
import cv2
import threading
from shared_state import SharedState
from Symbol import symbol_detect, build_templatesF
from Shape import process_shapes
from ColouredLineErrorV2 import toilet

state = SharedState()


# ------ Templates ------


templates = {
    'button': cv2.imread('symbols/button.png', 0),
    'fingerprint': cv2.imread('symbols/fingerprint.png', 0),
    'qr': cv2.imread('symbols/qr.png', 0),
    'recycle': cv2.imread('symbols/recycle.png', 0),
    'warning': cv2.imread('symbols/warning.png', 0)
}


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
    if 0 < speed < 0.3:
        adjusted_speed = 0.3

    elif -0.3 < speed < 0:
        adjusted_speed = -0.3

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


# ------ PID parameters ------


def thread_line_follow():
    # Added coloredLine to global so the thread can access your toilet() instance
    global cam, coloredLine

    Kp = 5
    Ki = 0
    Kd = 0.36

    base_speed = 0.3 #0.7
    last_error = 0
    integral = 0
    dt = 0.02

    lost_counter = 0
    search_speed = 0.7

    fps_start_time = time()
    fps_frame_count = 0

    while state.running:

        # ------ Capture Frame ------
        frame = cam.read()
        if frame is None:
            continue

        state.update_frame(frame)
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
        sleep(dt)

def thread_vision():

    global orb, matcher, templatesF, vision_roi
    
    symbol_cooldown = 0
    while state.running:
        start = time()
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
        detected_shapes, shape_thresh = process_shapes(resized_roi)

        state.shape_mask = shape_thresh

        for shape in detected_shapes:

            if shape['label']=='Arrow':
                #print(f"Detected {shape['color']} Arrow pointing {shape['direction']}")
                if shape['direction'] == 'Left':
                    state.set_override('SPIN_LEFT')
                elif shape['direction'] == 'Right':
                    state.set_override('SPIN_RIGHT')
                break  

            elif shape['label'] != 'Noise':
                #print(f"Detected Shape: {shape['label']}")
                state.set_override('STOP')
                sleep(1)
                state.set_override('NONE') 
                break

            elif shape['label'] == 'Noise':
                orb_start_time = time()

                symbol = symbol_detect(frame_gray, templatesF, orb, matcher)

                orb_delay = (time() - orb_start_time) *1000
                #print(f"WARNING: ORB Stalled motor for {orb_delay:.1f} ms!")

                if symbol !=None:
                    if symbol == 'fingerprint' or symbol == 'qr':
                        print(symbol)

                    elif symbol == 'recycle':
                        state.set_override('spongebob')
                    
                    elif symbol == 'warning' or symbol == 'button':
                        state.set_override('STOP')
                break

        # 10 FPS
        sleep(0.01)

        fps = 1.0 / (time() - start)
        print(fps)

def thread_motor():
    while state.running:
        override = state.get_override()

        if override == 'STOP':
            stop()
            sleep(1)
            state.set_override('NONE')

        elif override == 'spongebob':
            stop()
            Turn(480, speed = 1, clockwise = True) 
            state.set_override('NONE')

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
orb = cv2.ORB_create(nfeatures=1200,fastThreshold=10,nlevels=8,scaleFactor=1.2,edgeThreshold=15,patchSize=31)#(nfeatures=2200, fastThreshold=15, nlevels=12, scaleFactor=1.2, patchSize=31)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
templatesF = build_templatesF(templates, orb)

print("Starting Threads ...")

threads = [
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
            cv2.imshow("Robot View", display_frame)
            cv2.imshow("Vision ROI", vision_roi)

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
