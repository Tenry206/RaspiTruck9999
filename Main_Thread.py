from gpiozero import OutputDevice, PWMOutputDevice as pwm
from time import time,sleep
from Camera import Camera
import cv2
import threading
from shared_state import SharedState
from Symbol import symbol_detect, build_templatesF
from Shape import process_shapes

state = SharedState()
# ------ Templates ------

templates = {
    'button': cv2.imread('symbols/button.png', 0),
    'fingerprint': cv2.imread('symbols/fingerprint.png', 0),
    'qr': cv2.imread('symbols/qr.png', 0),
    'recycle': cv2.imread('symbols/recycle.png', 0),
    'warning': cv2.imread('symbols/warning.png', 0)
}

#
# ------ Motor A GPIO setup ------


ENA = pwm(17)
IN1 = OutputDevice(22)
IN2 = OutputDevice(27)

# ------ Motor B GPIO setup ------


ENB = pwm(13)
IN3 = OutputDevice(5)
IN4 = OutputDevice(6)
    

# ------ Motor with negative pwm control ------


def set_motor(motor_pwm, in1, in2, speed):
    # Ensure speed is within -1 to 1 [cite: 61, 62]
    speed = max(min(speed, 1), -1) 

    # PWM Dead-zone clamping [cite: 91, 124]
    if 0 < speed < 0.4:
        adjusted_speed = 0.4
    elif -0.4 < speed < 0:
        adjusted_speed = -0.4
    else:
        adjusted_speed = speed

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

# Move function for degree turn
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
     
    # Experimentally determined time for 360 degree rotation at speed = 1
    T_360 = 1.5 #seconds

    # Calculate rotation time
    turn_time = (angle / 360) * T_360/speed

    if clockwise: 
        Move(speed, -speed)
    else:
        Move(-speed, speed)
    
    sleep(turn_time)
    Move(0,0)

# ------ PID parameters ------

def thread_line_follow():
    global cam
    Kp = 11.6 #kp = 5 , kd = 0.4; kp = 7, kd = 0.4q
    Ki = 0
    Kd = 0.36
    #11.7

    base_speed = 0.7  # duty cycle 0-1
    last_error = 0
    integral = 0
    dt = 0.02  # control loop 50 Hz


    

    lost_counter = 0
    search_speed = 0.7
    #counter = 0
    '''
    orb = cv2.ORB_create(nfeatures=2200, fastThreshold=15, nlevels=12, scaleFactor=1.2, patchSize=31)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    templatesF = build_templatesF(templates, orb)
    '''
    symbol_cooldown = 0

    fps_start_time = time()
    fps_frame_count = 0

    while state.running:
        # ---- Capture frame and get error ----
        frame = cam.read()
        if frame is None:
            continue

        state.update_frame(frame)
        fps_frame_count +=1
        current_time = time()
        if (current_time - fps_start_time) >=1.0:
            fps = fps_frame_count / (current_time - fps_start_time)
            print(f"--- Sequential Loop Speed: {fps:.1f} FPS ---")
            fps_frame_count = 0
            fps_start_time = current_time

        '''
        if symbol is not None:
            stop()
            
            symbol_cooldown = 60   # ignore symbols for ~1 sec at 60fps
            continue
        '''
        error, thresh, cx, turn, area = cam.get_error(frame)
        #print(area)
        # ------ Sharp 90 turn ------
        '''
        if turn == "RIGHT":
            print("Executing Sharp Right")
            Turn(50,speed =  0.6, clockwise = True)

            #Reset PID to prevent integral windup from the sudden jump
            integral = 0
            last_error = 0

            cam.display(frame, cx)
            cv2.waitKey(1)
            continue
        elif turn == "LEFT":
            print("Executing Sharp Left")
            Turn(50, speed = 0.6, clockwise = False)
            #Reset PID to prevent integral windup from the sudden jump
            #integral = 0
            #last_error = 0

            cam.display(frame, cx)
            cv2.waitKey(1)
            continue
        '''

        # ------ Handle line lost ------


        if error is None or area < 300:
            lost_counter += 1
            #print(f"Line lost! Counter: {lost_counter}")
            
            # Use previous error for PIDe
            print("I'm so lost")
            current_error = last_error
            
            
            # If lost for a while, do a gentle spin to search
            if lost_counter > 0: #3
                if last_error >= 0:
                    #set_motor(ENA, IN1, IN2, search_speed) 
                    #set_motor(ENB, IN3, IN4, -search_speed+0.175) # search speed negative 
                    state.set_steering(search_speed, -search_speed+0.175)
                else:
                    #set_motor(ENA, IN1, IN2, -search_speed+0.175)
                    #set_motor(ENB, IN3, IN4, search_speed)
                    state.set_steering(-search_speed+0.175, search_speed)
                sleep(dt)
                continue
        else:
            lost_counter = 0
            current_error = error/2.5

        # ------ PID calculation ------
        integral += current_error * dt
        derivative = (current_error - last_error) / dt
        pid_output = Kp * current_error + Ki * integral + Kd * derivative
        #print(derivative)
        #Save the property scaled error for the next loop
        last_error = current_error

        # ------ Compute motor speeds ------
        left_speed = base_speed + pid_output / 1000   # scale PID to 0-1
        right_speed = base_speed - pid_output / 1000

        # ------ Set motor directions ------
        #set_motor(ENA, IN1, IN2, left_speed)
        #set_motor(ENB, IN3, IN4, right_speed)
        state.set_steering(left_speed, right_speed)
        # ------ Display for debugging ------
        #cam.display(frame, cx)
        #cam.display_draw(thresh)

        sleep(dt)
        '''
        # ------ Exit ------
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''

def thread_vision():
    global orb, matcher, templatesF
    #counter = 0
    symbol_cooldown = 0
    while state.running:
        frame = state.get_frame()
        if frame is None:
            sleep(0.05)
            continue
            
        #counter += 1
        #symbol = None
        if symbol_cooldown > 0:
            symbol_cooldown -=1
            sleep(0.1)
            continue
        
        #if symbol_cooldown > 0:
        #    symbol_cooldown -= 1
        #if counter % 5 == 0:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_shapes, shape_thresh = process_shapes(frame)

        for shape in detected_shapes:
            if shape['label']=='Arrow':
                print(f"Detected {shape['color']} Arrow pointing {shape['direction']}")
                state.set_override('STOP')
                sleep(1)
                state.set_override('NONE') 
                symbol_cooldown = 15 # Ignore symbols for 1.5 seconds (15 loops at 10fps)
                break  
            elif shape['label'] != 'Noise':
                print(f"Detected Shape: {shape['label']}")
                state.set_override('STOP')
                sleep(1)
                state.set_override('NONE') 
                symbol_cooldown = 15 # Ignore symbols for 1.5 seconds (15 loops at 10fps)
                break
            elif shape['label'] == 'Noise':
                orb_start_time = time()

                symbol = symbol_detect(frame_gray, templatesF, orb, matcher)

                orb_delay = (time() - orb_start_time) *1000
                print(f"WARNING: ORB Stalled motor for {orb_delay:.1f} ms!")
                if symbol !=None:
                    print(symbol)
                    state.set_override('STOP')
                    sleep(1)
                    state.set_override('NONE')
                    symbol_cooldown = 15 # Ignore symbols for 1.5 seconds (15 loops at 10fps)
                break
    sleep(0.1)

def thread_motor():
    while state.running:
        override = state.get_override()

        if override == 'STOP':
            stop()
        elif override == 'SPIN_360':
            Turn(360, speed = 0.5 , clockwise = True)   
        else:
            left,right = state.get_steering()
            set_motor(ENA, IN1, IN2, left)
            set_motor(ENB, IN3, IN4, right) 

        sleep(0.05)

print("Initializing System ...")
cam = Camera(resolution=(640,480), fps=60)
orb = cv2.ORB_create(nfeatures=2200, fastThreshold=15, nlevels=12, scaleFactor=1.2, patchSize=31)
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
