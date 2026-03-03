from gpiozero import OutputDevice, PWMOutputDevice as pwm
from time import sleep, time


# ------ Motor A GPIO setup ------


ENA = pwm(17)
IN1 = OutputDevice(22)
IN2 = OutputDevice(27)


# ------ Motor B GPIO setup ------


ENB = pwm(13)
IN3 = OutputDevice(5)
IN4 = OutputDevice(6)

def Stop():
    ENA.off()
    ENB.off()
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()


# ------ Move Function ------


def Move(left, right):
    # Stop if both speeds are zero
    if left == 0 and right == 0:
        Stop()
        return

    # Left motor backward
    if left < 0:
        ENB.value = abs(left)
        IN1.off()
        IN2.on()
    else:
        ENB.value = left
        IN1.on()
        IN2.off()

    # Right motor backward
    if right < 0:
        ENA.value = abs(right)
        IN3.off()
        IN4.on()
    else:
        ENA.value = right
        IN3.on()
        IN4.off()

# ------ Frequency Control ------


def move_distance(distance, frequency):
# model constants (from regression)
    # Gradient
    x = 0.05921

    # Y-intercept
    y = 2.247
    
    #deviation offset
    delta = 0.098

    #overshoot factor
    alpha = 0

    #pwm fix on 50% duty cycle
    pwm_base = 0.5

    #Apply pwm frequency
    ENA.frequency = frequency
    ENB.frequency = frequency

    #velocity from frequency
    time_per_1m = x*(frequency/100) + y

    #m/s
    velocity = 1/time_per_1m 

    #distance compensation
    distance_cmd = distance-alpha*time_per_1m
    total_time = distance_cmd * time_per_1m

    #PWM from velocity
    pwm_L = pwm_base +delta
    pwm_R = pwm_base -delta

    Move(pwm_L,pwm_R)

    start_time = time()
    last_print = start_time
    
    while True:
        now = time()
        elapsed = now - start_time
        
        #print every 0.1 second
        if now - last_print >=0.1:
            d_travelled = min(velocity *elapsed,distance_cmd)

            print(f"Time: {elapsed: .1f}s | Distance: {d_travelled:.3f} m | Velocity: {velocity:.2f} m/s")
            last_print = now
        if elapsed >= total_time:
            break
        sleep(0.01)

    
    Stop()

move_distance(1, 100)
