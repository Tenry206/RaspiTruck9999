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
    a = -0.0071198

    # Y-intercept
    b = 0.431553
    
    #deviation offset
    delta = 0.098

    #overshoot factor
    alpha = 0.10

    #velocity from frequency
    velocity = a*(frequency/100) +b

    #PWM from velocity
    pwm = (velocity - b)/a
    pwm = max(0.3,min(pwm,0.8))

    #PWM from velocity
    pwm_L = pwm +delta
    pwm_R = pwm -delta

    #Distance compensation
    d_cmd = distance - alpha *velocity

    t_total = d_cmd / velocity

    Move(pwm_L,pwm_R)

    start_time = time()
    last_print = start_time
    
    while True:
        now = time()
        elapsed = now - start_time
        
        #print every 0.1 second
        if now - last_print >=0.1:
            d_travelled = velocity *elapsed
            d_travelled = min(d_travelled, d_cmd)

            print(f"Time: {elapsed: .1f}s | Distance: {d_travelled:.3f} m | Velocity: {velocity:.2f} m/s")
            last_print = now
        if elapsed >= t_total:
            break
        sleep(0.01)

    
    Stop()

move_distance(1, 500)