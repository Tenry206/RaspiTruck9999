from gpiozero import OutputDevice, PWMOutputDevice as pwm
from time import sleep

# ------ Motor A GPIO setup ------


ENA = pwm(17)
IN1 = OutputDevice(22)
IN2 = OutputDevice(27)


# ------ Motor B GPIO setup ------


ENB = pwm(13)
IN3 = OutputDevice(5)
IN4 = OutputDevice(6)


def Forward():
    ENA.value = 1
    ENB.value = 1
    IN1.on()
    IN2.off()
    IN3.on()
    IN4.off()


def Stop():
    ENA.off()
    ENB.off()
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()


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
"""
Move(1,1)
sleep(1.72)
Stop()
"""
#Frequency vs velocity

def frequencyTest(frequency):
    ENA.frequency = frequency
    ENB.frequency = frequency

    ENA.value = 0.5   # 50% duty cycle
    ENB.value = 0.5

    IN1.on()
    IN2.off()
    IN3.on()
    IN4.off()

frequencyTest(2000)
sleep(10)
Stop()