from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import RPi.GPIO as GPIO
import time

def resetXY():

    GPIO.setmode(GPIO.BCM)      # Use Broadcom pin numbering
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP) # 17 is the x limit
    GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP) # 27 is the y limit

    kit = MotorKit(address=0x61)

    while GPIO.input(17) or GPIO.input(27):
        if GPIO.input(17):
            kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
        if GPIO.input(27):
            kit.stepper2.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
        time.sleep(0.1)
    
    print("XY reset completed")
    
    return