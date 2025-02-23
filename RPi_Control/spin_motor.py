from adafruit_motorkit import MotorKit
import gc
# Setting parameters for DC stir motors
th1,th2,th3,th4 = 0.01,0.01,-0.01,-0.01 # throttle for 4 motors (top two CW, bottom two CCW)
dc_addr = []

# We delete the object created each iteration to prevent excessive memory usage,
# which can be important during extended imaging period
# Question: Does the order of stopping the motor affect where the organoids situates eventully?
def spin_start():
    for addr in dc_addr:
        dc = MotorKit(address = addr)
        dc.motor1.throttle = th1
        dc.motor2.throttle = th2
        dc.motor3.throttle = th3
        dc.motor4.throttle = th4
        del dc
        gc.collect()
        

def spin_stop(): 
    for addr in dc_addr:
        dc = MotorKit(address = addr)
        dc.motor1.throttle = 0
        dc.motor2.throttle = 0
        dc.motor3.throttle = 0
        dc.motor4.throttle = 0
        del dc
        gc.collect()