from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import time

kit = MotorKit(address=0x61) # 0x60 = Z motors 0x61 = XY motors (stepper 1 = x motor, stepper 2 = y motor)

# Step both motors forward together
for i in range(1000):
    # One step forward on stepper1
    kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
    # One step forward on stepper2
    # kit.stepper2.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
    time.sleep(0.001)

# Then step both backward
# for i in range(1000):
#     kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#     kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#     time.sleep(0.01)
