from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import time

kit = MotorKit(address=0x61) # 0x60 = Z motors 0x61 = XY motors (stepper 1 = x motor, stepper 2 = y motor)
# z stepper 1 = left motor stepper 2 = right motor

# X motor: FORWARD = away from origin, BACKWARD = toward the origin
# Y motor: FORWARD = towards stepper motor, BACKWARD = towards edge of table
# Z motor: FORWARD = down, BACKWARD = up



# Step both motors forward together
for i in range(2000):
    # One step forward on stepper1
   kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
   print(i)
   kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#    time.sleep(0.005)

# Then step both backward
# for i in range(1000):
#     kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#     kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#     time.sleep(0.01)

# for i in range(6):
#     time.sleep(0.005)  # Delay before each main loop iteration
#     if i == 0:
#         for j in range(950):
#             kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
#             print(j)
#         time.sleep(5)  # Pause after completing the steps

#     if i == 1 or i == 2:
#         for j in range(4000):
#             kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
#             print(j)
#         time.sleep(5)  # Pause after completing the steps

#     if i == 3:
#         for j in range(4000):
#             kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#             print(j)
#         time.sleep(5)  # Pause after completing the steps

#     if i == 4 or i == 5:
#         for j in range(4000):
#             kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#             print(j)
#         time.sleep(5)  # Pause after completing the steps

            

