from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

# 0x60 = Z motors 0x61 = XY motors (stepper 1 = x motor, stepper 2 = y motor)
# z stepper 1 = left motor stepper 2 = right motor

# X motor: FORWARD = away from origin, BACKWARD = toward the origin
# Y motor: FORWARD = towards stepper motor, BACKWARD = towards edge of table
# Z motor: FORWARD = down, BACKWARD = up

# Imaging area: 475 steps along y, 410 steps along x(+-30 steps)

kit = MotorKit(address = 0x61)

steps = 600

for i in range(steps):
	kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
	kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
	print(i)

