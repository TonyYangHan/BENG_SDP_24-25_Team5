from reset_motor import resetXY
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import time, subprocess

# coords are (x,y) system, viewing from the direction where the notes on the tape are inverted
# x difference between consecutive plates = 4000, also y_diff = 4000
# stepper motor 1 backward and stepper motor 2 forward goes to origin (negative direction)
coord = [(0,0), (950,0),(4950,0),(8950,0),(8950,4000),(4950,4000),(950,4000)]

# cam = Picamera2()

# cam.start()

kit = MotorKit(address = 0x61)

def from_to(start, end, step_wait):

    if start[0] < 0 or start[1] < 0 or end[0] < 0 or end[1] < 0:
        print("Negative coordinate received, aborting the move")
        return
    
    x_diff, y_diff = end[0] - start[0], end[1] - start[1]

    # Move along one direction at a time
    while x_diff > 0:
        kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
        x_diff -= 1
        time.sleep(step_wait)
    while x_diff < 0:
        kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
        x_diff += 1
        time.sleep(step_wait)

    while y_diff > 0:
        kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
        y_diff -= 1
        time.sleep(step_wait)
    while y_diff < 0:
        kit.stepper2.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
        y_diff += 1
        time.sleep(step_wait)
    return

def XY_cycle(step_wait = 0):
    resetXY()

    for i in range(len(coord)-1):
        from_to(coord[i], coord[i+1], step_wait)
        
        # cam.capture_file(f"temp_img_cache/well_{i+1}.jpg")
        command = ["libcamera-still", "-o", f"temp_img_cache/well_{i+1}.jpg"]
        subprocess.run(command, check=True)

        # time.sleep(5)
    
    return

# cam.stop()
