from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import cv2, subprocess as sp

# 2500-3500 steps seem to be the best focus range (beads with water, cell phone light)
# 3150 from the bottom, or 1850 from top gives the best focus

# Initialize Motor Kit and Camera
kit = MotorKit(address = 0x60)

# Function to calculate image sharpness
def calculate_sharpness(ip):
    gray = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

# Autofocus routine
def autofocus(zoom_range = 1000, stride = 50):
    max_sharpness = 0
    best_position = 0

    for i in range(0, zoom_range, stride):  # Limit the number of steps

        path_name = f"z_images/z-pos_{i}.jpg"
        cmd = ["libcamera-still", "-o", path_name, "--nopreview", "--shutter", "18000", "--gain", 
               "20", "--timeout", "1500"]
        sp.run(cmd, check = True)
        sharpness = calculate_sharpness(path_name)
        
        print(f"Position: {i}, Sharpness: {sharpness}")

        # Check if this position is the best
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_position = i

        for i in range(stride):
            kit.stepper1.onestep(direction = stepper.BACKWARD)
            kit.stepper2.onestep(direction = stepper.BACKWARD)
    
    for i in range(zoom_range-best_position):
        kit.stepper1.onestep(direction = stepper.FORWARD)
        kit.stepper2.onestep(direction = stepper.FORWARD)

    print(f"Best focus position: {best_position}")

# Run autofocus
autofocus()
