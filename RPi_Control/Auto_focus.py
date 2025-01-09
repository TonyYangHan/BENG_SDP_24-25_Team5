# Make sure to reset z to lowest position before start running this code

import cv2, numpy as np
from time import sleep
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
from picamera2 import Picamera2

# Initialize the motor kit (Adafruit Motor HAT)
kit = MotorKit(address = 0x60)
picam2 = Picamera2()
picam2.start()

up , down, sgl_step = stepper.FORWARD, stepper.BACKWARD, stepper.SINGLE

# Function to calculate image sharpness using the Laplacian method
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


# Focus adjustment using motor
def adjust_focus(camera, step_delay=0.01, max_steps=1000, stride = 10):
    max_sharpness = 0
    optimal_position = 0

    for i in range(0, max_steps, stride):

        for _ in range(stride):
            kit.stepper1.onestep(direction = up, style = sgl_step)
            kit.stepper1.onestep(direction = up, style = sgl_step)
            sleep(step_delay)

        camera.capture_file(f"../focus_images/{i}.jpg")
        image = cv2.imread(f"../focus_images/{i}.jpg")
        sharpness = calculate_sharpness(image)

        if sharpness > max_sharpness:
            max_sharpness = sharpness
            optimal_position = i

    print(f"Optimal position: {optimal_position}, Max sharpness: {max_sharpness}")
    steps_back = max_steps - optimal_position

    for _ in range(steps_back):
        kit.stepper1.onestep(direction = down, style = sgl_step)
        kit.stepper1.onestep(direction = down, style = sgl_step)
        sleep(step_delay)
    
    print("Focus complete!")
    
adjust_focus(picam2)

picam2.stop()