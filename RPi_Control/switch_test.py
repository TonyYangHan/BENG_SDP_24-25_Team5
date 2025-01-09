import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)      # Use Broadcom pin numbering
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    input_state_17 = GPIO.input(17)
    input_state_27 = GPIO.input(27)
    if input_state_17 == False and input_state_27 == False:
        print("Both pressed")
    
    elif input_state_17 == False:
        print("17 pressed")

    elif input_state_27 == False:
        print("27 pressed")        
    
    time.sleep(0.5)
