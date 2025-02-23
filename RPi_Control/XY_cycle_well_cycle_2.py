from reset_motor import resetXY
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import time, subprocess

# coords are (x,y) system, viewing from the direction where the notes on the tape are inverted
# x difference between consecutive plates = 4000, also y_diff = 4000
# stepper motor 1 backward and stepper motor 2 forward goes to origin (negative direction)

# Imaging area: 475 steps along y, 410 steps along x(+-30 steps)
# coord_center = [(0,0), (950,0),(4950,0),(8950,0),(8950,4000),(4950,4000),(950,4000)]
# coord = [(0,0), (100,1000),(4100,1000),(8100,1000),(8100,5000),(4100,5000),(100,5000)]
coord = [(0,0), (400,1000)]
fov = (440,505) # Slightly larger than actual FOV to ensure images does not overlap

kit = MotorKit(address = 0x61)

save_dir = "field_test/"

def image_capture(file_name):

    command = ["libcamera-still", "-o", file_name, "--nopreview", 
                   "--shutter", "18000", "--gain", "20"]
    subprocess.run(command, check=True)

    return


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


def move_pos_fov(num_fovs, count, well_num, wait = 0.001):
    for _ in range(num_fovs):
        from_to((0,0), (fov[0],0), wait)
        image_capture(save_dir + f"well_{well_num}_img_{count}.jpg")
        count += 1

    return count

def move_neg_fov(num_fovs, count, well_num, wait = 0.001):
    for _ in range(num_fovs):
        from_to((fov[0],0), (0,0), wait)
        image_capture(save_dir + f"well_{well_num}_img_{count}.jpg")
        count += 1
    
    return count

def move_y_fov(count, well_num, wait = 0.001):

    from_to((0,0), (0,fov[1]), wait)
    image_capture(save_dir + f"well_{well_num}_img_{count}.jpg")
    count += 1
    
    return count

# y_fov should be even number to save some effort (or x movement is required to reset)
def image_well(well_num, wait):
    count = 1
    x_fov, y_fov = 6,4
    image_capture(save_dir + f"well_{well_num}_img_{count}.jpg")
    count += 1
    for i in range(y_fov):
        if i%2:
            count = move_neg_fov(x_fov, count, well_num, wait)
            
        else:
            count = move_pos_fov(x_fov, count, well_num, wait)
        
        count = move_y_fov(count, well_num, wait)
    
    from_to((0,fov[1]*y_fov), (0,0), wait)
        
    return

def XY_cycle(step_wait = 0):
    resetXY(step_wait)

    for i in range(len(coord)-1):
        from_to(coord[i], coord[i+1], step_wait)
        image_well(i+1, step_wait)
    
    return

if __name__ == "__main__":
    XY_cycle()