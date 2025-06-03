from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from ultralytics import YOLO
import picamera2
import cv2
import time
import RPi.GPIO as gpio

# GPIO Setup
gpio.setmode(gpio.BCM)

# Constants
MOVING_AVG_THRESHOLD = 25
raw_distances = [[], [], [], []]
moving_avg = [0, 0, 0, 0]
movement_functions = ["left", "right", "forward", "backward"]
# Sensor pins: [TRIG, ECHO] for  left, Right, Front, Back,
sensors = [[27, 17], [24, 23], [8, 25], [21, 20]]

# Setup GPIO pins
for trig, echo in sensors:
    gpio.setup(trig, gpio.OUT, initial=gpio.LOW)
    gpio.setup(echo, gpio.IN)

# Load YOLO model
model = YOLO("best_ncnn_model")

# Vehicle connection and arming
def connect_vehicle():
    vehicle = connect("/dev/serial0", baud=57600, timeout=90)
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode != "GUIDED":
        print(vehicle.mode)
        time.sleep(1)
    
    while not vehicle.is_armable:
        print("Not Ready to arm...")
        time.sleep(0.5)

    vehicle.armed = True
    while not vehicle.armed:
        print("Arming...")
        time.sleep(0.5)

    print("Vehicle Armed")
    return vehicle

# Movement functions (to be implemented)
def right(): 
    print("Right")
    
def left(): 
    print("Left")
def forward(): 
    print("Forward")
def backward(): 
    print("Backward")
def up(): print("Up")
def down(): print("Down")

def ultrasonic(TRIG, ECHO):
    gpio.output(TRIG, gpio.LOW)
    time.sleep(0.000002)
    gpio.output(TRIG, gpio.HIGH)
    time.sleep(0.00001)
    gpio.output(TRIG, gpio.LOW)

    pulse_start = None
    timeout = time.time() + 0.02
    while gpio.input(ECHO) == gpio.LOW:
        pulse_start = time.time()
        if pulse_start > timeout:
            return 400

    if pulse_start is None:
        return 400

    pulse_end = None
    timeout = time.time() + 0.02
    while gpio.input(ECHO) == gpio.HIGH:
        pulse_end = time.time()
        if pulse_end > timeout:
            return 400

    if pulse_end is None:
        return 400

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # cm
    return distance

# Main loop
try:
    while True:
        for i in range(4):
            distance = ultrasonic(sensors[i][0], sensors[i][1])
            if distance > 10:
                raw_distances[i].append(distance)

        if len(raw_distances[0]) >= MOVING_AVG_THRESHOLD:
            for i in range(4):
                moving_avg[i] = int(sum(raw_distances[i][-MOVING_AVG_THRESHOLD:]) / MOVING_AVG_THRESHOLD)
            print("Moving Averages:", moving_avg)
            if moving_avg[0] < 70:
                movement_functions[0]()
            elif moving_avg[1] < 70:
                movement_functions[1]()
            elif moving_avg[2] < 70:
                movement_functions[2]()
            else:
                movement_functions[3]()    
            
        else:
            print("Not Enough Samples...")
except Exception as e:
    print(f"Error occured : {e}")
    print("LAND MODE")
except KeyboardInterrupt:
    print("Keyboard Interrupted")
    print("LAND MODE")
    gpio.cleanup()
