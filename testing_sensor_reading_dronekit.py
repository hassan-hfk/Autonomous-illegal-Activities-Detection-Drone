from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil
from ultralytics import YOLO
import picamera2
import cv2
import time
import RPi.GPIO as gpio

# GPIO Setup
gpio.setmode(gpio.BCM)

def NED_velocity(vX,vY,vZ):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
    0,
    0,0,
    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 
    3527, # Bitmask for Position + Velocity
    0,0,0, # Position
    vX,vY,vZ, # Velocity
    0,0,0, # acceleration
    0,0
)
    vehicle.send_mavlink(msg)
def NED(X,Y,Z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 
        3576, # Bitmask for Position + Velocity
        X,Y,Z, # Position
        0,0,0, # Velocity
        0,0,0, # acceleration
        0,0
    )
    vehicle.send_mavlink(msg)

def YAW(angle):
    if angle < 0:
        heading = -1
        angle = angle * -1
    else:
        heading = 1
    msg = vehicle.message_factory.command_long_encode(
        0,0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,    
        0,
        angle,
        20, # speed Deg/s
        heading, # CCW or CW
        1,  # Relative or Absolute
        0,0,0
    )
    vehicle.send_mavlink(msg)
    

# Constants
MOVING_AVG_THRESHOLD = 35
raw_distances = [[], [], [], []]
moving_avg = [0, 0, 0, 0]
movement_functions = ["left", "right", "forward", "backward"]
sensors = [[27, 17], [24, 23], [8, 25], [21, 20]]
obstacle_avoiding_distance = 70
speed = 0.75
# Setup GPIO pins
for trig, echo in sensors:
    gpio.setup(trig, gpio.OUT, initial=gpio.LOW)
    gpio.setup(echo, gpio.IN)

# Load YOLO model
#model = YOLO("best_ncnn_model")

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
    NED_velocity(0,speed,0)
def left(): 
    print("Left")
    NED_velocity(0,-speed,0)
def forward(): 
    NED_velocity(speed,0,0)
    print("Forward")
def backward(): 
    NED_velocity(-speed,0,0)
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
    vehicle = connect_vehicle()
    time.sleep(3)
    vehicle.simple_takeoff(1.5)
    while vehicle.location.global_relative_frame.alt < 0.9 * 1.5:
        print(vehicle.location.global_relative_frame.alt < 0.9 * 1.5)
        time.sleep(0.2)
    while True:
        for i in range(4):
            distance = ultrasonic(sensors[i][0], sensors[i][1])
            if distance > 10:
                raw_distances[i].append(distance)

        if len(raw_distances[0]) >= MOVING_AVG_THRESHOLD:
            for i in range(4):
                moving_avg[i] = int(sum(raw_distances[i][-MOVING_AVG_THRESHOLD:]) / MOVING_AVG_THRESHOLD)
            print("Moving Averages:", moving_avg)
            if moving_avg[0] < obstacle_avoiding_distance:
                movement_functions[0]()
            elif moving_avg[1] < obstacle_avoiding_distance:
                movement_functions[1]()
            elif moving_avg[2] < obstacle_avoiding_distance:
                movement_functions[2]()
            elif moving_avg[3] < obstacle_avoiding_distance:
                movement_functions[3]()    
            else:
                print("")
        else:
            print("Not Enough Samples...")
except Exception as e:
    print(f"Error occured : {e}")
    vehicle.mode = VehicleMode("LAND")
    print("LAND MODE")
except KeyboardInterrupt:
    print("Keyboard Interrupted")
    vehicle.mode=VehicleMode("LAND")
    print("LAND MODE")
    gpio.cleanup()
