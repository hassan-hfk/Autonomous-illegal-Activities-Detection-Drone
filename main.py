from ultralytics import YOLO
import cv2
import dlib
import numpy as np
import os
import math
from picamera2 import Picamera2
from libcamera import Transform
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time
import RPi.GPIO as gpio

gpio.setmode(gpio.BCM)

def NED_velocity(vX, vY, vZ):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 
        3527, 0, 0, 0, vX, vY, vZ, 0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)

def YAW(angle):
    heading = -1 if angle < 0 else 1
    angle = abs(angle)
    msg = vehicle.message_factory.command_long_encode(
        0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0, angle, 20, heading, 1, 0, 0, 0
    )
    vehicle.send_mavlink(msg)

def connect_vehicle():
    vehicle = connect("/dev/serial0", baud=57600, timeout=90)
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode != "GUIDED":
        print("Waiting for GUIDED mode...")
        time.sleep(1)
    while not vehicle.is_armable:
        print("Waiting to be armable...")
        time.sleep(0.5)
    vehicle.armed = True
    while not vehicle.armed:
        print("Arming...")
        time.sleep(0.5)
    print("Vehicle Armed")
    return vehicle

def ultrasonic(TRIG, ECHO):
    gpio.output(TRIG, gpio.LOW)
    time.sleep(0.000002)
    gpio.output(TRIG, gpio.HIGH)
    time.sleep(0.00001)
    gpio.output(TRIG, gpio.LOW)
    timeout = time.time() + 0.02
    while gpio.input(ECHO) == gpio.LOW:
        if time.time() > timeout:
            return 400
    pulse_start = time.time()
    timeout = time.time() + 0.02
    while gpio.input(ECHO) == gpio.HIGH:
        if time.time() > timeout:
            return 400
    pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    return pulse_duration * 17150

model = YOLO("best_ncnn_model")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_known_faces():
    encodings, names = [], []
    for name in ['Hassan', 'Ibrahim', 'Ali']:
        path = f"/home/pi/Desktop/face_recognition/illegal_activities_drone/known_faces/{name}.jpg"
        img = cv2.imread(path)
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = dlib.get_frontal_face_detector()(rgb)
        if faces:
            shape = shape_predictor(rgb, faces[0])
            encoding = face_encoder.compute_face_descriptor(rgb, shape)
            encodings.append(np.array(encoding))
            names.append(name)
    return encodings, names

known_encodings, known_names = load_known_faces()

def match_face(embedding):
    for i, known_embedding in enumerate(known_encodings):
        if np.linalg.norm(known_embedding - embedding) < 0.6:
            return known_names[i]
    return "Unknown"

def get_face_embedding(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    rect = dlib.rectangle(0, 0, rgb.shape[1], rgb.shape[0])
    shape = shape_predictor(rgb, rect)
    return np.array(face_encoder.compute_face_descriptor(rgb, shape))

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (3840, 2640)})
config["transform"] = Transform(vflip=1)
picam2.configure(config)
picam2.start()

sensors = [[27, 17], [24, 23], [8, 25], [21, 20]]
for trig, echo in sensors:
    gpio.setup(trig, gpio.OUT, initial=gpio.LOW)
    gpio.setup(echo, gpio.IN)

MOVING_AVG_THRESHOLD = 10
obstacle_avoiding_distance = 80
raw_distances = [[] for _ in range(4)]
moving_avg = [0] * 4
speed = 0.45

def left(): NED_velocity(0, speed, 0)
def right(): NED_velocity(0, -speed, 0)
def forward(): NED_velocity(-speed, 0, 0)
def backward(): NED_velocity(speed, 0, 0)
movement_functions = [left, right, forward, backward]

try:
    vehicle = connect_vehicle()
    time.sleep(3)
    vehicle.simple_takeoff(1.5)
    while vehicle.location.global_relative_frame.alt < 1.35:
        time.sleep(0.2)

    while True:
        frame = picam2.capture_array()
        frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model.track(source=frame, conf=0.5, persist=True)

        face_boxes, cig_boxes = [], []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (255, 0, 0) if cls_name == "face" else (0, 0, 255)
                label = "FACE" if cls_name == "face" else "CIGARETTE"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if cls_name == "face":
                    face_boxes.append((x1, y1, x2, y2))
                elif cls_name == "cig":
                    cig_boxes.append((x1, y1, x2, y2))

            if face_boxes and cig_boxes:
                for (x1, y1, x2, y2) in face_boxes:
                    try:
                        face_img = frame[y1:y2, x1:x2]
                        embedding = get_face_embedding(face_img)
                        name = match_face(embedding)
                        cv2.putText(frame, f"{name}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"Smoking Detected: {name}")
                    except Exception as e:
                        print("Recognition error:", e)

        for i in range(4):
            dist = ultrasonic(*sensors[i])
            if dist > 10:
                raw_distances[i].append(dist)

        if len(raw_distances[0]) >= MOVING_AVG_THRESHOLD:
            for i in range(4):
                moving_avg[i] = int(sum(raw_distances[i][-MOVING_AVG_THRESHOLD:]) / MOVING_AVG_THRESHOLD)

            if moving_avg[0] < obstacle_avoiding_distance:
                left()
            elif moving_avg[1] < obstacle_avoiding_distance:
                right()
            elif moving_avg[2] < obstacle_avoiding_distance:
                forward()
            elif moving_avg[3] < obstacle_avoiding_distance:
                backward()
            else:
                NED_velocity(0, 0, 0)
        else:
            print("Collecting distance data...")

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
    vehicle.mode = VehicleMode("LAND")
except KeyboardInterrupt:
    print("Keyboard Interrupted")
    vehicle.mode = VehicleMode("LAND")
finally:
    gpio.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
