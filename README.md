# Autonomous-illegal-Activities-Detection-Drone

This project implements a drone-based surveillance system capable of detecting illegal activities—specifically cigarette smoking—and identifying involved individuals using facial recognition. The system uses YOLOv8 for object detection, facial recognition for identity verification, and onboard sensors for real-time obstacle avoidance during autonomous navigation.

---

## ***Table of Contents***

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Facial Recognition](#facial-recognition)
- [Object Detection and Tracking](#object-detection-and-tracking)
- [Drone Obstacle Avoidance](#drone-obstacle-avoidance)
- [Hardware Components](#hardware-components)
- [Model Training Process](#model-training-process)
- [Output and Logging](#output-and-logging)
- [Future Improvements](#future-improvements)

---

## ***Project Overview***

This drone project is designed to detect and record illegal cigarette smoking in public or restricted areas. The drone autonomously navigates using sensors and actively monitors its environment using a camera. When both a cigarette and a human face are detected in the same frame, facial recognition is triggered to identify the person involved. The system logs the person's name and timestamp and stores a video clip of the activity for evidence.

---

## ***Core Features***

- Cigarette detection using YOLOv8 object detection
- Real-time face detection and facial recognition
- Automatic logging of name and timestamp on positive identification
- Onboard video recording of detected incidents
- Autonomous drone navigation with sensor-based obstacle avoidance

---

## ***System Architecture***

The drone system is composed of the following subsystems:

### 1. **Detection Pipeline**
- The camera continuously streams video to the onboard computer.
- YOLOv8 detects the presence of a **cigarette** and a **face** in each frame.
- If both objects are detected, the pipeline triggers the facial recognition module.

### 2. **Facial Recognition Module**
- The detected face is encoded and compared against a database of known individuals.
- If a match is found, the person's name is logged along with the current timestamp.
- A short video segment is saved as evidence.

### 3. **Navigation and Obstacle Avoidance**
- The drone navigates autonomously using onboard flight control and real-time obstacle sensing.
- Ultrasonic or infrared distance sensors detect obstacles in the drone's path.
- If an obstacle is detected within a threshold range, the drone automatically adjusts its trajectory to avoid collision.

### 4. **Logging and Storage**
- Detections are logged in a `.txt` file containing name and timestamp.
- Video evidence of each incident is stored locally on the onboard system.

---

## ***Facial Recognition***

Facial recognition is performed using the `face_recognition` Python library and is only activated when both a cigarette and a face are detected in the same frame. 

- Known individuals in the database include:
  - Hassan
  - Ali
  - Ibrahim
- Others are marked as "Unknown"

Each recognized face is matched using precomputed encodings. The recognition process is designed to be lightweight to ensure real-time performance on embedded systems like Raspberry Pi 4.

---

## ***Object Detection and Tracking***

Object detection is powered by YOLOv8 (Ultralytics) trained to recognize:
- Cigarettes
- Human faces

Once both objects are detected in the same frame:
- The system tracks the bounding boxes
- Maintains target identity across frames
- Triggers the facial recognition module

YOLOv8's high inference speed allows real-time detection and tracking on edge devices.

---

## ***Drone Obstacle Avoidance***

Obstacle avoidance is handled using hardware sensors:

- **Ultrasonic or infrared sensors** placed around the drone detect nearby objects.
- If an object is detected within a critical distance threshold:
  - The drone halts or reroutes its path.
  - Decision-making is handled by the onboard microcontroller or flight controller.
- This ensures safe and autonomous navigation without relying on camera-based depth estimation.

Obstacle avoidance operates independently from the detection pipeline to ensure constant flight safety.

---

## ***Hardware Components***

| Component             | Description                                  |
|----------------------|----------------------------------------------|
| Drone Frame           | Quadcopter with 25 min flight time, 1kg payload |
| Flight Controller     | Pixhawk / Betaflight-supported FC           |
| Onboard Computer      | Raspberry Pi 4 or Jetson Nano                |
| Camera Module         | Raspberry Pi Camera or USB webcam            |
| Sensors               | Ultrasonic or IR distance sensors for obstacle detection |
| Power System          | 4S Li-Po battery, ESCs, and Brushless Motors |

---

## ***Model Training Process***

### 1. **Cigarette and Face Detection (YOLOv8)**

- **Dataset**: Custom dataset containing labeled images with bounding boxes for cigarettes and faces
- **Annotation**: Performed using tools like LabelImg
- **Training Configuration**:
  - Base model: `yolov8n.pt`
  - Image size: `640x640`
  - Epochs: 50–100
  - Data augmentations: flipping, brightness adjustment, rotation
- **Framework**: Ultralytics YOLOv8 with PyTorch

### 2. **Facial Recognition**

- **Data Collection**: 5–10 clear frontal face images per known individual
- **Encoding**: Face encodings generated using the `face_recognition` library
- **Matching Threshold**: Tuned to minimize false positives and negatives
- Encoded faces are stored in memory or a file for fast lookup

---

## ***Output and Logging***

- **Log File**: Text log records the name and timestamp of every detection
