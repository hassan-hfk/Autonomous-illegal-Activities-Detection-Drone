from ultralytics import YOLO
import cv2
import dlib
import numpy as np
import os
from datetime import datetime

# Setup
cam = cv2.VideoCapture(0)
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
video_writer = cv2.VideoWriter("test_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

# Load YOLO model
yolo_model = YOLO("best.pt")

# Load Dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("/home/hassan-hfk/yolo/illegal_activities_drone/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load known faces
known_encodings = []
known_names = []

known_faces_dir = "/home/hassan-hfk/yolo/illegal_activities_drone/known_faces"
for filename in os.listdir(known_faces_dir):
    name = filename.split('.')[0]  # Hassan.jpg → Hassan
    img_path = os.path.join(known_faces_dir, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb)
    if len(dets) > 0:
        shape = shape_predictor(img_rgb, dets[0])
        encoding = np.array(face_rec_model.compute_face_descriptor(img_rgb, shape))
        known_encodings.append(encoding)
        known_names.append(name)

# Log file
log_file = open("recognition_log.txt", "w")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = yolo_model(frame, conf=0.5)
    detected_classes = []
    face_boxes = []

    if results[0] and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = yolo_model.model.names[cls_id]
            detected_classes.append(cls_name)

            if cls_name == "face":
                face_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Check for both 'face' and 'cigarette'
    if "face" in detected_classes and "cigarette" in detected_classes:
        for (x1, y1, x2, y2) in face_boxes:
            face_img = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            dets = detector(face_rgb)
            if len(dets) > 0:
                shape = shape_predictor(face_rgb, dets[0])
                encoding = np.array(face_rec_model.compute_face_descriptor(face_rgb, shape))

                # Match with known faces
                distances = [np.linalg.norm(encoding - known_enc) for known_enc in known_encodings]
                min_dist = min(distances)
                best_match_idx = distances.index(min_dist)

                name = "Unknown"
                if min_dist < 0.6:
                    name = known_names[best_match_idx]

                # Draw label
                cv2.putText(frame, f"{name}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Log detection
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{timestamp} - {name}\n")
                log_file.flush()

    video_writer.write(frame)
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
video_writer.release()
log_file.close()
cv2.destroyAllWindows()
