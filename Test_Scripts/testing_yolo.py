from ultralytics import YOLO
import cv2
import dlib
import numpy as np
from datetime import datetime
import os

# Load YOLO model
model = YOLO("best.pt")  # Ensure best.pt has class names: 'face' and 'cig'

# Load dlib models
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize camera
cam = cv2.VideoCapture(0)

# Prepare known faces
def load_known_faces():
    known_encodings = []
    known_names = []
    for name in ['Hassan', 'Ibrahim', 'Ali']:
        path = f"known_faces/{name}.jpg"
        img = cv2.imread(path)
        if img is None:
            print(f"Image not found: {path}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = dlib.get_frontal_face_detector()(rgb)
        if dets:
            shape = shape_predictor(rgb, dets[0])
            encoding = face_encoder.compute_face_descriptor(rgb, shape)
            known_encodings.append(np.array(encoding))
            known_names.append(name)
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

def match_face(face_embedding):
    for i, known_embedding in enumerate(known_encodings):
        dist = np.linalg.norm(known_embedding - face_embedding)
        if dist < 0.6:
            return known_names[i]
    return "Unknown"

def get_face_embedding(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    rect = dlib.rectangle(0, 0, rgb.shape[1], rgb.shape[0])
    shape = shape_predictor(rgb, rect)
    return np.array(face_encoder.compute_face_descriptor(rgb, shape))

def log_detection(name):
    os.makedirs("logs", exist_ok=True)
    with open("logs/detected.txt", "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} - Detected: {name}\n"
        f.write(line)
        print(line.strip())

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 640))
    results = model(frame, conf=0.5)[0]

    face_boxes = []
    cig_boxes = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.model.names[cls_id]  # Properly access class names

            if cls_name == "face":
                face_boxes.append((x1, y1, x2, y2))
            elif cls_name == "cig":
                cig_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if face_boxes and cig_boxes:
        print("[INFO] Face and Cigarette detected. Starting recognition...")
        for (x1, y1, x2, y2) in face_boxes:
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                print("Face image too small or invalid.")
                continue

            try:
                embedding = get_face_embedding(face_img)
                name = match_face(embedding)
                print(f"[INFO] Recognition Result: {name}")
                log_detection(name)
                cv2.putText(frame, name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"[ERROR] Recognition failed: {e}")
                log_detection("Unknown")

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
