import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyautogui
import random
import threading
import time

# Initialize YOLO model
model = YOLO("models/yolo11s.pt")
model.half()  # Use FP16 for faster inference

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=15, n_init=3, nn_budget=50)  # Smaller thresholds for speed

# Get screen size dynamically using PyAutoGUI
screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

# Function to generate random colors for tracked IDs
def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Map to store colors for each track ID
color_map = {}
tracked_data = {}  # Format: {object_id: [(x, y, w, h), ...]}
PLAYER_CLASS_ID = 0

# Function to capture screen
def capture_screen():
    with mss() as sct:
        while True:
            screenshot = sct.grab(screen_region)
            frame_queue.append(np.array(screenshot))
            time.sleep(0.01)  # Prevent overloading the queue

# Queue for frame sharing
frame_queue = []
thread = threading.Thread(target=capture_screen, daemon=True)
thread.start()

try:
    while True:
        if not frame_queue:
            continue

        # Get the latest frame
        frame = frame_queue.pop(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize frame for faster inference
        input_frame = cv2.resize(frame, (640, 640))

        # Run YOLO detection
        results = model.predict(input_frame)
        detections = []

        # Filter detections for the "person" class
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            confidence = float(r.conf[0])
            class_id = int(r.cls[0])
            if confidence > 0.5 and class_id == PLAYER_CLASS_ID:
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

        # Update DeepSORT with player detections
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if track_id not in color_map:
                color_map[track_id] = get_random_color()

            color = color_map[track_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        cv2.imshow("Person Tracking", cv2.resize(frame, (960, 540)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
