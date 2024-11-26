import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyautogui
import random

# Initialize YOLO model
model = YOLO("models/yolo11s.pt")  # Replace with your custom-trained YOLO model if available

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Get screen size dynamically using PyAutoGUI
screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

# Function to generate random colors for tracked IDs
def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Map to store colors for each track ID
color_map = {}

# Dictionary to store tracked positions
tracked_data = {}  # Format: {object_id: [(x, y, w, h), (x, y, w, h), None, ...]}

# Class ID for "person" (in COCO dataset, it's usually ID 0)
PLAYER_CLASS_ID = 0

# Main loop for tracking the player
with mss() as sct:
    try:
        while True:
            # Capture the screen
            screenshot = sct.grab(screen_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLO detection
            results = model.predict(frame)
            detections = []

            # Filter detections for the "person" class
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                confidence = float(r.conf[0])
                class_id = int(r.cls[0])

                # Only add detections for the "person" class
                if confidence > 0.5 and class_id == PLAYER_CLASS_ID:  # Adjust confidence threshold if needed
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

            # Update DeepSORT with player detections
            tracks = tracker.update_tracks(detections, frame=frame)

            # Track the IDs processed in this frame
            active_track_ids = set()

            # Process each track (only tracks associated with the "person" class)
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                active_track_ids.add(track_id)
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                w, h = x2 - x1, y2 - y1

                # Assign a color to the track ID
                if track_id not in color_map:
                    color_map[track_id] = get_random_color()

                # Log position
                if track_id not in tracked_data:
                    tracked_data[track_id] = []
                tracked_data[track_id].append((x1, y1, w, h))

                # Draw the bounding box and ID
                color = color_map[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add None for objects not detected in this frame
            for track_id in tracked_data.keys():
                if track_id not in active_track_ids:
                    tracked_data[track_id].append(None)

            # Display the frame
            cv2.imshow("Person Tracking", frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Write tracked data to a file
        with open("data.txt", "w") as file:
            for obj_id, positions in tracked_data.items():
                file.write(f"Object {obj_id} -> {positions}\n")

        cv2.destroyAllWindows()
