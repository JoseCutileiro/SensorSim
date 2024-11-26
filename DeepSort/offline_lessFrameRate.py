import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

# Initialize YOLO model
model = YOLO("models/yolo11s.pt")  # Replace with your custom-trained YOLO model if available

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Function to generate random colors for tracked IDs
def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Map to store colors for each track ID
color_map = {}

# Dictionary to store tracked positions
tracked_data = {}  # Format: {object_id: [(x, y, w, h), (x, y, w, h), None, ...]}

# Class ID for "person" (in COCO dataset, it's usually ID 0)
PLAYER_CLASS_ID = 0

# Path to the offline video
video_path = "offline_vid/basic_vid1080p.mkv"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format

# Output video writer
output_path = "out.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps // 5, (width, height))  # Divide FPS by 5

# Frame skip value to reduce frame rate
frame_skip = 5  # Process every 5th frame
frame_counter = 0  # Frame counter

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Skip frames to reduce frame rate
        if frame_counter % frame_skip != 0:
            frame_counter += 1
            continue

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

        # Write the frame with bounding boxes to the output video
        out.write(frame)

        # Increment frame counter
        frame_counter += 1

finally:
    # Release resources and write tracked data to a file
    cap.release()
    out.release()
    with open("data.txt", "w") as file:
        for obj_id, positions in tracked_data.items():
            file.write(f"Object {obj_id} -> {positions}\n")

print(f"Processing complete. Output saved to {output_path}")
