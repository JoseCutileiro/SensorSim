import numpy as np
from PIL import Image, ImageDraw
import cv2
import predictors  # Import the EKF function from predictors.py
import random

# Parameters
radius = 200  # Circle radius (pixels)
center = (250, 250)  # Center of the circle
speed = 0.01  # Angular speed in radians per frame
n_frames = 800  # Total number of frames
fps = 30  # Frames per second
video_size = (500, 500)  # Video resolution
output_path = "out.mp4"  # Output video file
history_limit = 30  # Number of previous positions to store
prediction_range = 15  # Number of future positions to predict
minimum_req = 3
collision_distance = 10  # Distance threshold for collision

# Collision logs
predicted_collision_frames = []
actual_collision_frames = []

noise_strenght = 20 * 0.15

# Create a blank video writer with OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

# Initialize history for ball positions
history_ball_1 = []
history_ball_2 = []


for frame in range(n_frames):
    # Create a blank white image using PIL
    img = Image.new('RGB', video_size, 'black')
    draw = ImageDraw.Draw(img)

    # Calculate angles for the current frame
    angle1 = speed * frame + 0.03
    angle2 = -speed * frame - 0.03

    # Compute positions of the two balls
    ball1_pos = (center[0] + int(radius * np.cos(angle1)),
                 center[1] + int(radius * np.sin(angle1)))
    ball2_pos = (center[0] + int(radius * np.cos(angle2)),
                 center[1] + int(radius * np.sin(angle2)))

    # Add current positions to history
    if (frame % 10 == 0):
        # 1/3 de chances de ser ignorado
        if (random.randint(0,2)):
            history_ball_1.append((ball1_pos[0] + random.randrange(0,noise_strenght),ball1_pos[1] + random.randrange(0,noise_strenght)))
        if (random.randint(0,2)):
            history_ball_2.append((ball2_pos[0] + random.randrange(0,noise_strenght),ball2_pos[1] + random.randrange(0,noise_strenght)))

    # Limit history to the last `history_limit` positions
    if len(history_ball_1) > history_limit:
        history_ball_1.pop(0)
    if len(history_ball_2) > history_limit:
        history_ball_2.pop(0)

    # Predict positions using EKF
    predicted_positions_ball_1 = []
    predicted_positions_ball_2 = []

    if len(history_ball_1) >= minimum_req:
        predicted_positions_ball_1 = predictors.ekf(np.array(history_ball_1).tolist(), prediction_range)

    if len(history_ball_2) >= minimum_req:
        predicted_positions_ball_2 = predictors.ekf(np.array(history_ball_2).tolist(), prediction_range)

    # Check for predicted collisions
    if predicted_positions_ball_1 and predicted_positions_ball_2:
        for p1 in predicted_positions_ball_1:
            for p2 in predicted_positions_ball_2:
                if (35 <= p1[0] <= 65 and 235 <= p1[1] <= 265):
                    distance = np.linalg.norm(np.array(p1) - np.array(p2))
                    if distance < collision_distance and frame not in predicted_collision_frames:
                        predicted_collision_frames.append(frame)
                        print(f"True positive at frame: {frame}")


    # Draw predicted positions
    for pos in predicted_positions_ball_1:
        draw.ellipse((pos[0] - 3, pos[1] - 3, pos[0] + 3, pos[1] + 3), fill='darkred')

    for pos in predicted_positions_ball_2:
        draw.ellipse((pos[0] - 3, pos[1] - 3, pos[0] + 3, pos[1] + 3), fill='darkblue')

    # Check for actual collisions
    actual_distance = np.linalg.norm(np.array(ball1_pos) - np.array(ball2_pos))
    if actual_distance < collision_distance and frame not in actual_collision_frames:
        actual_collision_frames.append(frame)
        print(f"Actual collision detected at frame {frame}")
        break

    # Draw the trails using smaller balls
    for idx, pos in enumerate(history_ball_1):
        alpha = (idx + 1) / history_limit  # Gradual fade effect
        color = (int(255 * alpha), 0, 0)  # Red fade
        draw.ellipse((pos[0] - 5, pos[1] - 5, pos[0] + 5, pos[1] + 5), fill=color)

    for idx, pos in enumerate(history_ball_2):
        alpha = (idx + 1) / history_limit  # Gradual fade effect
        color = (0, 0, int(255 * alpha))  # Blue fade
        draw.ellipse((pos[0] - 5, pos[1] - 5, pos[0] + 5, pos[1] + 5), fill=color)

    # Draw the current positions of the balls
    ball_radius = 10
    draw.ellipse((ball1_pos[0] - ball_radius, ball1_pos[1] - ball_radius,
                  ball1_pos[0] + ball_radius, ball1_pos[1] + ball_radius),
                 fill='red')
    draw.ellipse((ball2_pos[0] - ball_radius, ball2_pos[1] - ball_radius,
                  ball2_pos[0] + ball_radius, ball2_pos[1] + ball_radius),
                 fill='blue')

    # Convert the image to a numpy array for OpenCV
    frame_array = np.array(img)
    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

    # Write the frame to the video
    video_writer.write(frame_array)

# Release the video writer
video_writer.release()

# Print collision logs
print("Predicted Collision Frames:", predicted_collision_frames)
print("Actual Collision Frames:", actual_collision_frames)
print(f"Video saved as {output_path}")
