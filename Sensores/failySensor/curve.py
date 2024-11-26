import numpy as np
from PIL import Image, ImageDraw
import cv2
import predictors  # Import the EKF function from predictors.py
import random
import math

# Parameters
speed_ball_1 = 3  # Speed of Ball 1
speed_ball_2 = 1  # Speed of Ball 2
n_frames = 800  # Total number of frames
fps = 30  # Frames per second
video_size = (400, 400)  # Video resolution
output_path = "out.mp4"  # Output video file
history_limit = 30  # Number of previous positions to store
prediction_range = 15  # Number of future positions to predict
minimum_req = 3
collision_distance = 20  # Distance threshold for collision
angle = 0.02
width, height = video_size

# 15% of ball size
noise_strenght = 20 * 0.15


# Collision logs
predicted_collision_frames = []
actual_collision_frames = []

# Create a blank video writer with OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

# Initialize history for ball positions
history_ball_1 = []
history_ball_2 = []

# Initialize Ball 1 trajectory parameters
ball_1_pos = [0, 400]
phase = "up"  # Initial phase of Ball 1 trajectory

for frame in range(n_frames):
    # Create a blank image using PIL
    img = Image.new('RGB', video_size, 'black')
    draw = ImageDraw.Draw(img)

    # Ball 1 trajectory logic
    if phase == "up":
        ball_1_pos[1] -= speed_ball_1  # Move straight up
        if ball_1_pos[1] <= 350:  # Transition to turning right
            phase = "turning"

    elif phase == "turning":
        ball_1_pos[0] += speed_ball_1 * math.cos(angle*(frame + 200))  # Smooth curve to the right
        ball_1_pos[1] += speed_ball_1 * math.sin(angle*(frame + 200))  # Smooth curve to the right
        if ball_1_pos[0] >= 100:  # Transition to straight right
            phase = "right"

    elif phase == "right":
        ball_1_pos[0] += speed_ball_1  # Move straight right

    # Ball 2 trajectory (straight downward movement)
    ball_2_pos = [200, int(speed_ball_2 * frame)]

    # Add current positions to history
    if frame % 10 == 0:
        if random.randint(0, 2) != 0 :
            history_ball_1.append(tuple((ball_1_pos[0] + random.randrange(0,noise_strenght),ball_1_pos[1] + random.randrange(0,noise_strenght))))
        if random.randint(0, 2) != 0:
            history_ball_2.append(tuple((ball_2_pos[0] + random.randrange(0,noise_strenght),ball_2_pos[1] + random.randrange(0,noise_strenght))))

    # Limit history to the last `history_limit` positions
    if len(history_ball_1) > history_limit:
        history_ball_1.pop(0)
    if len(history_ball_2) > history_limit:
        history_ball_2.pop(0)

    # Predict positions using EKF
    predicted_positions_ball_1 = []
    predicted_positions_ball_2 = []

    if len(history_ball_1) >= minimum_req:
        predicted_positions_ball_1 = predictors.markov_model_predictor(np.array(history_ball_1).tolist(), prediction_range)

    if len(history_ball_2) >= minimum_req:
        predicted_positions_ball_2 = predictors.markov_model_predictor(np.array(history_ball_2).tolist(), prediction_range)

    # Check for predicted collisions
    if predicted_positions_ball_1 and predicted_positions_ball_2:
        for p1 in predicted_positions_ball_1:
            for p2 in predicted_positions_ball_2:
                if (191 - 30 <= p1[0] <= 191 + 30 and 147 - 30 <= p1[1] <= 147 + 30):
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
    actual_distance = np.linalg.norm(np.array(ball_1_pos) - np.array(ball_2_pos))
    if actual_distance < collision_distance and frame not in actual_collision_frames:
        actual_collision_frames.append(frame)
        print(f"Actual collision detected at frame {frame}", str(ball_1_pos))
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
    draw.ellipse((ball_1_pos[0] - ball_radius, ball_1_pos[1] - ball_radius,
                  ball_1_pos[0] + ball_radius, ball_1_pos[1] + ball_radius),
                 fill='red')
    draw.ellipse((ball_2_pos[0] - ball_radius, ball_2_pos[1] - ball_radius,
                  ball_2_pos[0] + ball_radius, ball_2_pos[1] + ball_radius),
                 fill='blue')


    # Draw the target circle at (191, 147) with radius 30
    target_center = (191, 147)
    radius = 30
    #draw.ellipse(
    #    (target_center[0] - radius, target_center[1] - radius,
    #     target_center[0] + radius, target_center[1] + radius),
    #    outline='yellow', width=2  # Yellow circle with a thickness of 2
    #)

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
