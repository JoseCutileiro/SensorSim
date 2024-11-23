import numpy as np
import random
import math
import predictors  # Import the EKF or polynomial regression predictor function from predictors.py

# Parameters
speed_ball_1 = 3  # Speed of Ball 1
speed_ball_2 = 1  # Speed of Ball 2
n_frames = 800  # Total number of frames
video_size = (400, 400)  # Video resolution
history_limit = 30  # Number of previous positions to store
prediction_range = 15  # Number of future positions to predict
minimum_req = 3
collision_distance = 20  # Distance threshold for collision
angle = 0.02
width, height = video_size

# 15% of ball size
noise_strength = 20 * 0.15

# Function to simulate one run
def simulate_run():
    predicted_collision_frames = []
    actual_collision_frames = []

    # Initialize history for ball positions
    history_ball_1 = []
    history_ball_2 = []

    # Initialize Ball 1 trajectory parameters
    ball_1_pos = [0, 400]
    phase = "up"  # Initial phase of Ball 1 trajectory

    for frame in range(n_frames):
        # Ball 1 trajectory logic
        if phase == "up":
            ball_1_pos[1] -= speed_ball_1  # Move straight up
            if ball_1_pos[1] <= 350:  # Transition to turning right
                phase = "turning"

        elif phase == "turning":
            ball_1_pos[0] += speed_ball_1 * math.cos(angle * (frame + 200))  # Smooth curve to the right
            ball_1_pos[1] += speed_ball_1 * math.sin(angle * (frame + 200))  # Smooth curve to the right
            if ball_1_pos[0] >= 100:  # Transition to straight right
                phase = "right"

        elif phase == "right":
            ball_1_pos[0] += speed_ball_1  # Move straight right

        # Ball 2 trajectory (straight downward movement)
        ball_2_pos = [200, int(speed_ball_2 * frame)]

        # Add current positions to history
        if frame % 10 == 0:
            history_ball_1.append(
                (ball_1_pos[0] + random.randrange(0, noise_strength), ball_1_pos[1] + random.randrange(0, noise_strength))
            )
            history_ball_2.append(
                (ball_2_pos[0] + random.randrange(0, noise_strength), ball_2_pos[1] + random.randrange(0, noise_strength))
            )

        # Limit history to the last `history_limit` positions
        if len(history_ball_1) > history_limit:
            history_ball_1.pop(0)
        if len(history_ball_2) > history_limit:
            history_ball_2.pop(0)

        # Predict positions using polynomial regression
        predicted_positions_ball_1 = []
        predicted_positions_ball_2 = []

        if len(history_ball_1) >= minimum_req:
            predicted_positions_ball_1 = predictors.markov_model_predictor(
                np.array(history_ball_1).tolist(), prediction_range
            )

        if len(history_ball_2) >= minimum_req:
            predicted_positions_ball_2 = predictors.markov_model_predictor(
                np.array(history_ball_2).tolist(), prediction_range
            )

        # Check for predicted collisions
        if predicted_positions_ball_1 and predicted_positions_ball_2:
            for p1 in predicted_positions_ball_1:
                for p2 in predicted_positions_ball_2:
                    if 191 - 30 <= p1[0] <= 191 + 30 and 147 - 30 <= p1[1] <= 147 + 30:
                        distance = np.linalg.norm(np.array(p1) - np.array(p2))
                        if distance < collision_distance and frame not in predicted_collision_frames:
                            predicted_collision_frames.append(frame)

        # Check for actual collisions
        actual_distance = np.linalg.norm(np.array(ball_1_pos) - np.array(ball_2_pos))
        if actual_distance < collision_distance and frame not in actual_collision_frames:
            actual_collision_frames.append(frame)
            break

    return predicted_collision_frames, actual_collision_frames

# Run the simulation 10 times and compute averages
all_predicted = []
all_actual = []
for _ in range(100):
    predicted, actual = simulate_run()
    if predicted:
        all_predicted.append(predicted[0])  # First predicted collision frame
    if actual:
        all_actual.append(actual[0])  # First actual collision frame

# Calculate averages
average_predicted = np.mean(all_predicted) if all_predicted else None
average_actual = np.mean(all_actual) if all_actual else None

# Print results
print("Predicted Collision Frames (first occurrences):", all_predicted)
print("Actual Collision Frames (first occurrences):", all_actual)
print("Average Predicted Collision Frame:", average_predicted)
print("Average Actual Collision Frame:", average_actual)
