import numpy as np
import random
import predictors  # Import the EKF or Kalman filter predictor function from predictors.py

# Parameters
radius = 200  # Circle radius (pixels)
center = (250, 250)  # Center of the circle
speed = 0.01  # Angular speed in radians per frame
n_frames = 800  # Total number of frames
history_limit = 30  # Number of previous positions to store
prediction_range = 15  # Number of future positions to predict
minimum_req = 3
collision_distance = 10  # Distance threshold for collision

noise_strength = 20 * 0.15

# Function to simulate one run
def simulate_run():
    predicted_collision_frames = []
    actual_collision_frames = []

    # Initialize history for ball positions
    history_ball_1 = []
    history_ball_2 = []

    for frame in range(n_frames):
        # Calculate angles for the current frame
        angle1 = speed * frame + 0.03
        angle2 = -speed * frame - 0.03

        # Compute positions of the two balls
        ball1_pos = (center[0] + int(radius * np.cos(angle1)),
                     center[1] + int(radius * np.sin(angle1)))
        ball2_pos = (center[0] + int(radius * np.cos(angle2)),
                     center[1] + int(radius * np.sin(angle2)))

        # Add current positions to history
        if frame % 10 == 0 and random.randint(0,2) != 0:
            history_ball_1.append(
                (ball1_pos[0] + random.randrange(0, noise_strength), ball1_pos[1] + random.randrange(0, noise_strength))
            )
            history_ball_2.append(
                (ball2_pos[0] + random.randrange(0, noise_strength), ball2_pos[1] + random.randrange(0, noise_strength))
            )

        # Limit history to the last `history_limit` positions
        if len(history_ball_1) > history_limit:
            history_ball_1.pop(0)
        if len(history_ball_2) > history_limit:
            history_ball_2.pop(0)

        # Predict positions using Kalman filter
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
                    if 35 <= p1[0] <= 65 and 235 <= p1[1] <= 265:
                        distance = np.linalg.norm(np.array(p1) - np.array(p2))
                        if distance < collision_distance and frame not in predicted_collision_frames:
                            predicted_collision_frames.append(frame)

        # Check for actual collisions
        actual_distance = np.linalg.norm(np.array(ball1_pos) - np.array(ball2_pos))
        if actual_distance < collision_distance and frame not in actual_collision_frames:
            actual_collision_frames.append(frame)
            break

    return predicted_collision_frames, actual_collision_frames

# Run the simulation 10 times and compute averages
all_predicted = []
all_actual = []
for _ in range(50):
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
