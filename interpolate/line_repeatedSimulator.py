import numpy as np
import random
import predictors  # Import the EKF function from predictors.py

# Parameters
speed = 2  # Speed of movement (pixels per frame)
n_frames = 800  # Total number of frames
video_size = (500, 500)  # Video resolution
history_limit = 30  # Number of previous positions to store
prediction_range = 15  # Number of future positions to predict
minimum_req = 3
collision_distance = 10  # Distance threshold for collision

width, height = video_size
noise_strength = 20 * 0.15


from scipy.interpolate import UnivariateSpline

def interpolate_missing(history):
    """
    Interpolates missing points in a trajectory while ensuring the resulting trajectory is consistent.
    If the interpolation results in a trajectory that doesn't make sense (e.g., reversing direction),
    it falls back to a polynomial or spline smoothing approach.

    Parameters:
        history (list): List of tuples (x, y) or None representing the trajectory.

    Returns:
        interpolated (list): List of tuples with interpolated values or None where interpolation is impossible.
    """
    interpolated = []
    valid_points = [(i, p) for i, p in enumerate(history) if p is not None]
    
    if len(valid_points) < 3:
        # Not enough points to interpolate
        return history  # Return as is, with None for missing points
    
    indices, points = zip(*valid_points)
    x_coords, y_coords = zip(*points)
    
    # Interpolation or smoothing approach
    try:
        # Use a cubic spline for smoothing
        spline_x = UnivariateSpline(indices, x_coords, k=2, s=0.5)
        spline_y = UnivariateSpline(indices, y_coords, k=2, s=0.5)
    except Exception as e:
        print(f"Spline fitting failed: {e}")
        return history  # Return as is
    
    for i in range(len(history)):
        if history[i] is not None:
            interpolated.append(history[i])  # Keep existing points
        else:
            # Predict using the spline
            interpolated_x = spline_x(i)
            interpolated_y = spline_y(i)
            interpolated_point = (interpolated_x, interpolated_y)
            
            # Validate the predicted point
            if len(interpolated) > 1 and interpolated[-1] is not None and interpolated[-2] is not None:
                last_valid_point = interpolated[-1]
                second_last_valid_point = interpolated[-2]
                direction_vector = np.array(last_valid_point) - np.array(second_last_valid_point)
                new_vector = np.array(interpolated_point) - np.array(last_valid_point)
                
                # Check if the new point reverses direction or deviates significantly
                dot_product = np.dot(direction_vector, new_vector)
                if dot_product < 0:  # Opposite direction
                    interpolated.append(None)  # Discard prediction
                    continue

            interpolated.append((interpolated_x, interpolated_y))
    
    return interpolated

# Function to simulate one run
def simulate_run():
    predicted_collision_frames = []
    actual_collision_frames = []

    # Initialize history for ball positions
    history_ball_1 = []
    history_ball_2 = []

    for frame in range(n_frames):
        # Compute positions of the two balls
        ball_1_pos = (int(speed * frame), int(speed * frame))  # Top-left to bottom-right
        ball_2_pos = (width - int(speed * frame), height - int(speed * frame))  # Bottom-right to top-left

        # Add current positions to history
        if frame % 10 == 0:
            if (random.randint(0,2) != 0):
                history_ball_1.append(
                    (ball_1_pos[0] + random.randrange(0, noise_strength), ball_1_pos[1] + random.randrange(0, noise_strength))
                )
            else:
                history_ball_1.append(None)
            if (random.randint(0,2) != 0):
                history_ball_2.append(
                    (ball_2_pos[0] + random.randrange(0, noise_strength), ball_2_pos[1] + random.randrange(0, noise_strength))
                )
            else:
                history_ball_2.append(None)

        # Limit history to the last `history_limit` positions
        if len(history_ball_1) > history_limit:
            history_ball_1.pop(0)
        if len(history_ball_2) > history_limit:
            history_ball_2.pop(0)

        # Predict positions using EKF
        predicted_positions_ball_1 = []
        predicted_positions_ball_2 = []

        if len(history_ball_1) >= minimum_req:
            # Contar entradas válidas (não-None) no histórico

            interpolated_history_ball_1 = interpolate_missing(history_ball_1)
            smoothed_history_ball_1 = [p for p in interpolated_history_ball_1 if p is not None]  # Remove None
            if len(smoothed_history_ball_1) >= minimum_req:
                predicted_positions_ball_1 = predictors.ekf(smoothed_history_ball_1, prediction_range)

        if len(history_ball_2) >= minimum_req:
            # Contar entradas válidas (não-None) no histórico

            interpolated_history_ball_2 = interpolate_missing(history_ball_2)
            smoothed_history_ball_2 = [p for p in interpolated_history_ball_2 if p is not None]  # Remove None
            if len(smoothed_history_ball_2) >= minimum_req:
                predicted_positions_ball_2 = predictors.ekf(smoothed_history_ball_2, prediction_range)

        # Check for predicted collisions
        if predicted_positions_ball_1 and predicted_positions_ball_2:
            for p1 in predicted_positions_ball_1:
                for p2 in predicted_positions_ball_2:
                    if width / 2 - 10 <= p1[0] <= width / 2 + 10 and height / 2 - 10 <= p1[1] <= height / 2 + 10:
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
    else:
        all_predicted.append(actual[0])
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
