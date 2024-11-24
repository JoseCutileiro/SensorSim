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

noise_strength = 20 * 0.15

# Create a blank video writer with OpenCV
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

# Initialize history for ball positions
history_ball_1 = []
history_ball_2 = []


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
            if len(interpolated) > 1:
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



def interpolate_missing_adv(history):
    """
    Interpolate missing points in the history using an informed estimation.
    """
    interpolated = []
    for i in range(len(history)):
        if history[i] is not None:
            interpolated.append(history[i])  # Keep existing points
        else:
            # Find previous and next valid points
            prev_idx = next((j for j in range(i - 1, -1, -1) if history[j] is not None), None)
            next_idx = next((j for j in range(i + 1, len(history)) if history[j] is not None), None)

            if prev_idx is not None and next_idx is not None:
                # Use informed interpolation based on speed and position
                prev_point = np.array(history[prev_idx])
                next_point = np.array(history[next_idx])
                time_fraction = (i - prev_idx) / (next_idx - prev_idx)
                # Estimate velocity between adjacent points
                velocity = (next_point - prev_point) / (next_idx - prev_idx)
                # Predict the missing point
                estimated_point = prev_point + time_fraction * velocity * (next_idx - prev_idx)
                interpolated.append(tuple(estimated_point))
            elif prev_idx is not None:
                # Use last known velocity to estimate next position
                if prev_idx > 0 and history[prev_idx - 1] is not None:
                    last_point = np.array(history[prev_idx - 1])
                    prev_point = np.array(history[prev_idx])
                    velocity = prev_point - last_point
                    estimated_point = prev_point + velocity  # Extrapolate forward
                    interpolated.append(tuple(estimated_point))
                else:
                    interpolated.append(history[prev_idx])  # Use the last known point
            elif next_idx is not None:
                # Use next known velocity to estimate previous position
                if next_idx < len(history) - 1 and history[next_idx + 1] is not None:
                    next_point = np.array(history[next_idx])
                    future_point = np.array(history[next_idx + 1])
                    velocity = future_point - next_point
                    estimated_point = next_point - velocity  # Extrapolate backward
                    interpolated.append(tuple(estimated_point))
                else:
                    interpolated.append(history[next_idx])  # Use the next known point
            else:
                interpolated.append(None)  # If no points are available, leave as None
    return interpolated


# Interpolation function for filling missing points
def interpolate_missing_basic(history):
    # Create a new list to store interpolated values
    interpolated = []
    for i in range(len(history)):
        if history[i] is not None:
            interpolated.append(history[i])  # Keep existing points
        else:
            # Find previous and next valid points
            prev_idx = next((j for j in range(i - 1, -1, -1) if history[j] is not None), None)
            next_idx = next((j for j in range(i + 1, len(history)) if history[j] is not None), None)
            if prev_idx is not None and next_idx is not None:
                # Linear interpolation
                prev_point = np.array(history[prev_idx])
                next_point = np.array(history[next_idx])
                weight = (i - prev_idx) / (next_idx - prev_idx)
                interpolated.append(tuple(prev_point + weight * (next_point - prev_point)))
            elif prev_idx is not None:
                interpolated.append(history[prev_idx])  # Use previous point if no future point
            elif next_idx is not None:
                interpolated.append(history[next_idx])  # Use next point if no previous point
            else:
                interpolated.append(None)  # If no points are available, leave as None
    return interpolated

# Main loop
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

    # Add current positions to history with possible missing points
    if frame % 10 == 0:
        if random.randint(0, 2):
            history_ball_1.append((ball1_pos[0] + random.randrange(0, noise_strength),
                                   ball1_pos[1] + random.randrange(0, noise_strength)))
        else:
            history_ball_1.append(None)
        if random.randint(0, 2):
            history_ball_2.append((ball2_pos[0] + random.randrange(0, noise_strength),
                                   ball2_pos[1] + random.randrange(0, noise_strength)))
        else:
            history_ball_2.append(None)

    # Limit history to the last `history_limit` positions
    if len(history_ball_1) > history_limit:
        history_ball_1.pop(0)
    if len(history_ball_2) > history_limit:
        history_ball_2.pop(0)

    # Predict positions using EKF with interpolation and smoothing
    predicted_positions_ball_1 = []
    predicted_positions_ball_2 = []

    if len(history_ball_1) >= minimum_req:
        interpolated_history_ball_1 = interpolate_missing(history_ball_1)
        smoothed_history_ball_1 = [p for p in interpolated_history_ball_1 if p is not None]  # Remove None
        if len(smoothed_history_ball_1) >= minimum_req:
            predicted_positions_ball_1 = predictors.ekf(smoothed_history_ball_1, prediction_range)

    if len(history_ball_2) >= minimum_req:
        interpolated_history_ball_2 = interpolate_missing(history_ball_2)
        smoothed_history_ball_2 = [p for p in interpolated_history_ball_2 if p is not None]  # Remove None
        if len(smoothed_history_ball_2) >= minimum_req:
            predicted_positions_ball_2 = predictors.ekf(smoothed_history_ball_2, prediction_range)

    # Check for predicted collisions
    if predicted_positions_ball_1 and predicted_positions_ball_2:
        for p1 in predicted_positions_ball_1:
            for p2 in predicted_positions_ball_2:
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
        if pos is not None:
            alpha = (idx + 1) / history_limit  # Gradual fade effect
            color = (int(255 * alpha), 0, 0)  # Red fade
            draw.ellipse((pos[0] - 5, pos[1] - 5, pos[0] + 5, pos[1] + 5), fill=color)

    for idx, pos in enumerate(history_ball_2):
        if pos is not None:
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
