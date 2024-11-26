# Function to calculate statistics
def calculate_stats(data):
    num_objects = len(data)  # Total number of objects
    total_coordinates = 0  # Total detected coordinates
    total_failures = 0  # Total number of None values
    consecutive_fails = []  # List to track consecutive None counts
    current_consecutive = 0  # Current consecutive None count

    for values in data.values():
        for value in values:
            if value is None:
                total_failures += 1
                current_consecutive += 1
            else:
                total_coordinates += 1
                if current_consecutive > 0:
                    consecutive_fails.append(current_consecutive)
                current_consecutive = 0
        if current_consecutive > 0:  # Add remaining consecutive fails if any
            consecutive_fails.append(current_consecutive)
            current_consecutive = 0

    # Calculate stats
    percentage_failures = (total_failures / (total_coordinates + total_failures)) * 100 if (total_coordinates + total_failures) > 0 else 0
    avg_consecutive_fails = sum(consecutive_fails) / len(consecutive_fails) if consecutive_fails else 0

    return {
        "Number of detected objects": num_objects,
        "Number of coordinates detected": total_coordinates,
        "Number of failures": total_failures,
        "Percentage of failures": percentage_failures,
        "AVG consecutive fails": avg_consecutive_fails
    }

# Read data from data.txt
data = {}
with open("cleaned_data.txt", "r") as file:
    for line in file:
        key, values = line.strip().split("->")
        key = key.strip()
        values = eval(values.strip())
        data[key] = values

# Calculate statistics
stats = calculate_stats(data)

# Write statistics to results.txt
with open("results.txt", "w") as file:
    for key, value in stats.items():
        file.write(f"{key}: {value}\n")
