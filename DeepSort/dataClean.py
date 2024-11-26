# Function to clean the data by removing trailing `None` values
def clean_data(data):
    cleaned_data = {}
    for key, values in data.items():
        # Find the last non-None value's index
        if values:
            last_valid_index = len(values) - 1
            while last_valid_index >= 0 and values[last_valid_index] is None:
                last_valid_index -= 1
            # Slice the list up to the last non-None value
            cleaned_data[key] = values[:last_valid_index + 1]
        else:
            cleaned_data[key] = []  # If no values are present, return an empty list
    return cleaned_data

# Read data from data.txt
data = {}
with open("data.txt", "r") as file:
    for line in file:
        key, values = line.strip().split("->")
        key = key.strip()
        values = eval(values.strip())
        data[key] = values

# Clean the data
cleaned_data = clean_data(data)

# Write cleaned data to a text file
with open("cleaned_data.txt", "w") as file:
    for key, values in cleaned_data.items():
        file.write(f"{key} -> {values}\n")
