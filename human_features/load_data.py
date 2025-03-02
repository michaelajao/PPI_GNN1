import numpy as np

# Define the path to your .npy file
file_path = 'npy_file_new(human_dataset).npy'

# Load the data from the .npy file
data = np.load(file_path, allow_pickle=True)

# Print the loaded data
print(data)