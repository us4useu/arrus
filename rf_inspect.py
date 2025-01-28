import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = "rf.npy"  # Replace with your .npy file path
data = np.load(file_path)

# Inspect the shape and contents of the data
print(f"Data shape: {data.shape}")
print(data)

# Select a subset for visualization
# For example: first data set, first category
data_to_plot = data[0, 0, 0]  # Shape (4096, 256)

# Inspect subset shape
print(f"Subset shape for plotting: {data_to_plot.shape}")


# Define the value range for color mapping
vmin, vmax = -100, 100  # Replace with your desired range

# Plot 
plt.imshow(data_to_plot[:,:], cmap="viridis", aspect="auto", origin='upper', vmin=vmin, vmax=vmax)
plt.colorbar(label="Value")  # Optional: Colorbar for value reference
plt.show()