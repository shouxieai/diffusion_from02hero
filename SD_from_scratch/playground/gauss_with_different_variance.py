import numpy as np
import matplotlib.pyplot as plt

# Number of samples per distribution
N = 2000

# Variance values
variances = [1, 5, 10, 15]

# Number of rows and columns for subplots
n_rows = 2
n_cols = 2

# Define the range for both axes (you can adjust these values as needed)
x_range = (-10, 10)
y_range = (-10, 10)

# Create a plot for each variance
for index, var in enumerate(variances):
    # Generate samples from a standard normal distribution
    gauss = np.random.randn(N, 2)

    # Scale the samples to have the desired variance
    gauss_scaled = np.sqrt(var) * gauss

    # Create subplot
    plt.subplot(n_rows, n_cols, index + 1)

    # Plot the 2D samples as a scatter plot
    plt.scatter(gauss_scaled[:, 0], gauss_scaled[:, 1], alpha=0.5)
    plt.xlim(x_range) # Set the range for the x axis
    plt.ylim(y_range) # Set the range for the y axis
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Variance {var}')

# Adjust layout
plt.tight_layout()
plt.savefig("gauss_scaled.png") # Save before showing
plt.show()
