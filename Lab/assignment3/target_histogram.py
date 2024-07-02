import numpy as np
import matplotlib.pyplot as plt

def double_gaussian_distribution(x, mu1, sigma1, mu2, sigma2):
    gaussian1 = np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) / (sigma1 * np.sqrt(2 * np.pi))
    gaussian2 = np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) / (sigma2 * np.sqrt(2 * np.pi))
    return gaussian1 + gaussian2

# Parameters for the double Gaussian distribution
mu1 = 30
sigma1 = 8
mu2 = 165
sigma2 = 20

# Generate x values
x = np.linspace(0, 255, 256)

# Generate the double Gaussian distribution
double_gaussian = double_gaussian_distribution(x, mu1, sigma1, mu2, sigma2)

# Save the target histogram to a file
# np.save('target_histogram.npy', double_gaussian)
double_gaussian = double_gaussian/np.min(double_gaussian)
double_gaussian = double_gaussian/10
# Plot x vs double Gaussian
plt.plot(x, double_gaussian)
plt.title("x vs Double Gaussian Distribution")
plt.xlabel("x")
plt.ylabel("Double Gaussian Value")
plt.show()