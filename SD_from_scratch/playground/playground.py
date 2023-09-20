
####### rv.rvs(N) ####### how to draw samples from a multivariate normal distribution
import numpy as np
from scipy.stats import multivariate_normal

# Define the mean and covariance matrix for the distribution
mean = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])

# Create a multivariate normal random variable
rv = multivariate_normal(mean, cov) # type: ignore

# Draw N random samples from this distribution
N = 10
samples = rv.rvs(N)

# Print the samples
print(samples)


######### How to integrate a f(x) in computer ###########
import numpy as np
import matplotlib.pyplot as plt

# Define the function to be integrated
def f(x):
    return x**2

# Define the limits of integration and the number of trapezoids
a = 0
b = 1
n = 100

# Calculate the height of the trapezoids
h = (b - a) / n 

# Calculate the integral
integral = 0
for i in range(n):
    integral += h * (f(a + i*h) + f(a + (i+1)*h)) / 2 # eq for computing the area of the trapezoid

print("The numerical integral is:", integral)

# # Plot the function and the trapezoids
x = np.linspace(a, b, 1000)
y = f(x)

plt.plot(x, y, 'r', label='x^2')

# plot the numerical integration of f(x) using trapezoidal rule
plt.fill_between(x, y, where=(x>a) & (x<b), color='blue', alpha=0.5, label='Trapezoidal Rule')
plt.title('Trapezoidal Rule')
plt.legend()
plt.show()
plt.savefig('trapezoidal_rule.png', dpi=300, bbox_inches='tight')

