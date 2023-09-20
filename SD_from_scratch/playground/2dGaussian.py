import numpy as np
from scipy.stats import multivariate_normal

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[1., -0.5],
                 [-0.5, 1.5]])

# Create a 2-d dim multivariate normal distribution
rv = multivariate_normal(mu, Sigma) # type: ignore

# print the important information of the above rv
print(rv.mean)
print(rv.cov)

# key info
'''
    ...
    dim: 2
    mean:
        array([0., 1.])
    cov:
        array([[ 1. , -0.5],
                [-0.5,  1.5]])
    ...
'''