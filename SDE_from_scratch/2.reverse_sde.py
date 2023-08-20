import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Exact transition probability for 1D diffusion
def transition_probability_diffusion_exact(x, t, params):
  x0, sigma= params['x0'], params['sigma']

  pdf = norm.pdf(x, loc=x0, scale=np.sqrt((sigma**2)*t))  # pdf of normal distribution with mean x0 and variance (sigma^2)*t
  return pdf

def f_diff_simple(x, t, params):
    return np.zeros((*x.shape,))

def g_diff_simple(x, t, params):
    sigma = params["sigma"]
    return sigma * np.ones((*x.shape,))

def f_diff_simple_rev(x, t, params):
    T = params["T"]
    return -f_diff_simple(x, T-t, params) + (g_diff_simple(x, T-t, params))**2 * score(x, T-t, params)

def g_diff_simple_rev(x, t, params):
    sigma = params["sigma"]
    return sigma * np.ones((*x.shape,))

def score(x, t, params):
    sigma = params["sigma"]
    score_ = (x0 - x) / ((sigma**2) * t)
    return score_
    
# Simulate SDE with drift function f and noise amplitude g for arbitrary number of steps
def SDE_simulation(x0, nsteps, dt, f, g, params):
    t = 0
    x_traj = np.zeros((nsteps + 1, *x0.shape))
    x_traj[0] = np.copy(x0)
    
    # Perform many Euler-maruyama time steps
    for i in range(nsteps):
        random_normal = np.random.randn(*x0.shape)
        x_traj[i+1] = x_traj[i] + f(x_traj[i], t, params) * dt + g(x_traj[i], t, params) * np.sqrt(dt) * random_normal
        t = t + dt
        
    return x_traj

if __name__ == "__main__":
    sigma = 1         # noise amplitude for 1D diffusion

    num_samples = 1000
    x0 = np.zeros(num_samples)    # initial condition for diffusion

    nsteps = 2000      # number of simulation steps
    dt = 0.001          # size of small time steps 
    T = nsteps*dt
    t = np.linspace(0, T, nsteps + 1)

    params = {'sigma': sigma, 'x0':x0, 'T':T}
    
    # Forward simulate the model. Then reverse diffuse the results.
    x_traj = SDE_simulation(x0, nsteps, dt, f_diff_simple, g_diff_simple, params)
    # Reverse diffusion
    x_traj_rev = SDE_simulation(x0=x_traj[-1], nsteps=nsteps, \
        dt=dt, f=f_diff_simple_rev, g=g_diff_simple_rev, params=params)
    
    # Compute exact transition probability 
    x_f_min, x_f_max = np.amin(x_traj[-1]), np.amax(x_traj[-1])
    num_xf = 1000
    x_f_arg = np.linspace(x_f_min, x_f_max, num_xf)
    pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

    # Plot final distribution (distribution after diffusion / before reverse diffusion)
    plt.hist(x_traj[-1], bins=100)
    plt.plot(x_f_arg, pdf_final, color='black', linewidth=5)
    plt.title("$t = $"+str(T), fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.show()
    plt.savefig('reverse_initial_distribution.png')
    plt.clf()

    # Plot initial distribution (distribution before diffusion / after reverse diffusion)
    #fig, ax = plt.subplots(1, 2, width=)
    plt.hist(x_traj_rev[-1], bins=100, label='rev. diff.')
    plt.hist(x_traj[0], bins=100, label='true')

    plt.title("$t = 0$", fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    plt.savefig('reverse_final_distribution.png')
    plt.clf()

    # Plot some trajectories
    sample_trajectories = [0, 1, 2, 3, 4]
    for s in sample_trajectories:
        plt.plot(t, x_traj_rev[:,s])
    plt.title("Sample trajectories (reverse process)", fontsize=20)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("x", fontsize=20)
    plt.show()
    plt.savefig('reverse_trajectories.png')
    plt.clf()
