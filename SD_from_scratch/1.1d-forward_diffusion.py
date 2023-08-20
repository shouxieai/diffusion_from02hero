import numpy as np
import matplotlib.pyplot as plt

def noise_strength_constant(t):
    return 1

def forward_diffusion_1D(x0, t, dt, nsteps, noise_strength_fn):
    # Initialize trajectory
    x = np.zeros((nsteps+1, x0.shape[0]), dtype = float)
    x[0] = x0
    ts = [t] # Initialize time array with starting time
    
    # Perform for loop to compute t+dela_t from t
    for i in range(nsteps):
        noise_strength = noise_strength_fn(t)
        random_normal = np.random.randn(x0.shape[0])
        x[i+1] = x[i] + np.sqrt(dt)*noise_strength*random_normal
        t = t + dt
        ts.append(t)
    return x, ts

if __name__ == "__main__":
    nsteps = 100
    t = 0
    dt = 0.1
    noise_strength_fn = noise_strength_constant
    
    num_particles = 5
    x0 = np.zeros(num_particles) # 5
    x, ts = forward_diffusion_1D(x0, t, dt, nsteps, noise_strength_fn)
    
    plt.plot(ts, x)
    plt.xlabel('time', fontsize=20)
    plt.ylabel('$x$', fontsize=20)
    plt.title('Forward diffusion visualized', fontsize=20)
    plt.savefig('./fig/1d-forward-diffusion.png')
    
    