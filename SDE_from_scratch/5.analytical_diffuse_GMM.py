import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

def kdeplot(pnts, label="", ax=None, titlestr=None, **kwargs):
  if ax is None:
    ax = plt.gca()#figh, axs = plt.subplots(1,1,figsize=[6.5, 6])
  sns.kdeplot(x=pnts[:,0], y=pnts[:,1], ax=ax, label=label, **kwargs)
  if titlestr is not None:
    ax.set_title(titlestr)
    
def quiver_plot(pnts, vecs, *args, **kwargs):
  plt.quiver(pnts[:, 0], pnts[:,1], vecs[:, 0], vecs[:, 1], *args, **kwargs)

class GaussianMixture():
    def __init__(self, mus, covs, weights) -> None:
        self.n_components = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [np.linalg.inv(cov) for cov in covs] # precision matrices
        self.weights = weights # e.g. 1:1 
        self.norm_weights = weights / np.sum(weights) # 0.5:0.5
        self.RVs =[]
        
        for i in range(len(mus)):
            self.RVs.append(multivariate_normal(mus[i], covs[i]))
        
        self.dims = len(mus[0])
        
    def score(self, x): # x (5000, 2)
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis,:]
        participance = weighted_compon_pdf / np.sum(weighted_compon_pdf, axis=1, keepdims=True)
        # (1) distance to mus, (2) weight assigned to each component
        
        scores = np.zeros_like(x) # (5000, 2) ！！！！
        
        for i in range(self.n_components):
          gradvec = -(x - self.mus[i]) @ self.precs[i]
          scores += participance[:, i, np.newaxis] * gradvec # (5000, 2) ！2->feaeture dims instead of 2 compoents
        
        return scores


    def sample(self, N):
        """draw N samples from the mixture model"""
        rand_component = np.random.choice(self.n_components, size=N, p=self.norm_weights)
        all_samples = np.array([rv.rvs(N) for rv in self.RVs]) # 5000 dog samples + 5000 wolf samples
        gmm_samples = all_samples[rand_component, np.arange(N), :] # 2, 5000, 2: 2 components, 5000 samples, 2D Gaussian distribution
        return gmm_samples, rand_component, all_samples

def marginal_prob_std_np(t, sigma): # std->new_sigma
  return np.sqrt((sigma**(2*t) - 1 ) /( 2 * np.log(sigma))) # return a std(i.e. sigma)instead of a var(i.e. sigma**2)

def diffuse_gmm(gmm, t, sigma):
  beta_t = marginal_prob_std_np(t, sigma)**2 # sigma --> std   sigma**2 --> var
  noise_cov = np.eye(gmm.dims) * beta_t
  covs_diff = [cov + noise_cov for cov in gmm.covs] # see them as two indepent gaussian distriubtion
  return GaussianMixture(gmm.mus, covs_diff, gmm.weights)


if __name__ == "__main__":
    mu1 = np.array([0, 1.0]) # 2D Gaussian distribution
    Cov1 = np.array([[1.0, 0.0],
                     [0.0, 1.0]]) # covariance matrix 2x2
    
    mu2 = np.array([2.0, -1.0])
    Cov2 = np.array([[2.0, 0.5],
                    [0.5, 1.0]])
    
    ##### iterative solution
    gmm = GaussianMixture([mu1, mu2], [Cov1, Cov2], [1.0, 1.0])
    x0, _, _ = gmm.sample(2000)
   
    sigma = 5
    nsteps = 200

    x_traj = np.zeros((*x0.shape, nsteps, ))
    x_traj[:,:,0] = x0
    dt = 1 / nsteps
    for i in range(1, nsteps):
        t = i * dt
        eps_z = np.random.randn(*x0.shape)
        x_traj[:,:,i] = x_traj[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) # central limit theorem
    
    
    # Set up the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=[12, 6])

    # Plot density of target distribution of x_0
    sns.kdeplot(x=x_traj[:, 0, 0], y=x_traj[:, 1, 0], ax=axs[0],color="black")
    axs[0].set_title("Density of Target distribution of $x_0$")
    axs[0].axis("equal")
    
    ##### anaylytical solution
    # Diffuse GMM and sample from it
    gmm_t = diffuse_gmm(gmm, nsteps/nsteps, sigma)
    samples_t, _, _ = gmm_t.sample(2000) # (2000, 2)

    # Plot density of x_T samples after nsteps step diffusion (both diffused and analytical GMM)
    sns.kdeplot(x=x_traj[:, 0, nsteps-1], y=x_traj[:, 1, nsteps-1], ax=axs[1], label="iter solution of GMM")
    sns.kdeplot(x=samples_t[:, 0], y=samples_t[:, 1], ax=axs[1], label="analy solution of GMM")
    axs[1].set_title(f"Density of $x_T$ samples after {nsteps} step diffusion")
    axs[1].axis("equal")
    axs[1].legend()

    # Save the figure
    plt.savefig("target_dist_iterative_and_analytical_dist.png")