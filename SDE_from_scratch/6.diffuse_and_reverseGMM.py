import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import torch

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
    nsteps = 1000

    x_traj = np.zeros((*x0.shape, nsteps, ))
    x_traj[:,:,0] = x0
    dt = 1 / nsteps
    for i in range(1, nsteps):
        t = i * dt
        eps_z = np.random.randn(*x0.shape)
        x_traj[:,:,i] = x_traj[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) # central limit theorem
    
    
    # Set up the figure and axes
    fig, axs = plt.subplots(1, 3, figsize=[18, 6])

    # Plot density of target distribution of x_0
    sns.kdeplot(x=x_traj[:, 0, 0], y=x_traj[:, 1, 0], ax=axs[0],color="black")
    axs[0].set_title("Density of Target distribution of $x_0$")
    axs[0].axis("equal")
    
    ##### anaylytical solution
    # Diffuse GMM and sample from it
    sampleN = 2000
    gmm_t = diffuse_gmm(gmm, nsteps/nsteps, sigma)
    samples_t, _, _ = gmm_t.sample(sampleN) # (2000, 2)

    # Plot density of x_T samples after nsteps step diffusion (both diffused and analytical GMM)
    sns.kdeplot(x=x_traj[:, 0, nsteps-1], y=x_traj[:, 1, nsteps-1], ax=axs[1], label="iter solution of GMM")
    sns.kdeplot(x=samples_t[:, 0], y=samples_t[:, 1], ax=axs[1], label="analy solution of GMM")
    axs[1].set_title(f"Density of $x_T$ samples after {nsteps} step diffusion")
    axs[1].axis("equal")
    axs[1].legend()
    
    ###### reverse diffusion sampling
    betaT = (sigma**2 - 1) / (2 * np.log(sigma)) # analytical solution
    xT = np.sqrt(betaT) * np.random.randn(sampleN, 2) # x~N(0,1) --> x_T ~ N(0, betaT) :  x * sqrt(betaT)
    x_traj_rev = np.zeros((*x0.shape, nsteps, ))
    x_traj_rev[:, :, 0] = xT # x_traj_rev.shape = (2000, 2, 1000) # 2000 samples, 2D Gaussian distribution, 1000 steps
    dt = 1 / nsteps # 0 ~ T --> 0 ~ 1
    
    for i in range(1, nsteps): # starting from 1
      t = (nsteps - i) * dt # time step: 1 ~ 0; t init = 0.999
      gmm_t = diffuse_gmm(gmm, t, sigma)
      score_xt= gmm_t.score(x_traj_rev[:, :, i-1])
      eps_z   = np.random.randn(*x0.shape) 
      x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + sigma**(2*t) * score_xt * dt + (sigma**t)* np.sqrt(dt) * eps_z
    
    # Plot for x_traj_rev
    sns.kdeplot(x=x_traj_rev[:,0,-1], y=x_traj_rev[:,1,-1], ax=axs[2])
    axs[2].set_title("reversed Distribution")
    plt.savefig("gt_diffused_and_reversed.png")

############
def reverse_diffusion_time_dep(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False):
  betaT = (sigma**2 - 1) / (2 * np.log(sigma))
  xT = np.sqrt(betaT) * np.random.randn(sampN, ndim)
  x_traj_rev = np.zeros((*xT.shape, nsteps, ))
  x_traj_rev[:,:,0] = xT
  dt = 1 / nsteps
  for i in range(1, nsteps):
    t = 1 - i * dt
    tvec = torch.ones((sampN)) * t
    eps_z = np.random.randn(*xT.shape)
    if exact: # given a known GMM
      gmm_t = diffuse_gmm(score_model_td, t, sigma) # score_model_td is gmm
      score_xt = gmm_t.score(x_traj_rev[:,:,i-1]) # <====== (1) given known gmm_t, gmm_t.score()
    else: # the target distribution is unknown, use a model to learn the score. Check next section
      with torch.no_grad():
        # score_xt = score_model_td(torch.cat((torch.tensor(x_traj_rev[:,:,i-1]).float(),tvec),dim=1)).numpy()
        score_xt = score_model_td(torch.tensor(x_traj_rev[:,:,i-1]).float(), tvec).numpy() # (2)<===== score = model.predict(x, t)
    x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + sigma**(2*t) * score_xt * dt + (sigma**t)* np.sqrt(dt) * eps_z
    
  return x_traj_rev
