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

if __name__ == "__main__":
    mu1 = np.array([0, 1.0]) # 2D Gaussian distribution
    Cov1 = np.array([[1.0, 0.0],
                     [0.0, 1.0]]) # covariance matrix 2x2
    
    mu2 = np.array([2.0, -1.0])
    Cov2 = np.array([[2.0, 0.5],
                    [0.5, 1.0]])
    
    gmm = GaussianMixture([mu1, mu2], [Cov1, Cov2], [1.0, 1.0])
    gmm_samples, rand_component, all_samples = gmm.sample(5000)
    scorevecs = gmm.score(gmm_samples)
    
    # output
    # >>> gmm_samples  (5000, 2) # 5000 samples, 2D Gaussian distribution
    # rand_component  (5000,) # 0, 1
    # all_samples  (2, 5000, 2) --> index (0, 3, 1) 
    # scorevecs  (5000, 2) # 2 --> feature dims.     ---> 2 components
    
    print("gmm_samples ",gmm_samples.shape)
    print("rand_component ",rand_component.shape)
    print("all_samples ",all_samples.shape)
    print("scorevecs ",scorevecs.shape)

    figh, ax = plt.subplots(1,1,figsize=[6,6])
    kdeplot(all_samples[0,:,:], label="comp1", )
    kdeplot(all_samples[1,:,:], label="comp2", )
    plt.title("Empirical density of each component")
    plt.legend()
    plt.axis("image");
    plt.savefig("Empirical_density_of_each_component.png")
    plt.clf()

    figh, ax = plt.subplots(1,1,figsize=[6,6])
    kdeplot(gmm_samples, )
    plt.title("Empirical density of Gaussian mixture density")
    plt.axis("image");
    plt.savefig("Empirical_density_of_Gaussian_mixture_density.png")
    plt.clf()

    plt.figure(figsize=[8,8])
    quiver_plot(gmm_samples, scorevecs)
    plt.title("Score vector field $\log p(x)$")
    plt.axis("image");  
    plt.savefig("Score_vector_field_log_p(x).png")
    plt.clf()
    