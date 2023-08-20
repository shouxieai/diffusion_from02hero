import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import torch

def sample_X_and_score(gmm, trainN, testN):
  X_train, _, _ = gmm.sample(trainN)
  y_train       = gmm.score(X_train) # analytical score function
  X_test, _, _  = gmm.sample(testN)
  y_test        = gmm.score(X_test)
  X_train_tsr = torch.tensor(X_train).float()
  y_train_tsr = torch.tensor(y_train).float()
  X_test_tsr = torch.tensor(X_test).float()
  y_test_tsr = torch.tensor(y_test).float()
  
  return X_train_tsr, y_train_tsr, X_test_tsr, y_test_tsr

def sample_X_and_score_t_depend(gmm, trainN=10000, testN=2000, sigma=5, partition=20, EPS=0.02):
  trainN_part, testN_part = trainN // partition, testN // partition
  X_train_list, y_train_list, X_test_list, y_test_list, T_train_list, T_test_list = [], [], [], [], [], []
  for t in np.linspace(0, 1, partition): # 0~T --> 0~1
    gmm_t = diffuse_gmm(gmm, t, sigma) # analytical solution to diffuse the GMM. gmm_t
    X_train_tsr, y_train_tsr, X_test_tsr, y_test_tsr = sample_X_and_score(gmm_t, trainN=trainN_part, testN=testN_part)
    T_train_tsr, T_test_tsr = t * torch.ones(trainN_part), t * torch.ones(testN_part)
    X_train_list.append(X_train_tsr), y_train_list.append(y_train_tsr), T_train_list.append(T_train_tsr)
    X_test_list.append(X_test_tsr), y_test_list.append(y_test_tsr), T_test_list.append(T_test_tsr)
  X_train_tsr = torch.cat(X_train_list, dim=0)
  y_train_tsr = torch.cat(y_train_list, dim=0)
  X_test_tsr = torch.cat(X_test_list, dim=0)
  y_test_tsr = torch.cat(y_test_list, dim=0)
  T_train_tsr = torch.cat(T_train_list, dim=0)
  T_test_tsr = torch.cat(T_test_list, dim=0)
  return X_train_tsr, y_train_tsr, T_train_tsr, X_test_tsr, y_test_tsr, T_test_tsr

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

def marginal_prob_std(t, sigma):
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) )

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
    
    gmm = GaussianMixture([mu1, mu2], [Cov1, Cov2], [1.0, 1.0])
   
    nsteps = 200
    Train_sampleN = 100000
    Test_sampleN = 2000
    sigma = 10

    X_train, y_train, T_train, X_test, y_test, T_test = \
      sample_X_and_score_t_depend(gmm, sigma=sigma, trainN=Train_sampleN, testN=Test_sampleN,
                                partition=nsteps, EPS=0.0001)
      
    score_norm = y_train.norm(dim=1) # 100000,2 ---> 100000
    samp_norm = X_train.norm(dim=1) # 100000
    fig,axs= plt.subplots(1,2,figsize=[12,6])
    sns.lineplot(x=T_train, y=score_norm, ax=axs[0]) # blue
    sns.lineplot(x=T_train, y=score_norm* marginal_prob_std(T_train, sigma), ax=axs[0]) # (sigma**(T_train))
    # orange
    axs[0].set(xlabel="diffusion time t", ylabel="norm s(x,t)", title="Score norm ~ time")
    sns.lineplot(x=T_train, y=samp_norm, ax=axs[1])
    axs[1].set(xlabel="diffusion time t", ylabel="norm x", title="Sample norm / std ~ time")
    plt.savefig("score_norm.png")