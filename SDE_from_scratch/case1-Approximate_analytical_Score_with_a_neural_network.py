import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixture:
  def __init__(self, mus, covs, weights):
    """
    mus: a list of K 1d np arrays (D,)
    covs: a list of K 2d np arrays (D, D)
    weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each branch.
      They will be normalized to sum to 1. If they sum to zero, it will err.
    """
    self.n_component = len(mus)
    self.mus = mus
    self.covs = covs
    self.precs = [np.linalg.inv(cov) for cov in covs]
    self.weights = np.array(weights)
    self.norm_weights = self.weights / self.weights.sum()
    self.RVs = []
    for i in range(len(mus)):
      self.RVs.append(multivariate_normal(mus[i], covs[i]))
    self.dim = len(mus[0])

  def add_component(self, mu, cov, weight=1):
    self.mus.append(mu)
    self.covs.append(cov)
    self.precs.append(np.linalg.inv(cov))
    self.RVs.append(multivariate_normal(mu, cov))
    self.weights.append(weight)
    self.norm_weights = self.weights / self.weights.sum()
    self.n_component += 1

  def pdf(self, x):
    """
      probability density (PDF) at $x$.
    """
    component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
    prob = np.dot(component_pdf, self.norm_weights)
    return prob

  def score(self, x):
    """
    Compute the score $\nabla_x \log p(x)$ for the given $x$.
    """
    component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T # (5000,)(5000,) ---> (5000, 2)
    weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :] # (5000, 2)
    participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

    scores = np.zeros_like(x)
    for i in range(self.n_component):
      gradvec = - (x - self.mus[i]) @ self.precs[i]
      scores += participance[:, i:i+1] * gradvec

    return scores

  def score_decompose(self, x):
    """
    Compute the grad to each branch for the score $\nabla_x \log p(x)$ for the given $x$.
    """
    component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
    weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
    participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

    gradvec_list = []
    for i in range(self.n_component):
      gradvec = - (x - self.mus[i]) @ self.precs[i]
      gradvec_list.append(gradvec)
      # scores += participance[:, i:i+1] * gradvec

    return gradvec_list, participance

  def sample(self, N):
    """ Draw N samples from Gaussian mixture
    Procedure:
      Draw N samples from each Gaussian
      Draw N indices, according to the weights.
      Choose sample between the branches according to the indices.
    """
    rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
    all_samples = np.array([rv.rvs(N) for rv in self.RVs])
    gmm_samps = all_samples[rand_component, np.arange(N),:]
    return gmm_samps, rand_component, all_samples

def quiver_plot(pnts, vecs, *args, **kwargs):
  plt.quiver(pnts[:, 0], pnts[:,1], vecs[:, 0], vecs[:, 1], *args, **kwargs)

def kdeplot(pnts, label="", ax=None, titlestr=None, **kwargs):
  if ax is None:
    ax = plt.gca()#figh, axs = plt.subplots(1,1,figsize=[6.5, 6])
  sns.kdeplot(x=pnts[:,0], y=pnts[:,1], ax=ax, label=label, **kwargs)
  if titlestr is not None:
    ax.set_title(titlestr)

def visualize_diffusion_distr(x_traj_rev, leftT=0, rightT=-1, explabel=""):
  if rightT == -1:
    rightT = x_traj_rev.shape[2]-1
  figh, axs = plt.subplots(1,2,figsize=[12,6])
  sns.kdeplot(x=x_traj_rev[:,0,leftT], y=x_traj_rev[:,1,leftT], ax=axs[0])
  axs[0].set_title("Density of Gaussian Prior of $x_T$\n before reverse diffusion")
  plt.axis("equal")
  sns.kdeplot(x=x_traj_rev[:,0,rightT], y=x_traj_rev[:,1,rightT], ax=axs[1])
  axs[1].set_title(f"Density of $x_0$ samples after {rightT} step reverse diffusion")
  plt.axis("equal")
  plt.suptitle(explabel)
  return figh

#### initialize two gaussian components and form a GMM
mu1 = np.array([0,1.0])
Cov1 = np.array([[1.0,0.0],
          [0.0,1.0]])

mu2 = np.array([2.0,-1.0])
Cov2 = np.array([[2.0,0.5],
          [0.5,1.0]])

gmm = GaussianMixture([mu1,mu2],[Cov1,Cov2],[1.0,1.0])

#### (Forward) Diffusion Process
def marginal_prob_std(t, sigma): # obtain the standard deviation of the marginal probability at time t
  # t is the time step mapped into [0,1]
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) )

def marginal_prob_std_np(t, sigma): # t is the time step mapped into [0,1]
  return np.sqrt( (sigma**(2*t) - 1) / 2 / np.log(sigma) )

def diffuse_gmm(gmm, t, sigma):
  lambda_t = marginal_prob_std_np(t, sigma)**2     # cumulative variance at time t
  noise_cov = np.eye(gmm.dim) * lambda_t           # covariance of the noise
  covs_dif = [cov + noise_cov for cov in gmm.covs] # add the covariance of the noise to the original covariance
  return GaussianMixture(gmm.mus, covs_dif, gmm.weights) # return the new GMM at time t


### Diffusion in Action
x0, _, _ = gmm.sample(1000)

sigma = 5
nsteps = 200

x_traj = np.zeros((*x0.shape, nsteps, ))
x_traj[:,:,0] = x0
dt = 1 / nsteps
for i in range(1, nsteps): # 1, 2, ..., nsteps-1
  t = i * dt
  eps_z = np.random.randn(*x0.shape)
  x_traj[:,:,i] = x_traj[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) # forward diffusion in descrete time

### Validate the time dependent density with theory
# t_step = 200
# gmm_t = diffuse_gmm(gmm, t_step / nsteps, sigma) # analytical one-step solution of the diffused distribution at time t
# samps_t, _, _ = gmm_t.sample(1000)


### Reverse Diffusion sampling(GMM is known)
sampN = 1000
sigma = 5

sigmaT = (sigma**2 - 1) / (2 * np.log(sigma))    # marginal variance at time T(i.e. t=1)
xT = np.sqrt(sigmaT) * np.random.randn(sampN, 2) # sample from the marginal distribution at time T (final diffused distribution)
x_traj_rev = np.zeros((*x0.shape, nsteps, ))     # initialize the reverse diffusion trajectory, x0 is the final diffused distribution.
x_traj_rev[:,:,0] = xT
dt = 1 / nsteps
for i in range(1, nsteps):                      
  t = (nsteps - i) * dt                       # time fly back. e.g. nsteps=200, i=1, t=199/200, i=2, t=198/200   
  gmm_t = diffuse_gmm(gmm, t, sigma)          # note the time fly back! start from the largest noise scale. Attention here, gmm is the original GMM, which is given.
  score_xt = gmm_t.score(x_traj_rev[:,:,i-1]) # for each time step backward, given the gmm_t, we know the score of each sample.
  eps_z = np.random.randn(*x0.shape)
  x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) + score_xt * dt * sigma**(2*t)

###########################
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.modules.loss import MSELoss

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreModel_Time(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ): #todo why is there a sigma here?
    super().__init__()
    self.embed = GaussianFourierProjection(10, scale=1)
    self.net = nn.Sequential(nn.Linear(12, 50),
               nn.Tanh(),
               nn.Linear(50,50),
               nn.Tanh(),
               nn.Linear(50,2)) # why is the output 2? 2-dim Gaussian distribution
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    t_embed = self.embed(t)
    pred = self.net(torch.cat((x,t_embed),dim=1))       # for each sample, we predict a score with shape (2, ). So pred's shape is (batch_size, 2)
    pred = pred / self.marginal_prob_std_f(t)[:, None,] #todo why dividing a marginal_prob_std_f(t) here, which might offset the effect of t.
    return pred   # the shape of marginal_prob_std_f(t) is (batch_size, )

def reverse_diffusion_time_dep(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False):
  lambdaT = (sigma**2 - 1) / (2 * np.log(sigma))
  xT = np.sqrt(lambdaT) * np.random.randn(sampN, ndim)
  x_traj_rev = np.zeros((*xT.shape, nsteps, ))
  x_traj_rev[:,:,0] = xT
  dt = 1 / nsteps
  for i in range(1, nsteps):        
    t = 1 - i * dt                  # time flying back e.g. nsteps=200, i=1, t=199/200=0.995, i=2, t=198/200=0.99, i=3, t=197/200=0.985
    tvec = torch.ones((sampN)) * t  # e.g. tensor([0.9950, 0.9950, 0.9950,  ..., 0.9950, 0.9950, 0.9950])
    eps_z = np.random.randn(*xT.shape)
    if exact:                       # if gmm is known, we can use the exact score function
      gmm_t = diffuse_gmm(score_model_td, t, sigma) # we obtain the diffused_gmm at time t and calculate the score.
      score_xt = gmm_t.score(x_traj_rev[:,:,i-1])   # This is where the exact score function is used.
    else:
      with torch.no_grad():
        # score_xt = score_model_td(torch.cat((torch.tensor(x_traj_rev[:,:,i-1]).float(),tvec),dim=1)).numpy()
        score_xt = score_model_td(torch.tensor(x_traj_rev[:,:,i-1]).float(), tvec).numpy() # use the trained model instead of .score method.
    x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) + score_xt * dt * sigma**(2*t)
  return x_traj_rev

def sample_X_and_score(gmm, trainN=10000, testN=2000):
  X_train,_,_ = gmm.sample(trainN)
  y_train = gmm.score(X_train)
  X_test,_,_ = gmm.sample(testN)
  y_test = gmm.score(X_test)
  X_train_tsr = torch.tensor(X_train).float()
  y_train_tsr = torch.tensor(y_train).float()
  X_test_tsr = torch.tensor(X_test).float()
  y_test_tsr = torch.tensor(y_test).float()
  return X_train_tsr, y_train_tsr, X_test_tsr, y_test_tsr

def sample_X_and_score_t_depend(gmm, trainN=10000, testN=2000, sigma=5, partition=20, EPS=0.02):
  """Uniformly partition [0,1] and sample t from it, and then
  sample x~ p_t(x) and compute \nabla \log p_t(x)
  finally return the dataset x, score, t (train and test)
  """
  trainN_part, testN_part = trainN //partition, testN //partition
  X_train_col, y_train_col, X_test_col, y_test_col, T_train_col, T_test_col = [], [], [], [], [], []
  for t in np.linspace(EPS, 1.0, partition):
    gmm_dif = diffuse_gmm(gmm, t, sigma)
    X_train_tsr, y_train_tsr, X_test_tsr, y_test_tsr = \
      sample_X_and_score(gmm_dif, trainN=trainN_part, testN=testN_part, )
    T_train_tsr, T_test_tsr = t * torch.ones(trainN_part), t * torch.ones(testN_part)
    X_train_col.append(X_train_tsr)
    y_train_col.append(y_train_tsr)
    X_test_col.append(X_test_tsr)
    y_test_col.append(y_test_tsr)
    T_train_col.append(T_train_tsr)
    T_test_col.append(T_test_tsr)
  X_train_tsr = torch.cat(X_train_col, dim=0)
  y_train_tsr = torch.cat(y_train_col, dim=0)
  X_test_tsr = torch.cat(X_test_col, dim=0)
  y_test_tsr = torch.cat(y_test_col, dim=0)
  T_train_tsr = torch.cat(T_train_col, dim=0)
  T_test_tsr = torch.cat(T_test_col, dim=0)
  return X_train_tsr, y_train_tsr, T_train_tsr, X_test_tsr, y_test_tsr, T_test_tsr

sigma = 10

X_train, y_train, T_train, X_test, y_test, T_test = sample_X_and_score_t_depend(gmm, sigma=sigma, trainN=100000, testN=2000,
                              partition=1000, EPS=0.0001)

# score_norm = y_train.norm(dim=1)
# samp_norm = X_train.norm(dim=1)

"""### Approximate time dependent score"""

score_model_analy = ScoreModel_Time(sigma=sigma, )

optim = Adam(score_model_analy.parameters(), lr=0.001)
loss_fun = MSELoss()
pbar = tqdm.trange(250)
std_vec = marginal_prob_std(T_train, sigma)
for ep in pbar:
  y_pred = score_model_analy(X_train, T_train) #todo, no time effect
  # loss = loss_fun(y_train, y_pred)
  loss = torch.mean(torch.sum((y_pred - y_train)**2 * std_vec[:, None], dim=(1))) #todo why is there a std_vec here? weigting loss?
  optim.zero_grad()
  loss.backward()
  optim.step()
  pbar.set_description(f"step {ep} loss {loss.item():.3f}")
  if ep == 0:
    print(f"step {ep} loss {loss.item():.3f}")
  if ep % 25==0:
    y_pred_test = score_model_analy(X_test, T_test)
    loss_test = loss_fun(y_test, y_pred_test)
    print(f"step {ep} test loss {loss.item():.3f}")

sampN = 2000

x_traj_rev_analy_pred = reverse_diffusion_time_dep(score_model_analy, sampN=sampN, sigma=sigma)
x_traj_rev = reverse_diffusion_time_dep(gmm, sampN=sampN, sigma=sigma, exact=True)
