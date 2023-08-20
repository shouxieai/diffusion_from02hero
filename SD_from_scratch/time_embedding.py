import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights (frequencies) during initialization.
    # These weights (frequencies) are fixed during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    # Cosine(2 pi freq x), Sine(2 pi freq x)
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps.
  Allow time repr to input additively from the side of a convolution layer.
  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]
    # this broadcast the 2d tensor to 4d, add the same value across space.

if __name__ == '__main__':
    # Sample data
    x = torch.rand((10,)) # 10 timestep

    # Initialize GaussianFourierProjection
    gaussian_proj = GaussianFourierProjection(embed_dim=8)
    out_gaussian = gaussian_proj(x)
    print(out_gaussian.shape)

    # Initialize Dense
    dense_layer = Dense(input_dim=10, output_dim=20)
    out_dense = dense_layer(x)
    print(out_dense.shape)
