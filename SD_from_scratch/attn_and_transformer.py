import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange # a Python library that provides a concise and powerful way to manipulate tensors using a simple and intuitive syntax

# for review the conclusions, just view the fig/attn_conclusions.jpg

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
        """
        Note: For simplicity reason, we just implemented 1-head attention. 
        Feel free to implement multi-head attention! with fancy tensor manipulations.
        """
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)      # My question: How is jifa's life?
        
        if context_dim is None:
            self.self_attn = True
            self.key   = nn.Linear(hidden_dim, embed_dim, bias=False)  # My friend told me how's jifa's life (not detailed)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False) # detailed version
        
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)   # u23 told me how's jifa's life (not detailed)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)# detailed version
            
    def forward(self, tokens, context=None):
        if self.self_attn:
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)
        
        else:
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)
        
        scoremats = torch.einsum("BTH,BSH->BTS", Q, K) # ([1, 4, 8])
        attnmats = F.softmax(scoremats/np.sqrt(self.embed_dim), dim=-1) # ([1, 4, 8])
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V) # ([1, 4, 64]) 
        return ctx_vecs
   
class TransformerBlock(nn.Module):
  """The transformer block that combines self-attn, cross-attn and feed forward neural net"""
  def __init__(self, hidden_dim, context_dim):
    super(TransformerBlock, self).__init__()
    self.attn_self = CrossAttention(hidden_dim, hidden_dim)
    self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)
    
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.norm3 = nn.LayerNorm(hidden_dim)
    
    # implement a 2 layer MLP with K*hidden_dim hidden units, and nn.GeLU nonlinearity
    self.ffn   = nn.Sequential(
      nn.Linear(hidden_dim, 3*hidden_dim),
      nn.GELU(),
      nn.Linear(3*hidden_dim, hidden_dim)
    )
  
  def forward(self, x, context=None): # ([1, 4, 64])
    x = self.attn_self(self.norm1(x)) + x # residual connection # ([1, 4, 64])
    x = self.attn_cross(self.norm2(x), context) + x # ([1, 4, 64])
    x = self.ffn(self.norm3(x)) + x   # ([1, 4, 64])
    
    return x

class SpatialTransformer(nn.Module):
  def __init__(self, hidden_dim, context_dim):
    super(SpatialTransformer, self).__init__()
    self.transformer = TransformerBlock(hidden_dim, context_dim)
    
  def forward(self, x, context=None):
    b, c, h, w = x.shape
    x_in = x # identity
    # Combine the spatial dimensions and move the channle dimension to the last dimension
    x = rearrange(x, "b c h w -> b (h w) c")
    # Apply the sequence transformer
    x = self.transformer(x, context)
    # Reverse the process
    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
    # Residue
    return x + x_in # x processed by attention mechanism + x_in

if __name__ == "__main__":
    batch_size = 1
    seq_len = 4 # num of char
    hidden_dim = 64
    embed_dim = 32
    context_seq_len = 8
    context_dim = 128
    # context_dim = None
    img_h = 32
    img_w = 32
    img_c = 64
    
    # model = CrossAttention(embed_dim, hidden_dim, context_dim)
    # Initialize the TransformerBlock instead of CrossAttention
    # model = TransformerBlock(hidden_dim, context_dim)
    # Initialize the SpatialTransformer instead of TransformerBlock
    model = SpatialTransformer(hidden_dim, context_dim)
    
    fake_img = torch.randn(batch_size, img_c, img_h, img_w)
    tokens = torch.randn(batch_size, seq_len, hidden_dim) # what my friend told me about the movie
    context = torch.randn(batch_size, context_seq_len, context_dim) # what u23 told me about the movie
    
    # Forward pass
    output = model(fake_img, context)
    
    print(fake_img.shape)
    print(context.shape)
    print(output.shape)  # Expected shape: [batch_size, seq_len(token), hidden_dim()]
    
    