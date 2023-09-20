import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# Multi-head attention implementation from the first code
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, context_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.key = nn.Linear(context_dim if context_dim else hidden_dim, embed_dim, bias=False)
        self.value = nn.Linear(context_dim if context_dim else hidden_dim, hidden_dim, bias=False)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, tokens, context=None):
        Q = self.query(tokens)
        K = self.key(context if context is not None else tokens)
        V = self.value(context if context is not None else tokens)

        # Transpose heads to the first dimension
        Q, K, V = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), [Q, K, V])

        # Scaled dot-product attention
        attn_weights = torch.einsum("b h n d, b h m d -> b h n m", Q, K) * self.scale
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.einsum("b h n m, b h m d -> b h n d", attn_weights, V)

        # Transpose back to the original shape
        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        output = self.out_proj(attn_output)

        return output

# Transformer Block with Multi-head attention
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads):
        super(TransformerBlock, self).__init__()
        
        self.attn_self = MultiHeadCrossAttention(hidden_dim, hidden_dim, num_heads)
        self.attn_cross = MultiHeadCrossAttention(hidden_dim, hidden_dim, num_heads, context_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3*hidden_dim),
            nn.GELU(),
            nn.Linear(3*hidden_dim, hidden_dim)
        )
  
    def forward(self, x, context=None):
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context) + x
        x = self.ffn(self.norm3(x)) + x
        return x

# Spatial Transformer with Multi-head attention
class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads):
        super(SpatialTransformer, self).__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim, num_heads)
  
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + x_in

# Initialize with num_heads argument
num_heads = 4
hidden_dim = 64
context_dim = 128

# Initialize the model
model = SpatialTransformer(hidden_dim, context_dim, num_heads)

# Generate some dummy data for testing
batch_size = 1
img_h = 32
img_w = 32
img_c = 64
context_seq_len = 8

fake_img = torch.randn(batch_size, img_c, img_h, img_w)
context = torch.randn(batch_size, context_seq_len, context_dim)

# Forward pass
output = model(fake_img, context)

# Display shapes
print(fake_img.shape)
print(context.shape)
print(output.shape)
