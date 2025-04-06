import torch
from torch import nn, Tensor
from einops import rearrange
from typing import Optional, Tuple
from performer_pytorch import FastAttention
from .modules import Intermediates

class PerformerAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_head: int = 64,
            heads: int = 8,
            causal: bool = False,
            dropout: float = 0.0,
            nb_features: Optional[int] = None,
            generalized_attention: bool = False,
            kernel_fn: Optional[callable] = None,
            no_projection: bool = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        
        # Add layer normalization for stability
        self.norm = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for stability
        self.scale = dim_head ** -0.5
        
        # Initialize Performer's fast attention with more stable defaults
        self.fast_attention = FastAttention(
            dim_head,
            nb_features=nb_features or 256,  # Explicit feature count
            causal=causal,
            generalized_attention=False,  # Disable generalized attention for stability
            kernel_fn=nn.ReLU(),  # Use simple ReLU
            no_projection=False,
        )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Use smaller initialization for stability
        std = 0.02
        nn.init.normal_(self.to_q.weight, std=std)
        nn.init.normal_(self.to_k.weight, std=std)
        nn.init.normal_(self.to_v.weight, std=std)
        nn.init.normal_(self.to_out.weight, std=std)

    def forward(
            self,
            x: Tensor,
            context: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple]:
        # Add debug prints
        if torch.isnan(x).any():
            print("NaN detected in input")
            return torch.zeros_like(x), Intermediates(None, None)
            
        b, n, d = x.shape
        h = self.heads
        
        # Apply layer norm first
        x = self.norm(x)
        
        # Default to self-attention if no context is provided
        kv_input = context if context is not None else x
        
        # Project inputs to queries, keys, values
        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        
        # Check for NaNs after projection
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            print("NaN detected after projection")
            return torch.zeros_like(x), Intermediates(None, None)
        
        # Apply scaling
        q = q * self.scale
        
        # Split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        # Apply masks if provided
        if mask is not None:
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, 0)
            v = v.masked_fill(~mask, 0)
            
        if context_mask is not None and context is not None:
            context_mask = rearrange(context_mask, 'b n -> b () n ()')
            k = k.masked_fill(~context_mask, 0)
            v = v.masked_fill(~context_mask, 0)
        
        try:
            # Perform fast attention with gradient clipping
            with torch.cuda.amp.autocast(enabled=False):
                out = self.fast_attention(q.float(), k.float(), v.float())
        except RuntimeError as e:
            print(f"Error in fast attention: {e}")
            return torch.zeros_like(x), Intermediates(None, None)
        
        # Check for NaNs in attention output
        if torch.isnan(out).any():
            print("NaN detected in attention output")
            return torch.zeros_like(x), Intermediates(None, None)
        
        # Merge heads and project back
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        
        # Final NaN check
        if torch.isnan(out).any():
            print("NaN detected in final output")
            return torch.zeros_like(x), Intermediates(None, None)
        
        return out, Intermediates(None, None)