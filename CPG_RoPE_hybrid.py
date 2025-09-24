import math
import torch
from torch import nn
from dataclasses import dataclass
from spikingjelly.activation_based import surrogate


class RotaryEmbedding2D(nn.Module):
    """2D Rotary Embedding for Time-Space dimensions"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embedding"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, L, device):
        # Ensure inv_freq is on the correct device
        inv_freq = self.inv_freq.to(device)
        
        t_indices = torch.arange(T, device=device).type_as(inv_freq)
        l_indices = torch.arange(L, device=device).type_as(inv_freq)
        
        freqs_t = torch.einsum("i,j->ij", t_indices, inv_freq)  # (T, dim//2)
        freqs_l = torch.einsum("i,j->ij", l_indices, inv_freq)  # (L, dim//2)
        
        cos_t = freqs_t.cos()  # (T, dim//2)
        sin_t = freqs_t.sin()  # (T, dim//2)
        cos_l = freqs_l.cos()  # (L, dim//2)
        sin_l = freqs_l.sin()  # (L, dim//2)
        
        cos_combined = cos_t[:, None, :] * cos_l[None, :, :] - sin_t[:, None, :] * sin_l[None, :, :]
        sin_combined = sin_t[:, None, :] * cos_l[None, :, :] + cos_t[:, None, :] * sin_l[None, :, :]
        
        cos_final = cos_combined[:, None, None, :, :] 
        sin_final = sin_combined[:, None, None, :, :]
        
        return cos_final, sin_final

class RotaryEmbedding1DTemporal(nn.Module):
    """1D Rotary Embedding for Time dimension only (T dimension rotation)"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embedding"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, L, device):
        # Only compute frequencies for Time dimension
        t_indices = torch.arange(T, device=device).type_as(self.inv_freq)
        
        freqs_t = torch.einsum("i,j->ij", t_indices, self.inv_freq)  # (T, dim//2)
        
        cos_t = freqs_t.cos()  # (T, dim//2)
        sin_t = freqs_t.sin()  # (T, dim//2)
        
        # For L dimension: no rotation (identity transformation)
        # Expand T dimension across all L positions
        cos_combined = cos_t[:, None, :].expand(T, L, -1)  # (T, L, dim//2)
        sin_combined = sin_t[:, None, :].expand(T, L, -1)  # (T, L, dim//2)
        
        # Final shape: (T, 1, 1, L, head_dim//2)
        cos_final = cos_combined[:, None, None, :, :] 
        sin_final = sin_combined[:, None, None, :, :]
        
        return cos_final, sin_final


class RotaryEmbedding1DSpatial(nn.Module):
    """1D Rotary Embedding for Spatial dimension only (L dimension rotation)"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embedding"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, L, device):
        # Only compute frequencies for Spatial dimension
        l_indices = torch.arange(L, device=device).type_as(self.inv_freq)
        
        freqs_l = torch.einsum("i,j->ij", l_indices, self.inv_freq)  # (L, dim//2)
        
        cos_l = freqs_l.cos()  # (L, dim//2)
        sin_l = freqs_l.sin()  # (L, dim//2)
        
        # For T dimension: no rotation (identity transformation)
        # Expand L dimension across all T positions
        cos_combined = cos_l[None, :, :].expand(T, L, -1)  # (T, L, dim//2)
        sin_combined = sin_l[None, :, :].expand(T, L, -1)  # (T, L, dim//2)
        
        # Final shape: (T, 1, 1, L, head_dim//2)
        cos_final = cos_combined[:, None, None, :, :] 
        sin_final = sin_combined[:, None, None, :, :]
        
        return cos_final, sin_final


def apply_spiking_rotary_pos_emb(x, cos, sin, threshold=0.5):
    """Apply spiking rotary position embedding with Heaviside function"""
    # x shape: (T, B, heads, L, head_dim)
    # cos, sin shape: (T, 1, 1, L, head_dim//2)
    
    # Split x into first half and second half
    x1 = x[..., 0::2]  # Even indices: (T, B, heads, L, head_dim//2)
    x2 = x[..., 1::2]  # Odd indices: (T, B, heads, L, head_dim//2)
    
    # Ensure cos, sin have the correct shape for broadcasting
    # cos, sin: (T, 1, 1, L, head_dim//2) -> expand to match x1, x2
    cos = cos.expand_as(x1)  # (T, B, heads, L, head_dim//2)
    sin = sin.expand_as(x1)  # (T, B, heads, L, head_dim//2)
    
    # Apply rotation transformation (more stable computation)
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    # Combine rotated terms back to original format
    result = torch.zeros_like(x)
    result[..., 0::2] = rotated_even
    result[..., 1::2] = rotated_odd
    
    # result = phase_to_latency(result, 7, 0.2)
    # return result.permute(-1,0,1,2,3,4)
    result = spike_function(result, threshold)
    return result

def spike_function(x, threshold=0.5):
    """Differentiable spike function using surrogate gradient"""
    return surrogate.heaviside(x - threshold) 



