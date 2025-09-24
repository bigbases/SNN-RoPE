from typing import Optional

from pathlib import Path
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from .module.spike_encoding import SpikeEncoder

from CPG import CPGLinear, CPG
from CPG_RoPE_hybrid import (
    RotaryEmbedding2D,
    RotaryEmbedding1DSpatial,
    RotaryEmbedding1DTemporal,
    apply_spiking_rotary_pos_emb,
    spike_function
)


tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class SSARoPE(nn.Module):
    """Spiking Self-Attention with RoPE (Rotary Position Embedding)"""
    def __init__(
        self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25,
        use_rope=True, rope_theta=10000.0
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qk_scale = qk_scale
        self.use_rope = use_rope

        # RoPE 2D embedding
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding1DSpatial(
                dim=self.head_dim, 
                max_position_embeddings=length,
                base=rope_theta
            )

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.attn_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr / 2,
            backend=backend,
        )

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

    def forward(self, x):
        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB L D
        
        # Q, K, V 계산
        q_m_out = self.q_m(x_for_qkv)  # TB L D
        q_m_out = (
            self.q_bn(q_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        q_m_out = self.q_lif(q_m_out)
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = (
            self.k_bn(k_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        k_m_out = self.k_lif(k_m_out)
        k = (
            k_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = (
            self.v_bn(v_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        v_m_out = self.v_lif(v_m_out)
        v = (
            v_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # RoPE 적용 (Q와 K에만)
        if self.use_rope:
            try:
                # Get rotary embeddings: (T, 1, 1, L, head_dim//2)
                cos, sin = self.rotary_emb(T, L, device=q.device)
                
                # Apply RoPE to Q and K
                q = apply_spiking_rotary_pos_emb(q, cos, sin, threshold=0.5)
                k = apply_spiking_rotary_pos_emb(k, cos, sin, threshold=0.5)
                
            except Exception as e:
                print(f"RoPE application failed: {e}, using original Q, K")

        # T_lat 제거, SpikformerCPG와 동일한 방식으로 attention 계산
        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        x = attn @ v  # x_shape: T * B * heads * L * D//heads

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        # Final projection
        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x


class MLPRoPE(nn.Module):
    """MLP with CPG integration - same as original"""
    def __init__(
        self,
        length,
        tau,
        common_thr,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = CPGLinear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.fc2 = CPGLinear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.transpose(0, 1).flatten(1, 2)  # B TL D
        x = self.fc1(x)  # B TL H
        x = (
            self.bn1(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, self.hidden_features)
            .contiguous()
        )  # B T L H
        x = self.lif1(x.transpose(0, 1)).transpose(0, 1)  # B T L H
        x = x.flatten(1, 2)  # B TL H
        x = self.fc2(x)  # B TL D
        x = (
            self.bn2(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, D)
            .contiguous()
        )  # B T L D
        x = self.lif2(x.transpose(0, 1))  # T B L D
        return x


class BlockRoPE(nn.Module):
    """Transformer block with RoPE-enhanced attention"""
    def __init__(
        self,
        length,
        tau,
        common_thr,
        dim,
        d_ff,
        heads=8,
        qkv_bias=False,
        qk_scale=0.125,
        use_rope=True,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.attn = SSARoPE(
            length=length,
            tau=tau,
            common_thr=common_thr,
            dim=dim,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.mlp = MLPRoPE(
            length=length,
            tau=tau,
            common_thr=common_thr,
            in_features=dim,
            hidden_features=d_ff,
        )

    def forward(self, x):
        # T B L D
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikformerCPGRoPE(nn.Module):
    """SpikformerCPG with RoPE (Rotary Position Embedding) support"""
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        num_pe_neuron: int = 10,
        pe_type: str = "none",  # Keep original PE disabled since we use RoPE
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 5000,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
        # RoPE specific parameters
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        # Input encoding (same as original SpikformerCPG)
        self.temporal_encoder = SpikeEncoder[self._snn_backend]["conv"](num_steps)
        self.encoder = CPGLinear(input_size, dim, CPG(num_neurons=num_pe_neuron))
        #self.encoder = nn.Linear(input_size, dim)
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        # Transformer blocks with RoPE
        self.blocks = nn.ModuleList(
            [
                BlockRoPE(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                )
                for _ in range(depths)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)

        # Same encoding process as SpikformerCPG
        x = self.temporal_encoder(x)  # B L C -> T B C L
        T, B, _, L = x.shape
        x = x.permute(1, 0, 3, 2)  # B T L C
        x = x.flatten(1, 2)  # B TL C
        x = self.encoder(x)  # B TL D
        x = x.reshape(B, T, L, -1).permute(1, 0, 2, 3)  # T B L D
        x = self.init_lif(x)

        # Apply transformer blocks with RoPE
        for blk in self.blocks:
            x = blk(x)  # T B L D
            
        # Same output processing as SpikformerCPG
        out = x.mean(0)
        return out, out.mean(dim=1)  # B L D, B D

    @property
    def output_size(self):
        return self.dim

    @property
    def hidden_size(self):
        return self.dim 