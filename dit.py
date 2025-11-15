import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"

        half = head_dim // 2
        inv_freq = rope_theta ** -(torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _compute_cos_sin(self, position_ids: torch.Tensor):
        # Outer product: (B,L) x (D/2) -> (B,L,D/2)
        freqs = torch.einsum("bt,d->btd", position_ids, self.inv_freq)

        # Duplicate frequencies to full head_dim
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, L, head_dim)
        return emb.cos(), emb.sin()

    def forward(
        self,
        q: torch.Tensor,     # Shape: (B, H, L, D)
        k: torch.Tensor,     # Shape: (B, H, L, D)
    ):
        position_ids = torch.arange(q.shape[-2], device=q.device).unsqueeze(0).expand(q.shape[0], -1)
        cos, sin = self._compute_cos_sin(position_ids)

        # Broadcast cos/sin to (B, H, L, D)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

class MHA(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, p_drop=0.1, causal=False):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.hidden_dim = num_heads * self.head_dim

        self.causal = causal
        self.scaling = self.head_dim ** -0.5
        self.p_drop = p_drop

        self.q_proj = nn.Linear(input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(input_dim, self.hidden_dim)
        self.v_proj = nn.Linear(input_dim, self.hidden_dim)
        self.o_proj = nn.Linear(self.hidden_dim, input_dim)

        self.rope = RotaryEmbedding(self.head_dim)


    def forward(self, x):

        D = self.head_dim
        H = self.num_heads

        input_shape = x.shape[:-1]  # (B, L)

        q = self.q_proj(x).view(*input_shape, H, D).transpose(1, 2)  # Shape: (B, H, L, D)
        k = self.k_proj(x).view(*input_shape, H, D).transpose(1, 2)  # Shape: (B, H, L, D)
        v = self.v_proj(x).view(*input_shape, H, D).transpose(1, 2)  # Shape: (B, H, L, D)
        
        q_rot, k_rot = self.rope(q, k)
        
        att = F.scaled_dot_product_attention(
            q_rot, 
            k_rot, 
            v, 
            dropout_p=self.p_drop if self.training else 0.0, 
            is_causal=self.causal, 
            scale=self.scaling
        )
        
        att = att.transpose(1, 2).contiguous().view(*input_shape, H * D)
        return self.o_proj(att)
    


    class DiT_Block(nn.Module):
        def __init__(
                self, 
                input_dim, 
                num_heads, 
                head_dim,
                attn_pdrop = 0.1,
                causal = False,
                ff_scale = 2,

                ):
            super().__init__()

            self.attention = self.attention(input_dim, num_heads, head_dim, p_drop=attn_pdrop, causal=causal)
            
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim * ff_scale),
                nn.Tanh(),
                nn.Linear(input_dim * ff_scale, input_dim)
            )
        
        def forward(self, x, cond):
            pass