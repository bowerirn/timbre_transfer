import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta=10000.0):
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
    def __init__(
        self, 
        input_dim, 
        num_heads, 
        head_dim, 
        p_drop=0.1, 
        causal=False
    ):
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


    def forward(self, x, cond=None):
        use_rope = cond is None

        if cond is None:
            cond = x

        assert x.shape[-1] == cond.shape[-1], f"x and cond must have the same embedding dim, but got {x.shape[-1]} and {cond.shape[-1]}"

        D = self.head_dim
        H = self.num_heads

        input_shape = x.shape[:-1]  # (B, L)
        cond_shape = cond.shape[:-1]

        q = self.q_proj(x).view(*input_shape, H, D).transpose(1, 2)  # Shape: (B, H, L, D)
        k = self.k_proj(cond).view(*cond_shape, H, D).transpose(1, 2)  # Shape: (B, H, L, D)
        v = self.v_proj(cond).view(*cond_shape, H, D).transpose(1, 2)  # Shape: (B, H, L, D)
        
        if use_rope:
            q, k = self.rope(q, k)
        
        att = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.p_drop if self.training else 0.0, 
            is_causal=self.causal, 
            scale=self.scaling
        )
        
        att = att.transpose(1, 2).contiguous().view(*input_shape, H * D)
        return self.o_proj(att)
    

# As described in https://arxiv.org/pdf/2212.09748
class DiT_Block(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        head_dim,
        attn_pdrop=0.1,
        causal=False,
        ff_scale=2,
        cond_dim=None,   # if None, assume same as input_dim
    ):
        super().__init__()

        if cond_dim is None:
            cond_dim = input_dim

        self.input_dim = input_dim

        # Self-attention
        self.attention = MHA(
            input_dim=input_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            p_drop=attn_pdrop,
            causal=causal,
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * ff_scale),
            nn.Tanh(),
            nn.Linear(input_dim * ff_scale, input_dim),
        )

        self.att_ln = nn.LayerNorm(input_dim)
        self.mlp_ln = nn.LayerNorm(input_dim)

        # AdaLN learned params: γ1, β1, α1, γ2, β2, α2
        self.adaln = nn.Sequential(
            nn.GELU(),
            nn.Linear(cond_dim, 6 * input_dim),
        )

    def forward(self, x, cond):
        B, L, D = x.shape
        assert D == self.input_dim

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(cond).chunk(6, dim=1)

        # Shape: (B, 1, D)
        gamma1 = gamma1.unsqueeze(1)
        beta1  = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2  = beta2.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)

        h = self.att_ln(x)
        h = h * (1 + gamma1) + beta1
        h = self.attention(h)
        x = x + alpha1 * h

        h2 = self.mlp_ln(x)
        h2 = h2 * (1 + gamma2) + beta2
        h2 = self.mlp(h2)
        x = x + alpha2 * h2

        return x
    


class DiT(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        head_dim,
        n_blocks=4,
        attn_pdrop=0.1,
        causal=False,
        ff_scale=2,
        cond_dim=None,   # if None, use input_dim
    ):
        super().__init__()

        if cond_dim is None:
            cond_dim = input_dim

        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.blocks = nn.ModuleList([
            DiT_Block(
                input_dim=input_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                attn_pdrop=attn_pdrop,
                causal=causal,
                ff_scale=ff_scale,
                cond_dim=cond_dim,
            )
            for _ in range(n_blocks)
        ])


        self.t_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.cond_embed = MHA(
            cond_dim, 
            num_heads=num_heads, 
            head_dim=head_dim, 
            p_drop=attn_pdrop,
            causal=False,
        )

        self.ln = nn.LayerNorm(input_dim)
        self.proj_out = nn.Linear(input_dim, input_dim)

    def forward(self, xt, t, cond):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        t_embed = self.t_mlp(t)   # (B, cond_dim)
        t_embed = t_embed.unsqueeze(1)   # (B, 1, cond_dim)

        # Cross attention with t and cond to get a cond vector
        cond_vec = self.cond_embed(t_embed, cond=cond) # (B, 1, cond_dim)
        cond_vec = cond_vec.squeeze(1)   # (B, cond_dim)

        h = xt
        for block in self.blocks:
            h = block(h, cond_vec)

        h = self.ln(h)
        return self.proj_out(h)