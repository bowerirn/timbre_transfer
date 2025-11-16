import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowModel(nn.Module):
    def __init__(self, model: nn.Module, cfg_drop_prob):
        super().__init__()
        self.model = model
        self.cfg_drop_prob = cfg_drop_prob

    @torch.no_grad()
    def sample(self, cond_inst: torch.Tensor, steps: int = 32, cfg_strength: float = 3.0):
        device = cond_inst.device
        dtype = cond_inst.dtype
        b = cond_inst.size(0)

        x = torch.randn_like(cond_inst, device=device, dtype=dtype)

        dt = 1.0 / steps
        for k in range(steps):
            t_scalar = k * dt

            t = torch.full((b, 1), t_scalar, device=self.device, dtype=self.dtype)

            v_cond = self.model(x, t, cond_inst)

            if cfg_strength is not None and cfg_strength != 0.0:
                zero_cond = torch.zeros_like(cond_inst)
                v_uncond = self.model(x, t, zero_cond)
                v = v_uncond + cfg_strength * (v_cond - v_uncond)
            else:
                v = v_cond

            x = x + dt * v

        return x

    def forward(self, cond_inst: torch.Tensor, target_inst: torch.Tensor):
        x0 = torch.randn_like(target_inst)
        t = torch.rand(x0.shape[0], device=self.device)
        t_expanded = t.view(-1, *[1]*(x0.ndim - 1))

        xt = t_expanded * target_inst + (1 - t_expanded) * x0
        vel = target_inst - x0

        mask = torch.rand_like(t) < self.cfg_drop_prob
        cond = cond_inst.where(~mask.view(-1, *[1]*(x0.ndim - 1)), torch.zeros_like(cond_inst))

        pred_vel = self.model(xt, t, cond)

        return F.mse_loss(pred_vel, vel)
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
