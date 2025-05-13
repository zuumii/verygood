from typing import List
import torch
import torch.nn as nn
import numpy as np
import utils.utils as ut

def tiny_residual(hidden: int = 8):

    net = nn.Sequential(
        nn.Linear(1, hidden, bias=False),
        nn.Tanh(),
        nn.Linear(hidden, 1, bias=False)
    )
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
    return net


class NNSelectiveComplianceModel(nn.Module):
    def __init__(self, lock: List[bool], k_init: float = 5e-7, hidden: int = 8):
        super().__init__()
        self.lock = lock
        self.log_k = nn.ParameterList([
            nn.Parameter(torch.log(torch.tensor(k_init, dtype=torch.float32))) if l else None
            for l in lock
        ])
        self.res_nets = nn.ModuleList([
            tiny_residual(hidden) if l else None for l in lock
        ])
        self.compliance_networks = self.res_nets  

    def _delta_q_batch(self, tau_norm: torch.Tensor) -> torch.Tensor:
        outs = []
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        for i in range(6):
            if self.lock[i]:
                t = tau_norm[:, i].unsqueeze(1).to(dtype=dtype, device=device)
                k = torch.exp(self.log_k[i]).to(dtype=dtype, device=device)
                outs.append(k * t + self.res_nets[i](t))
            else:
                outs.append(torch.zeros_like(tau_norm[:, i].unsqueeze(1)))
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor):
        q, tau_low, tau_high = x[:, :6], x[:, 6:12], x[:, 12:18]
        dq_low  = self._delta_q_batch(tau_low)
        dq_high = self._delta_q_batch(tau_high)

        fk_low, fk_high = [], []
        for i in range(q.size(0)):
            fk_low.append(self.get_fk(q[i] + dq_low[i]))
            fk_high.append(self.get_fk(q[i] + dq_high[i]))
        return torch.cat(fk_high, dim=0) - torch.cat(fk_low, dim=0)

    def get_fk(self, q: torch.Tensor):
        dtype, device = q.dtype, q.device
        kinvec = torch.tensor([[0, 0, 0.1679], [0, 0, 0.0971], [0, 0, 0.4440],
                               [0.113, 0.11, 0], [0.357, 0, 0], [0.101, 0.08, 0]],
                               dtype=dtype, device=device)
        tool = torch.tensor([0.135, -0.09, -0.07], dtype=dtype, device=device)
        axes = ['z', 'y', 'y', 'x', 'y', 'x']
        T = torch.eye(4, dtype=dtype, device=device)
        for i in range(6):
            T = T @ ut.torch_trans3d(kinvec[i])
            T = T @ ut.torch_rot3d(axes[i], q[i])
        T = T @ ut.torch_trans3d(tool)
        return T[:3, -1].unsqueeze(0)

    def get_dq(self, taul: torch.Tensor, tau_tr: np.ndarray):
        if taul.dim() == 3 and taul.shape[-1] == 1:
            taul = taul.squeeze(-1)
        assert taul.shape[1] == 6
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        tau_tr_tensor = torch.tensor(tau_tr, dtype=dtype, device=device)
        tau_norm = taul.to(dtype=dtype, device=device) / tau_tr_tensor
        return self._delta_q_batch(tau_norm).detach().cpu().numpy()

    def plot_compliance_NN(self, axis, tau_tr):
        num_pts = 1000
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        tau_tr_tensor = torch.tensor(tau_tr, dtype=dtype, device=device)

        τ = torch.stack([
            torch.linspace(-tau_tr[i], tau_tr[i], num_pts, dtype=dtype, device=device)
            for i in range(6)
        ], dim=1)  # Shape: (N, 6)

        q_est = self._delta_q_batch(τ / tau_tr_tensor).detach()

        for i in range(6):
            if self.lock[i]:
                ax = axis[i % 3, i // 3]
                ax.plot(
                    τ[:, i].cpu().numpy(),
                    q_est[:, i].cpu().numpy()
                )
                ax.set_title(f"Joint {i + 1}")
                ax.set_xlabel("τ (Nm)")
                ax.set_ylabel("δq (rad)")