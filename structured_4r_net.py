# structured_4r_net.py
import math
import torch
import torch.nn as nn
import numpy as np

class Structured4RFKNet(nn.Module):
    """
    4R 平面结构 + 残差：
      - 4段长度: L0, L1, L2, L3
      - 3个基础角: A1, A2, A3
        第1轴固定 π/2，后面依次:
          theta1 = theta0 + (A1 - q2)
          theta2 = theta1 + (A2 - q3)
          theta3 = theta2 + (A3 - q5)
      - 残差网络(MLP)输入 [q2,q3,q5], 输出 ∆y,∆z
      - 输出末端 (y,z)
    """
    def __init__(self, 
                 init_lengths,  # [L0, L1, L2, L3], 由外部提供
                 init_angles,   # [A1, A2, A3], 由外部提供
                 hidden_dim=32):
        super().__init__()

        # ---------- 1) 4 段长度 ----------
        # 用 raw + softplus, 确保 >0
        self.link_param_raw = nn.Parameter(torch.tensor(init_lengths, dtype=torch.float32))

        # ---------- 2) 3 个基础角 (可学习或固定) ----------
        # 若想固定，可改成普通tensor而非Parameter
        self.base_angles = nn.Parameter(torch.tensor(init_angles, dtype=torch.float32))

        # ---------- 3) 残差网络 (MLP) ----------
        self.fc1 = nn.Linear(3, hidden_dim)  # 输入 [q2, q3, q5]
        self.fc2 = nn.Linear(hidden_dim, 2)  # 输出 ∆y, ∆z

        # 可选: 全 0 初始化
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def link_lengths(self):
        """
        返回 (L0,L1,L2,L3) > 0 (通过 softplus)
        """
        return torch.nn.functional.softplus(self.link_param_raw)

    def forward(self, q_input):
        """
        q_input: shape (batch,3) => [q2, q3, q5]
        返回 (batch,2) => (y,z)
        """
        device_ = q_input.device  # 保证与输入在同一个 device 上

        # ---------- 1) 读参数 ----------
        L0, L1, L2, L3 = self.link_lengths    # 这四个是标量Tensor(shape=())
        A1, A2, A3 = self.base_angles         # 这三个也是标量Tensor(shape=())

        # ---------- 2) 提取关节输入 ----------
        # q2,q3,q5 的 shape = (batch,)
        q2 = q_input[:, 0]
        q3 = q_input[:, 1]
        q5 = q_input[:, 2]

        # ---------- 3) 计算 4 个累计角度 ----------
        # 第1轴固定 θ0 = π/2 (要做 batch 兼容 => shape(batch,))
        # 用 torch.full_like(q2, math.pi/2) 来构造与 q2 同shape的张量
        theta0 = torch.full_like(q2, math.pi/2)

        # θ1 = θ0 + (A1 - q2)
        #    这里 A1 是 shape=() scalar, 通过广播可与 (batch,) 相加
        theta1 = theta0 + (A1 - q2)
        theta2 = theta1 + (A2 - q3)
        theta3 = theta2 + (A3 - q5)

        # ---------- 4) 累加 (y,z) ----------
        # y = L0 cos(theta0) + L1 cos(theta1) + ...
        # z = L0 sin(theta0) + L1 sin(theta1) + ...
        # L0,L1,L2,L3 是 scalar Tensor, 通过广播与 (batch,) 相乘 => (batch,)
        y_struct = (L0 * torch.cos(theta0) +
                    L1 * torch.cos(theta1) +
                    L2 * torch.cos(theta2) +
                    L3 * torch.cos(theta3))
        z_struct = (L0 * torch.sin(theta0) +
                    L1 * torch.sin(theta1) +
                    L2 * torch.sin(theta2) +
                    L3 * torch.sin(theta3))

        # ---------- 5) 残差网络 (MLP) ----------
        hidden = torch.sin(self.fc1(q_input))  # (batch, hidden_dim)
        delta = self.fc2(hidden)              # (batch, 2)
        dy = delta[:, 0]
        dz = delta[:, 1]

        # 最终输出 (batch, 2)
        y_final = y_struct + dy
        z_final = z_struct + dz

        return torch.stack([y_final, z_final], dim=1)