# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class FKNet(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
#         super(FKNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, q):
#         x = F.relu(self.fc1(q))
#         x = F.relu(self.fc2(x))
#         tcp = self.fc3(x)
#         return tcp


# def run_stage2_nn_model(q_calib, tcp_calib_high, q_valid, tcp_valid_high, num_epochs=500, batch_size=32, learning_rate=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     q_calib = torch.tensor(q_calib, dtype=torch.float32).to(device)
#     tcp_calib_high = torch.tensor(tcp_calib_high, dtype=torch.float32).to(device)
#     q_valid = torch.tensor(q_valid, dtype=torch.float32).to(device)
#     tcp_valid_high = torch.tensor(tcp_valid_high, dtype=torch.float32).to(device)

#     model = FKNet().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()

#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         model.train()
#         indices = torch.randperm(q_calib.size(0))
#         for i in range(0, q_calib.size(0), batch_size):
#             idx = indices[i:i+batch_size]
#             q_batch = q_calib[idx]
#             tcp_batch = tcp_calib_high[idx]

#             pred = model(q_batch)
#             loss = loss_fn(pred, tcp_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_pred = model(q_valid)
#             val_loss = loss_fn(val_pred, tcp_valid_high)

#         train_losses.append(loss.item())
#         val_losses.append(val_loss.item())

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

#     print("\nFinished Training FK Network")

#     # 最终误差评估
#     model.eval()
#     with torch.no_grad():
#         tcp_pred_train = model(q_calib)
#         tcp_pred_val = model(q_valid)

#         train_error = torch.norm(tcp_pred_train - tcp_calib_high, dim=1)
#         val_error = torch.norm(tcp_pred_val - tcp_valid_high, dim=1)

#         print(f"\n[TRAIN SET] Mean error: {train_error.mean()*1000:.2f} mm, Max error: {train_error.max()*1000:.2f} mm")
#         print(f"[VALID SET] Mean error: {val_error.mean()*1000:.2f} mm, Max error: {val_error.max()*1000:.2f} mm")

#     return model


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class StructuredFKNet(nn.Module):
#     def __init__(self):
#         super(StructuredFKNet, self).__init__()

#         # sin layer weights: shape (6, 3), fixed to express [q1, q1+q2, q1+q2+q3] and their shifted versions for cos
#         fixed_weights = torch.tensor([
#             [1, 0, 0],    # q1
#             [1, 1, 0],    # q1+q2
#             [1, 1, 1],    # q1+q2+q3
#             [1, 0, 0],    # q1 (for cos)
#             [1, 1, 0],    # q1+q2 (for cos)
#             [1, 1, 1],    # q1+q2+q3 (for cos)
#         ], dtype=torch.float32)
#         self.register_buffer("w_sin", fixed_weights)

#         # bias terms to shift into cos via sin(q + pi/2), trainable (also includes angle error)
#         self.b_sin = nn.Parameter(torch.tensor([0, 0, 0, 0.5 * torch.pi, 0.5 * torch.pi, 0.5 * torch.pi]))

#         # output weights (link lengths), learnable
#         self.alpha_x = nn.Parameter(torch.tensor([0.4, 0.3, 0.2]))  # l1, l2, l3 (for x)
#         self.alpha_y = nn.Parameter(torch.tensor([0.4, 0.3, 0.2]))  # l1, l2, l3 (for y)

#     def forward(self, q):
#         # q: (batch_size, 3)
#         trig_input = torch.matmul(q, self.w_sin.t()) + self.b_sin  # (batch, 6)
#         h = torch.sin(trig_input)

#         # x = sum of cosines * link length = h[3:6] * alpha_x
#         # y = sum of sines   * link length = h[0:3] * alpha_y
#         x = torch.sum(h[:, 3:6] * self.alpha_x, dim=1)
#         y = torch.sum(h[:, 0:3] * self.alpha_y, dim=1)

#         return torch.stack([x, y], dim=1)


# def run_stage2_nn_model(q_calib, tcp_calib_high, q_valid, tcp_valid_high, num_epochs=500, batch_size=32, learning_rate=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     q_calib = torch.tensor(q_calib, dtype=torch.float32).to(device)
#     tcp_calib_high = torch.tensor(tcp_calib_high, dtype=torch.float32).to(device)
#     q_valid = torch.tensor(q_valid, dtype=torch.float32).to(device)
#     tcp_valid_high = torch.tensor(tcp_valid_high, dtype=torch.float32).to(device)

#     model = StructuredFKNet().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()

#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         model.train()
#         indices = torch.randperm(q_calib.size(0))
#         for i in range(0, q_calib.size(0), batch_size):
#             idx = indices[i:i+batch_size]
#             q_batch = q_calib[idx]
#             tcp_batch = tcp_calib_high[idx]

#             pred = model(q_batch)
#             loss = loss_fn(pred, tcp_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_pred = model(q_valid)
#             val_loss = loss_fn(val_pred, tcp_valid_high)

#         train_losses.append(loss.item())
#         val_losses.append(val_loss.item())

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

#     print("\nFinished Training Structured FK Network")

#     # 最终误差评估
#     model.eval()
#     with torch.no_grad():
#         tcp_pred_train = model(q_calib)
#         tcp_pred_val = model(q_valid)

#         train_error = torch.norm(tcp_pred_train - tcp_calib_high, dim=1)
#         val_error = torch.norm(tcp_pred_val - tcp_valid_high, dim=1)

#         print(f"\n[TRAIN SET] Mean error: {train_error.mean()*1000:.2f} mm, Max error: {train_error.max()*1000:.2f} mm")
#         print(f"[VALID SET] Mean error: {val_error.mean()*1000:.2f} mm, Max error: {val_error.max()*1000:.2f} mm")
#             # 打印结构参数（连杆长度 & 角度偏差）
#         print("\n=== Trained Parameters ===")
#         print("Link lengths in x direction (alpha_x):", model.alpha_x.data.cpu().numpy())
#         print("Link lengths in y direction (alpha_y):", model.alpha_y.data.cpu().numpy())
        
#         # 将 bias 角度偏差（单位弧度）转成角度方便查看
#         bias_deg = model.b_sin.data.cpu().numpy() * 180.0 / torch.pi
#         print("Angle biases (in degrees):", bias_deg)

#     return model


# import torch
# import torch.nn as nn
# import numpy as np

# class StructuredFKNet(nn.Module):
#     def __init__(self):
#         super(StructuredFKNet, self).__init__()

#         # 可训练的连杆长度参数（共享 x/y）
#         self.link_lengths = nn.Parameter(torch.tensor([0.4, 0.3, 0.2], dtype=torch.float32))

#         # 可训练的角度偏差（单位为弧度），初始化为0
#         self.bias_angles = nn.Parameter(torch.zeros(3, dtype=torch.float32))

#     def forward(self, q):
#         # q: (batch, 3)
#         l1, l2, l3 = self.link_lengths

#         # 角度偏差（前3），cos对应的后3加pi/2
#         b = self.bias_angles
#         b_cos = self.bias_angles + 0.5 * torch.pi

#         theta1 = q[:, 0] + b[0]
#         theta2 = q[:, 0] + q[:, 1] + b[1]
#         theta3 = q[:, 0] + q[:, 1] + q[:, 2] + b[2]

#         x = l1 * torch.cos(theta1 + 0.0) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
#         y = l1 * torch.sin(theta1 + 0.0) + l2 * torch.sin(theta2) + l3 * torch.sin(theta3)

#         return torch.stack([x, y], dim=1)


# def run_stage2_nn_model(q_calib, tcp_calib_high, q_valid, tcp_valid_high, num_epochs=500, batch_size=32, learning_rate=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     q_calib = torch.tensor(q_calib, dtype=torch.float32).to(device)
#     tcp_calib_high = torch.tensor(tcp_calib_high, dtype=torch.float32).to(device)
#     q_valid = torch.tensor(q_valid, dtype=torch.float32).to(device)
#     tcp_valid_high = torch.tensor(tcp_valid_high, dtype=torch.float32).to(device)

#     model = StructuredFKNet().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()

#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         model.train()
#         indices = torch.randperm(q_calib.size(0))
#         for i in range(0, q_calib.size(0), batch_size):
#             idx = indices[i:i+batch_size]
#             q_batch = q_calib[idx]
#             tcp_batch = tcp_calib_high[idx]

#             pred = model(q_batch)
#             loss = loss_fn(pred, tcp_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # 限制角度偏差在 ±5° 范围（±π/36）
#         with torch.no_grad():
#             model.bias_angles.clamp_(-np.pi/36, np.pi/36)

#         model.eval()
#         with torch.no_grad():
#             val_pred = model(q_valid)
#             val_loss = loss_fn(val_pred, tcp_valid_high)

#         train_losses.append(loss.item())
#         val_losses.append(val_loss.item())

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

#     print("\nFinished Training Structured FK Network with Constraints")

#     # 最终误差评估
#     model.eval()
#     with torch.no_grad():
#         tcp_pred_train = model(q_calib)
#         tcp_pred_val = model(q_valid)

#         train_error = torch.norm(tcp_pred_train - tcp_calib_high, dim=1)
#         val_error = torch.norm(tcp_pred_val - tcp_valid_high, dim=1)

#         print(f"\n[TRAIN SET] Mean error: {train_error.mean()*1000:.2f} mm, Max error: {train_error.max()*1000:.2f} mm")
#         print(f"[VALID SET] Mean error: {val_error.mean()*1000:.2f} mm, Max error: {val_error.max()*1000:.2f} mm")

#     # 打印训练后的结构参数
#     print("\n=== Trained Parameters ===")
#     print("Link lengths (shared):", model.link_lengths.data.cpu().numpy())

#     bias_deg = model.bias_angles.data.cpu().numpy() * 180.0 / np.pi
#     print("Angle biases (in degrees):", bias_deg)

#     return model





# import torch
# import torch.nn as nn
# import numpy as np

# class StructuredFKNet(nn.Module):
#     def __init__(self, init_lengths=None, init_bias=None):
#         super(StructuredFKNet, self).__init__()

#         # 初始化杆长
#         if init_lengths is None:
#             init_lengths=[0.444, 0.467, 0.101],
#         self.link_param_raw = nn.Parameter(torch.tensor(init_lengths, dtype=torch.float32))

#         # 初始化角度偏差（弧度）
#         if init_bias is None:
#             init_bias = [0.0, 0.0, 0.0]
#         self.bias_angles = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

#     @property
#     def link_lengths(self):
#         # 保证杆长为正数（使用 softplus 激活）
#         return torch.nn.functional.softplus(self.link_param_raw)

#     def forward(self, q):
#         # q: (batch, 3)
#         l1, l2, l3 = self.link_lengths
#         b = self.bias_angles

#         theta1 = q[:, 0] + b[0]
#         theta2 = q[:, 0] + q[:, 1] + b[1]
#         theta3 = q[:, 0] + q[:, 1] + q[:, 2] + b[2]

#         x = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
#         y = l1 * torch.sin(theta1) + l2 * torch.sin(theta2) + l3 * torch.sin(theta3)

#         return torch.stack([x, y], dim=1)


# def run_stage2_nn_model(q_calib, tcp_calib_high, q_valid, tcp_valid_high, num_epochs=500, batch_size=32, learning_rate=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     q_calib = torch.tensor(q_calib, dtype=torch.float32).to(device)
#     tcp_calib_high = torch.tensor(tcp_calib_high, dtype=torch.float32).to(device)
#     q_valid = torch.tensor(q_valid, dtype=torch.float32).to(device)
#     tcp_valid_high = torch.tensor(tcp_valid_high, dtype=torch.float32).to(device)

#     model = StructuredFKNet(init_lengths=[0.4, 0.3, 0.2], init_bias=[0.0, 0.0, 0.0]).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()

#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         model.train()
#         indices = torch.randperm(q_calib.size(0))
#         for i in range(0, q_calib.size(0), batch_size):
#             idx = indices[i:i+batch_size]
#             q_batch = q_calib[idx]
#             tcp_batch = tcp_calib_high[idx]

#             pred = model(q_batch)
#             loss = loss_fn(pred, tcp_batch)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # 限制角度偏差在 ±5°（弧度 ±π/36）
#         with torch.no_grad():
#             model.bias_angles.clamp_(-np.pi/36, np.pi/36)

#         model.eval()
#         with torch.no_grad():
#             val_pred = model(q_valid)
#             val_loss = loss_fn(val_pred, tcp_valid_high)

#         train_losses.append(loss.item())
#         val_losses.append(val_loss.item())

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

#     print("\nFinished Training Structured FK Network with Constraints")

#     # 最终误差评估
#     model.eval()
#     with torch.no_grad():
#         tcp_pred_train = model(q_calib)
#         tcp_pred_val = model(q_valid)

#         train_error = torch.norm(tcp_pred_train - tcp_calib_high, dim=1)
#         val_error = torch.norm(tcp_pred_val - tcp_valid_high, dim=1)

#         print(f"\n[TRAIN SET] Mean error: {train_error.mean()*1000:.2f} mm, Max error: {train_error.max()*1000:.2f} mm")
#         print(f"[VALID SET] Mean error: {val_error.mean()*1000:.2f} mm, Max error: {val_error.max()*1000:.2f} mm")

#     # 打印训练后的结构参数
#     print("\n=== Trained Parameters ===")
#     print("Link lengths (positive, via softplus):", model.link_lengths.data.cpu().numpy())

#     bias_deg = model.bias_angles.data.cpu().numpy() * 180.0 / np.pi
#     print("Angle biases (in degrees):", bias_deg)

#     return model



import torch
import torch.nn as nn
import numpy as np

class StructuredResidualFKNet(nn.Module):
    def __init__(self, 
                 init_lengths=None,   # 用来初始化的杆长
                 init_bias=None,      # 用来初始化的角度偏置
                 hidden_dim=32        # 残差网络的隐层大小
                ):
        super(StructuredResidualFKNet, self).__init__()

        # 1) 初始化机械臂结构参数（与原StructuredFKNet类似）
        if init_lengths is None:
            init_lengths = [0.444, 0.467, 0.101]
        if init_bias is None:
            init_bias = [0.0, 0.0, 0.0]

        # 原始可学习参数：杆长的“raw值” + 关节角度偏置
        self.link_param_raw = nn.Parameter(torch.tensor(init_lengths, dtype=torch.float32))
        self.bias_angles = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

        # 2) 定义一个残差网络(MLP)，例如两层:
        #    第一层: Linear -> Sine
        #    第二层: Linear (输出2维: Δx, Δy)
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

        # 3) 给残差网络做一个简单的初始化（可选）
        #    让初始的修正量尽量接近 0，从而依赖“物理部分”起主要作用
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def link_lengths(self):
        """
        用 softplus 保证三段杆长始终为正。
        你也可以用 torch.exp() 等方式，只要确保 > 0 即可。
        """
        return torch.nn.functional.softplus(self.link_param_raw)

    def forward(self, q):
        """
        参数:
            q: (batch, 3)  关节角度 [q1, q2, q3], 单位: 弧度
        返回:
            (batch, 2)  末端坐标 [x, y]
        """
        # ========== 1) 物理结构正运动学(3R 平面机械臂) ==========
        l1, l2, l3 = self.link_lengths
        b1, b2, b3 = self.bias_angles

        theta1 = q[:, 0] + b1
        theta2 = q[:, 0] + q[:, 1] + b2
        theta3 = q[:, 0] + q[:, 1] + q[:, 2] + b3

        x_struct = l1*torch.cos(theta1) + l2*torch.cos(theta2) + l3*torch.cos(theta3)
        y_struct = l1*torch.sin(theta1) + l2*torch.sin(theta2) + l3*torch.sin(theta3)

        # ========== 2) 计算残差网络输出 ==========
        # 例如先经过一层Linear -> Sine激活
        hidden = torch.sin(self.fc1(q))  # (batch, hidden_dim)
        delta_xy = self.fc2(hidden)      # (batch, 2)

        # 最终输出 = 物理结构输出 + 残差修正
        x = x_struct + delta_xy[:, 0]
        y = y_struct + delta_xy[:, 1]

        return torch.stack([x, y], dim=1)


def run_stage2_nn_model(q_calib, tcp_calib_high, 
                        q_valid, tcp_valid_high, 
                        num_epochs=500, 
                        batch_size=32, 
                        learning_rate=1e-3):
    """
    训练流程与原先类似，只是把原先的 StructuredFKNet
    换成了我们定义的 StructuredResidualFKNet。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转成 PyTorch 张量
    q_calib = torch.tensor(q_calib, dtype=torch.float32).to(device)
    tcp_calib_high = torch.tensor(tcp_calib_high, dtype=torch.float32).to(device)
    q_valid = torch.tensor(q_valid, dtype=torch.float32).to(device)
    tcp_valid_high = torch.tensor(tcp_valid_high, dtype=torch.float32).to(device)

    # 实例化我们新的网络：结构化 + 残差
    model = StructuredResidualFKNet(
        init_lengths=[0.4, 0.3, 0.2],  # 可以换成更接近真实值
        init_bias=[0.0, 0.0, 0.0],     # 初始偏置
        hidden_dim=32                 # 隐层大小(可调)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ---------- 训练 ----------
        model.train()
        # 打乱训练集索引
        indices = torch.randperm(q_calib.size(0))

        for i in range(0, q_calib.size(0), batch_size):
            idx = indices[i:i+batch_size]
            q_batch = q_calib[idx]
            tcp_batch = tcp_calib_high[idx]

            # 前向计算
            pred = model(q_batch)
            loss = loss_fn(pred, tcp_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 可选：对角度偏置在每个 epoch 后做限制（±5度）
        with torch.no_grad():
            model.bias_angles.clamp_(-np.pi/36, np.pi/36)

        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            val_pred = model(q_valid)
            val_loss = loss_fn(val_pred, tcp_valid_high)

        # 记录损失
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # 每隔50个epoch打印一次
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss.item():.6f}")

    print("\nFinished Training Structured + Residual FK Network")

    # ---------- 训练结束后的误差评估 ----------
    model.eval()
    with torch.no_grad():
        # 训练集误差
        tcp_pred_train = model(q_calib)
        train_error = torch.norm(tcp_pred_train - tcp_calib_high, dim=1)

        # 验证集误差
        tcp_pred_val = model(q_valid)
        val_error = torch.norm(tcp_pred_val - tcp_valid_high, dim=1)

        print(f"\n[TRAIN SET] Mean error: {train_error.mean()*1000:.2f} mm, Max error: {train_error.max()*1000:.2f} mm")
        print(f"[VALID SET] Mean error: {val_error.mean()*1000:.2f} mm, Max error: {val_error.max()*1000:.2f} mm")

    # ---------- 查看训练后的模型参数 ----------
    print("\n=== Trained Parameters ===")
    print("Link lengths (positive, via softplus):", model.link_lengths.data.cpu().numpy())
    bias_deg = model.bias_angles.data.cpu().numpy() * 180.0 / np.pi
    print("Angle biases (in degrees):", bias_deg)

    return model, train_losses, val_losses



def compute_4links_3angles_zero_pose(self):
    """
    计算:
      1) 四段长度 [L0, L1, L2, L3]:
         - L0:  基座(0,0,0) → 关节2
         - L1:  关节2 → 关节3
         - L2:  关节3 → 关节5
         - L3:  关节5 → TCP (工具尖端)

      2) 三个关节处的夹角 [alpha2, alpha3, alpha5], 单位: 弧度
         - alpha2: 在关节2处 (v0, v1) 的有向夹角
         - alpha3: 在关节3处 (v1, v2) 的有向夹角
         - alpha5: 在关节5处 (v2, v3) 的有向夹角
         (逆时针>0, 顺时针<0)

    同时打印并返回关节2、3、5以及 TCP 的 (x, y, z) 坐标，方便排查角度问题。

    注意：
      - 所有关节(1~6)都设 0°
      - 不考虑任何误差/扭矩
      - 仅在“理想刚体”下计算
    """

    import casadi as cs
    import numpy as np
    import math

    # ====== 1) 全零 err_params, 关节角, 扭矩 ======
    total_err_dim = 6 * (6 + self.num_comp_param)
    err_params_zero = cs.SX.zeros(total_err_dim, 1)

    q_full = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    taul_full = [0.0]*6

    # ====== 2) 计算每个关节末端的位姿 ======
    T_k = cs.SX_eye(4)
    joint_positions = []  # 存储关节k的坐标

    for k in range(6):
        start_idx = k*(6+self.num_comp_param)
        end_idx   = (k+1)*(6+self.num_comp_param)
        err_params_joint = err_params_zero[start_idx:end_idx]

        # 固定平移
        ax_trans = ut.trans_mat(self.kinvec[k])
        # 误差平移(=0)
        ax_err_trans = ut.trans_mat(err_params_joint[0:3])

        # 柔性修正(=0)
        if self.num_comp_param == 7:
            q_compl = self.comp_fct(taul_full[k], err_params_joint[6:], self.tau_t[k])
        else:
            q_compl = self.comp_fct(taul_full[k], err_params_joint[6:])

        # 旋转(=0)
        if self.rotation_axes[k] == ut.RotationAxis.X:
            ax_err_rot1 = ut.rot_mat_y(0)
            ax_err_rot2 = ut.rot_mat_z(0)
            ax_err_rot3 = ut.rot_mat_x(0)
        elif self.rotation_axes[k] == ut.RotationAxis.Y:
            ax_err_rot1 = ut.rot_mat_x(0)
            ax_err_rot2 = ut.rot_mat_z(0)
            ax_err_rot3 = ut.rot_mat_y(0)
        else:  # Z
            ax_err_rot1 = ut.rot_mat_x(0)
            ax_err_rot2 = ut.rot_mat_y(0)
            ax_err_rot3 = ut.rot_mat_z(0)

        ax_rot = cs.mtimes(ax_err_rot1, cs.mtimes(ax_err_rot2, ax_err_rot3))
        ax_mat = cs.mtimes(ax_trans, cs.mtimes(ax_err_trans, ax_rot))

        T_k = cs.mtimes(T_k, ax_mat)
        p_k = T_k[0:3, 3]  # 关节 k 的坐标
        joint_positions.append(p_k)

    # ====== 3) 工具末端 ======
    tool_err_dim = 6 + self.num_comp_param
    tool_err = err_params_zero[-tool_err_dim:]
    tool_trans = ut.trans_mat(self.tcp)
    tool_err_trans = ut.trans_mat(tool_err)
    tool_mat = cs.mtimes(tool_trans, tool_err_trans)

    T_tcp = cs.mtimes(T_k, tool_mat)
    p_tcp = T_tcp[0:3, 3]  # TCP坐标

    # ====== 4) 取出关节2,3,5 和基座、TCP ======
    # 在你的索引:
    #   joint2 => p2 = joint_positions[1]
    #   joint3 => p3 = joint_positions[2]
    #   joint5 => p5 = joint_positions[4]
    # 基座 => p0 = [0,0,0]
    p0 = cs.SX([0.0, 0.0, 0.0])
    p2 = joint_positions[1]
    p3 = joint_positions[2]
    p5 = joint_positions[4]

    # ====== 5) 四段向量 v0,v1,v2,v3 ======
    v0 = p2 - p0             # 基座→关节2
    v1 = p3 - p2             # 关节2→关节3
    v2 = p5 - p3             # 关节3→关节5
    v3 = p_tcp - p5          # 关节5→TCP

    # ====== 6) 计算四段长度 ======
    L0 = float(cs.norm_2(v0))
    L1 = float(cs.norm_2(v1))
    L2 = float(cs.norm_2(v2))
    L3 = float(cs.norm_2(v3))
    lengths = np.array([L0, L1, L2, L3], dtype=float)

    # ====== 7) 计算三个关节处的夹角(逆时针为正) ======
    def angle_between_2d(a: np.ndarray, b: np.ndarray) -> float:
        """
        返回 a->b 的有向夹角(弧度),
        若 >0 => 逆时针, <0 => 顺时针.
        """
        cross = a[0]*b[1] - a[1]*b[0]  # 2D "叉乘"
        dot   = a[0]*b[0] + a[1]*b[1]  # 2D 点乘
        return math.atan2(cross, dot)

    # 只取 (y,z) 分量
    import math
    v0_yz = np.array([float(v0[1]),  float(v0[2])])
    v1_yz = np.array([float(v1[1]),  float(v1[2])])
    v2_yz = np.array([float(v2[1]),  float(v2[2])])
    v3_yz = np.array([float(v3[1]),  float(v3[2])])

    alpha2 = angle_between_2d(v0_yz, v1_yz)  # 关节2处
    alpha3 = angle_between_2d(v1_yz, v2_yz)  # 关节3处
    alpha5 = angle_between_2d(v2_yz, v3_yz)  # 关节5处
    angles = np.array([alpha2, alpha3, alpha5], dtype=float)

    # ====== 8) 打印各点坐标(供调试) ======
    def to_np3(p_sx):
        return np.array([float(p_sx[0]), float(p_sx[1]), float(p_sx[2])])

    p2_np  = to_np3(p2)
    p3_np  = to_np3(p3)
    p5_np  = to_np3(p5)
    tcp_np = to_np3(p_tcp)

    print("\n--- Zero Pose Joint Positions (x,y,z) ---")
    print("p2  =", p2_np)
    print("p3  =", p3_np)
    print("p5  =", p5_np)
    print("TCP =", tcp_np)

    print("\n--- Link Lengths (m) ---")
    print(f"L0={L0:.5f}, L1={L1:.5f}, L2={L2:.5f}, L3={L3:.5f}")

    print("\n--- Joint Angles (radians) [alpha2, alpha3, alpha5] ---")
    print(angles)
    print("Joint Angles (deg):", angles * 180.0 / math.pi)

    # ====== 9) 返回结果 ======
    return lengths, angles, p2_np, p3_np, p5_np, tcp_np