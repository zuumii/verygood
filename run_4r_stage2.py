# run_4r_stage2.py
import torch
import torch.nn as nn
import numpy as np
from calibration.structured_4r_net import Structured4RFKNet  # <-- 引用我们上面那个文件

def run_stage2_nn_model(q_calib, tcp_calib_high, 
                        q_valid, tcp_valid_high,
                        init_lengths,  # [L0,L1,L2,L3]
                        init_angles,   # [A1,A2,A3]
                        num_epochs=500,
                        batch_size=32,
                        learning_rate=1e-3):
    """
    类似你之前的stage2, 但构造 4R 的 Structured4RFKNet。
    init_lengths, init_angles来自你之前的 compute_planar_4links_3angles_zero_pose
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转成 PyTorch tensor
    q_calib = torch.tensor(q_calib, dtype=torch.float32).to(device)
    tcp_calib_high = torch.tensor(tcp_calib_high, dtype=torch.float32).to(device)
    q_valid = torch.tensor(q_valid, dtype=torch.float32).to(device)
    tcp_valid_high = torch.tensor(tcp_valid_high, dtype=torch.float32).to(device)

    # ---------- 实例化 4R网络 ----------
    model = Structured4RFKNet(
        init_lengths=init_lengths,  # 例如 [L0,L1,L2,L3]
        init_angles=init_angles,    # 例如 [A1,A2,A3]
        hidden_dim=32
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        indices = torch.randperm(q_calib.size(0))

        for i in range(0, q_calib.size(0), batch_size):
            idx = indices[i:i+batch_size]
            q_batch = q_calib[idx]
            tcp_batch = tcp_calib_high[idx]

            # forward
            pred = model(q_batch)
            loss = loss_fn(pred, tcp_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 可选：对 base_angles clamp
        with torch.no_grad():
            model.base_angles.clamp_(-np.pi/36, np.pi/36)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(q_valid)
            val_loss = loss_fn(val_pred, tcp_valid_high)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Train Loss={loss.item():.6f}, Val Loss={val_loss.item():.6f}")

    print("\nFinished Training 4R Network + Residual")

    # ---------- 训练结束，评估误差 ----------
    model.eval()
    with torch.no_grad():
        tcp_pred_train = model(q_calib)
        train_error = torch.norm(tcp_pred_train - tcp_calib_high, dim=1)

        tcp_pred_val = model(q_valid)
        val_error = torch.norm(tcp_pred_val - tcp_valid_high, dim=1)

        print(f"\n[TRAIN SET] Mean err: {train_error.mean()*1000:.2f} mm, Max err: {train_error.max()*1000:.2f} mm")
        print(f"[VALID SET] Mean err: {val_error.mean()*1000:.2f} mm, Max err: {val_error.max()*1000:.2f} mm")

    # 查看学到的参数
    print("\n=== Trained 4R Net Params ===")
    L_out = model.link_lengths.data.cpu().numpy()
    print("Link lengths (softplus):", L_out)
    b_out = model.base_angles.data.cpu().numpy() * 180/np.pi
    print("Base angles (deg):", b_out)

    return model, train_losses, val_losses