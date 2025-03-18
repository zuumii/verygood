import torch
import torch.nn as nn
import torch.optim as optim   
    
    
    # fk_cmd = model.get_symbolic_meas_fct()    
    # # 仅当已有 do_pi_cali==True & w2_sol,w3_sol,w5_sol 等于 None 时，此网络才有意义。
    # # 如果您想无论如何都执行神经网络补偿，直接删除此判断。
    # if (w2_sol is not None) and (w3_sol is not None) and (w5_sol is not None):
    #     print("\n==== Starting Neural Network Residual Training ====")
        
    #     # 1) 先准备数据：将关节角、TCP数据转为 PyTorch 张量
    #     #    - 这里 q_calib, tcp_calib 是分段后的 (N,3)/(N,2)，
    #     #      同理 q_valid, tcp_valid
    #     #    - 我们假设已经经过 PI 补偿：q_corr_np_calib, q_corr_np_valid
    #     #      如果你想让 NN 输入的是“未补偿的 q”，可自行更改。
        
    #     # 转为 numpy
    #     q_corr_np_calib = np.array(q_corr_np_calib)  # shape (N,3)
    #     q_corr_np_valid = np.array(q_corr_np_valid)  # shape (N,3)
    #     tcp_calib_arr   = np.array(tcp_calib)        # shape (N,2)
    #     tcp_valid_arr   = np.array(tcp_valid)
    #     # 准备 pytorch 张量
    #     q_torch_calib  = torch.from_numpy(q_corr_np_calib).float()   # (N,3)
    #     tcp_torch_calib= torch.from_numpy(tcp_calib_arr).float()     # (N,2)
        
    #     # 2) 定义一个“小型网络”，输入=3维关节角，输出=2维TCP残差
    #     #    你也可以自行加层/加神经元
    #     # class ResidualNet(nn.Module):
    #     #     def __init__(self):
    #     #         super(ResidualNet, self).__init__()
    #     #         self.fc1 = nn.Linear(3, 16)
    #     #         self.fc2 = nn.Linear(16, 8)
    #     #         self.fc3 = nn.Linear(8, 2)
    #     #         self.relu= nn.ReLU()
    #     #     def forward(self, x):
    #     #         # x shape: (batch, 3)
    #     #         x = self.relu(self.fc1(x))
    #     #         x = self.relu(self.fc2(x))
    #     #         x = self.fc3(x)     # 输出=2
    #     #         return x
    #     class ResidualNet(nn.Module):
    #         def __init__(self):
    #             super(ResidualNet, self).__init__()
    #             self.fc1 = nn.Linear(3, 16)
    #             self.fc2 = nn.Linear(16, 8)
    #             self.fc3 = nn.Linear(8, 2)
    #             # self.relu = nn.ReLU()   # 不再需要

    #         def forward(self, x):
    #             # 用 torch.sin 替代 ReLU
    #             x = torch.sin(self.fc1(x))  # 第一层
    #             x = torch.sin(self.fc2(x))  # 第二层
    #             x = self.fc3(x)             # 输出层(线性)
    #             return x
        
    #     net = ResidualNet()
        
    #     # 3) 定义优化器 & 损失函数
    #     optimizer = optim.Adam(net.parameters(), lr=5e-3)
    #     criterion = nn.MSELoss()
        
    #     # 4) 训练循环
    #     #    这里演示简单整批(batch)训练，也可以做mini-batch
    #     n_epochs = 5000
    #     for epoch in range(n_epochs):
    #         optimizer.zero_grad()
            
    #         # 先用 fk_cmd 计算“基础TCP”
    #         #   由于 fk_cmd 是 CasADi Function，需要对 q_corr_np_calib 做循环
    #         #   并将结果堆叠
    #         #   (您可以缓存, epoch 内部不变, 这里只是演示)
    #         tcp_pred_base = []
    #         for i in range(len(q_corr_np_calib)):
    #             q_i_3 = q_corr_np_calib[i]   # shape=(3,)
    #             # 调用 fk_cmd
    #             # 注意 fk_cmd 返回 shape=(2,)  (yz)
    #             tcp_i = fk_cmd(q_i_3, solved_params)  # casadi DM(2,1)
    #             tcp_i_np = np.array(tcp_i.full()).squeeze()  # -> shape (2,)
    #             tcp_pred_base.append(tcp_i_np)
    #         tcp_pred_base = np.array(tcp_pred_base)  # (N,2)
            
    #         # 转 pytorch
    #         tcp_pred_base_torch = torch.from_numpy(tcp_pred_base).float()  # (N,2)
            
    #         # 然后让神经网络输出 residual
    #         # net_input = q_corr_np_calib => q_torch_calib (N,3)
    #         residual_out = net(q_torch_calib)  # (N,2)
            
    #         # final tcp_pred = tcp_pred_base + residual_out
    #         #  -> shape (N,2)
    #         tcp_pred_final = tcp_pred_base_torch + residual_out
            
    #         # 计算loss
    #         loss = criterion(tcp_pred_final, tcp_torch_calib)
    #         loss.backward()
    #         optimizer.step()
            
    #         if (epoch+1) % 100 == 0:
    #             print(f"Epoch {epoch+1}/{n_epochs}, Loss={loss.item():.6f}")
        
    #     # 5) 训练结束后, 测试
    #     print("\n==== NN Residual Training Done. Evaluate new error. ====")
        
    #     # 对 valid 数据做同样处理
    #     q_corr_np_valid = np.array(q_corr_np_valid) # (M,3)
    #     tcp_valid_arr   = np.array(tcp_valid)       # (M,2)
        
    #     # 先算基本的 tcp_pred_base_valid
    #     tcp_pred_base_valid = []
    #     for i in range(len(q_corr_np_valid)):
    #         q_i_3v = q_corr_np_valid[i]
    #         tcp_i_v = fk_cmd(q_i_3v, solved_params)
    #         tcp_i_v_np = np.array(tcp_i_v.full()).squeeze()
    #         tcp_pred_base_valid.append(tcp_i_v_np)
    #     tcp_pred_base_valid = np.array(tcp_pred_base_valid)  # (M,2)
        
    #     tcp_pred_base_calib = []
    #     for i in range(len(q_corr_np_calib)):
    #         q_i_3c = q_corr_np_calib[i]
    #         tcp_i_c = fk_cmd(q_i_3c, solved_params)
    #         tcp_i_c_np = np.array(tcp_i_c.full()).squeeze()
    #         tcp_pred_base_calib.append(tcp_i_c_np)
    #     tcp_pred_base_calib = np.array(tcp_pred_base_calib)  # (M,2)
        
    #     # NN 输出 residual
    #     q_torch_valid  = torch.from_numpy(q_corr_np_valid).float()  # (M,3)
    #     tcp_torch_valid= torch.from_numpy(tcp_valid_arr).float()    # (M,2)
    #     q_torch_calib  = torch.from_numpy(q_corr_np_calib).float()  # (M,3)
    #     tcp_torch_calib= torch.from_numpy(tcp_calib_arr).float()    # (M,2)
        
    #     with torch.no_grad():
    #         residual_valid = net(q_torch_valid)  # (M,2)
            
    #     with torch.no_grad():
    #         residual_calib = net(q_torch_calib)  # (M,2)
        
    #     # final tcp
    #     tcp_final_valid = tcp_pred_base_valid + residual_valid.numpy()
    #     # final tcp
    #     tcp_final_calib = tcp_pred_base_calib + residual_calib.numpy()
        
        
    #     # 计算最终误差
    #     diff_valid = tcp_final_valid - tcp_valid_arr
    #     diff_calib = tcp_final_calib - tcp_calib_arr
    #     diff_valid_no_nn = tcp_pred_base_valid- tcp_valid_arr
    #     diff_calib_no_nn = tcp_pred_base_calib - tcp_calib_arr
    #     mse_valid = np.mean(np.linalg.norm(diff_valid, axis=1))
    #     mse_calib = np.mean(np.linalg.norm(diff_calib, axis=1))
    #     mse_valid_no_nn = np.mean(np.linalg.norm(diff_valid_no_nn , axis=1))
    #     mse_calib_no_nn = np.mean(np.linalg.norm(diff_calib_no_nn, axis=1))
    #     print(f"NN final valid error (mean L2) = {mse_valid*1000:.3f} mm")
    #     print(f"NN final calib error (mean L2) = {mse_calib*1000:.3f} mm")
    #     print(f"no NN final valid error (mean L2) = {mse_valid_no_nn*1000:.3f} mm")
    #     print(f"no NN final calib error (mean L2) = {mse_calib_no_nn*1000:.3f} mm")
        
    #     # 也可做更详细分析:  max误差, etc
    #     max_err_valid = np.max(np.linalg.norm(diff_valid, axis=1))
    #     print(f"NN final valid error (max L2) = {max_err_valid*1000:.3f} mm")

    # else:
    #     print("No PI or no solved_params found; skip NN residual training.")
    ################################################################
    #                [ UPDATED CODE: Residual Learning ]           #
    ################################################################
    fk_cmd = model.get_symbolic_meas_fct()    
    # 仅当已有 do_pi_cali==True & w2_sol,w3_sol,w5_sol 等不为 None 时，此网络才有意义。
    # 如果您想无论如何都执行神经网络补偿，直接删除此判断。
    if (w2_sol is not None) and (w3_sol is not None) and (w5_sol is not None):
        print("\n==== Starting Neural Network Residual Training ====")
        
        # -------------------------------------------------------------
        # 1) 准备数据：将 关节角 q_corr + TCP 数据 转为 PyTorch 张量
        #    假设已经有 q_corr_np_calib, q_corr_np_valid, tcp_calib, tcp_valid
        #    (从前面取到的)
        # -------------------------------------------------------------
        q_corr_np_calib = np.array(q_corr_np_calib)  # (N,3)
        q_corr_np_valid = np.array(q_corr_np_valid)  # (M,3)
        tcp_calib_arr   = np.array(tcp_calib)        # (N,2)
        tcp_valid_arr   = np.array(tcp_valid)        # (M,2)

        q_torch_calib   = torch.from_numpy(q_corr_np_calib).float()  # (N,3)
        tcp_torch_calib = torch.from_numpy(tcp_calib_arr).float()    # (N,2)

        # -------------------------------------------------------------
        # 2) 定义一个小型网络 (Sine激活)，输入=3维关节角，输出=2维TCP残差
        # -------------------------------------------------------------
        class ResidualNet(nn.Module):
            def __init__(self):
                super(ResidualNet, self).__init__()
                self.fc1 = nn.Linear(3, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 2)
                # 不再使用 ReLU，改为 Sine

            def forward(self, x):
                # 使用 Sine激活
                x = torch.sin(self.fc1(x))  # 第一层
                x = torch.sin(self.fc2(x))  # 第二层
                x = self.fc3(x)             # 输出层(线性2维)
                return x
        
        net = ResidualNet()
        
        # -------------------------------------------------------------
        # 3) 定义优化器 & 损失函数
        # -------------------------------------------------------------
        optimizer = optim.Adam(net.parameters(), lr=5e-3)  # 可酌情改小，如1e-3、1e-4
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()

        # -------------------------------------------------------------
        # 4) 在“训练循环”外部，计算一次基础TCP & 残差标签
        #    -> residual_target = (真实TCP - 基础TCP)
        # -------------------------------------------------------------
        # 计算基础TCP(tcp_pred_base)：不包含NN修正
        tcp_pred_base = []
        for i in range(len(q_corr_np_calib)):
            q_i_3 = q_corr_np_calib[i]   # shape=(3,)
            tcp_i = fk_cmd(q_i_3, solved_params)       # CasADi DM(2,1)
            tcp_i_np = np.array(tcp_i.full()).squeeze()# (2,)
            tcp_pred_base.append(tcp_i_np)
        tcp_pred_base = np.array(tcp_pred_base)        # (N,2)

        tcp_pred_base_torch = torch.from_numpy(tcp_pred_base).float() # (N,2)
        
        # residual_target = (真实TCP - 基础TCP)
        residual_target = tcp_torch_calib - tcp_pred_base_torch       # (N,2)

        # -------------------------------------------------------------
        # 5) 训练循环:
        #    现在的损失函数= MSE(网络输出, residual_target)
        # -------------------------------------------------------------
        n_epochs = 100000
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # 仅计算网络输出
            residual_out = net(q_torch_calib)  # (N,2)
            
            # Residual Learning：网络只学"剩余误差"
            loss = criterion(residual_out, residual_target)*1000
            
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss={loss.item():.6f}")
        
        # -------------------------------------------------------------
        # 6) 训练结束后, 在 Calib & Valid 数据上评估
        # -------------------------------------------------------------
        print("\n==== NN Residual Training Done. Evaluate new error. ====")
        
        # 先计算 Calib / Valid 基础TCP
        #   tcp_pred_base_calib: shape (N,2)
        #   tcp_pred_base_valid: shape (M,2)
        tcp_pred_base_calib = []
        for i in range(len(q_corr_np_calib)):
            q_i_3c = q_corr_np_calib[i]
            tcp_i_c = fk_cmd(q_i_3c, solved_params)
            tcp_i_c_np = np.array(tcp_i_c.full()).squeeze() # (2,)
            tcp_pred_base_calib.append(tcp_i_c_np)
        tcp_pred_base_calib = np.array(tcp_pred_base_calib)

        tcp_pred_base_valid = []
        for i in range(len(q_corr_np_valid)):
            q_i_3v = q_corr_np_valid[i]
            tcp_i_v = fk_cmd(q_i_3v, solved_params)
            tcp_i_v_np = np.array(tcp_i_v.full()).squeeze() # (2,)
            tcp_pred_base_valid.append(tcp_i_v_np)
        tcp_pred_base_valid = np.array(tcp_pred_base_valid)

        # 转成 torch
        q_torch_valid   = torch.from_numpy(q_corr_np_valid).float()  # (M,3)
        tcp_torch_valid = torch.from_numpy(tcp_valid_arr).float()    # (M,2)

        # 得到 NN 输出
        with torch.no_grad():
            residual_calib = net(q_torch_calib)   # (N,2)
            residual_valid = net(q_torch_valid)   # (M,2)

        # 加回基础TCP => 最终预测
        tcp_final_calib = tcp_pred_base_calib + residual_calib.numpy()
        tcp_final_valid = tcp_pred_base_valid + residual_valid.numpy()

        # 计算最终误差
        diff_calib = tcp_final_calib - tcp_calib_arr
        diff_valid = tcp_final_valid - tcp_valid_arr
        
        # 计算 "无NN" 情况下
        diff_calib_no_nn = tcp_pred_base_calib - tcp_calib_arr
        diff_valid_no_nn = tcp_pred_base_valid - tcp_valid_arr
        
        mse_calib       = np.mean(np.linalg.norm(diff_calib, axis=1))
        mse_valid       = np.mean(np.linalg.norm(diff_valid, axis=1))
        mse_calib_no_nn = np.mean(np.linalg.norm(diff_calib_no_nn, axis=1))
        mse_valid_no_nn = np.mean(np.linalg.norm(diff_valid_no_nn, axis=1))

        print(f"NN final calib error (mean L2) = {mse_calib*1000:.3f} mm")
        print(f"NN final valid error (mean L2) = {mse_valid*1000:.3f} mm")
        print(f"no NN final calib error (mean L2) = {mse_calib_no_nn*1000:.3f} mm")
        print(f"no NN final valid error (mean L2) = {mse_valid_no_nn*1000:.3f} mm")

        max_err_valid = np.max(np.linalg.norm(diff_valid, axis=1))
        print(f"NN final valid error (max L2) = {max_err_valid*1000:.3f} mm")

    else:
        print("No PI or no solved_params found; skip NN residual training.")