from calibration.calib_models import (
    CalibrationModelPlanar3RComplNl,
)
from calibration.calib_NN import *
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from calibration.FK_NN import run_stage2_nn_model
import math



# GoFa 5 params
NDOF = 6
KINVEC = [
    [0, 0, 0.167899996],
    [0, -0.060899999, 0.0970999971],
    [0, 0, 0.444000006],
    [0.112999998, 0.060899999, 0.109999999],
    [0.356999993, 0.056499999, 0],
    [0.101000004, -0.056499999, 0.0799999982],
]

# note: TCP not in RAPID convention, but x axis pointing out of the flange
tcp=[0.135, 0.0, -0.07]

#joint limits for only the planar joint 2,3 and 5
joint_limits=np.array([[-180,180],[-225,85],[0,0]])/180*np.pi


def main() -> None:

    # The basic parameters to set
    compliance_model_gt="Cubic"
    calib_with_high=True
    
    # training parameters 
    num_epochs=100
    learning_rate=0.0005
    batch_size=15
    
    #If using synthetic data
    use_fake_data = False
    num_pts=500
    seed=1
    noise_level=0.0000
    
    #If using real dataset
    validation_ratio=0.2
    use_small_dataset = False
    project_on_plane=False
    
    filename_load_fct_high = "load_func_holder_5kg.pkl"
    if use_small_dataset:        
        filename_data_high = "data/JT100_holder_with_load.tri"
    else:
        filename_data_high = "data/JT500_holder_with_load.tri"
                
                
    filename_load_fct_low = "load_func_holder_only.pkl"
    if use_small_dataset: 
        filename_data_low = "data/JT100_holder_only.tri" 
    else:
        filename_data_low = "data/JT500_holder_only.tri"   
   

    wrench_fct_high = None
    with open(filename_load_fct_high, "rb") as f:
        wrench_fct_high= pickle.load(f)
        
    wrench_fct_low = None
    with open(filename_load_fct_low, "rb") as f:
        wrench_fct_low= pickle.load(f)
        
    
    if use_fake_data:
        np.random.seed(seed+10)
        noise_calib_high=np.random.normal(0,noise_level,(num_pts,2))
        np.random.seed(seed+20)
        noise_calib_low=np.random.normal(0,noise_level,(num_pts,2))
        np.random.seed(seed+30)
        noise_valid_high=np.random.normal(0,noise_level,(num_pts,2))
        np.random.seed(seed+40)
        noise_valid_low=np.random.normal(0,noise_level,(num_pts,2))
        
        if compliance_model_gt == "Cubic":
            err_params_gt= [-6.36236267e-03, -5.49660343e-04, -4.69902594e-04,  5.40976174e-05, -1.12037179e-05,  1.20539419e-06,
                            4.28820339e-35,  7.07404136e-05,  2.49662289e-03,  6.31024159e-05, -1.25767623e-05, -4.79546170e-06,
                            -3.32923954e-29,  9.79377506e-31, -6.70531769e-30,  4*4.47019452e-05, 4*-1.98230925e-04,  4*2.17380214e-04,
                            4.97167260e-29, -2.89458957e-04, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00] 
        else: 
            compliance_model_gt="Lin"
            err_params_gt= [-1.66730280e-02,5.94729485e-05,-6.64014605e-04,3.90348425e-05,
                        1.25820949e-30,3.43967774e-05,1.93361196e-03,4.49819275e-05,
                        -9.70674805e-32,0.00000000e+00,0.00000000e+00,1.79762791e-05,
                        -4.31288877e-35,-1.37306817e-03,0.00000000e+00,0.00000000e+00]
                            
        
        
        model_high = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_high, comp_model=compliance_model_gt)
        model_low = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_low, comp_model=compliance_model_gt)
        
        joint_limits=np.array([[-180,180],[-225,85],[0,0]])/180*np.pi

        # generate fake data
        q_calib_fake =  angle_generator(num_pts=num_pts,joint_limits=joint_limits,seed=1)
        
        tcp_calib_fake_high = model_high.generate_fake_data(q=q_calib_fake, err_params=err_params_gt) + noise_calib_high
        tcp_calib_fake_low = model_low.generate_fake_data(q=q_calib_fake, err_params=err_params_gt) + noise_calib_low
        
        q_calib = np.array(q_calib_fake)
        tcp_calib_high = torch.tensor(tcp_calib_fake_high)
        tcp_calib_low = torch.tensor(tcp_calib_fake_low)
        
        q_valid_fake =  angle_generator(num_pts=num_pts,joint_limits=joint_limits,seed=10)
        
        tcp_valid_fake_high = model_high.generate_fake_data(q=q_valid_fake, err_params=err_params_gt) + noise_valid_high
        tcp_valid_fake_low = model_low.generate_fake_data(q=q_valid_fake, err_params=err_params_gt) + noise_valid_low
        
        q_valid = np.array(q_valid_fake)
        tcp_valid_high = torch.tensor(tcp_valid_fake_high)
        tcp_valid_low = torch.tensor(tcp_valid_fake_low)
        
    else:
        compliance_model="Lin"
            
        model_high = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_high, comp_model=compliance_model)
        model_low = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_low, comp_model=compliance_model)
        
        # load measurement data
        q_calib_high, tcp_calib_high = parse_tri_file(filename_data_high)
        q_calib_low, tcp_calib_low = parse_tri_file(filename_data_low)
        q_low = np.array(q_calib_low)[:, [1, 2, 4]]
        
        num_pts=np.shape(q_calib_high)[0]

        
        q_calib_high =(np.array(q_calib_high)[:, [1, 2, 4]])
        q_calib_low =(np.array(q_calib_low)[:, [1, 2, 4]])
        q_valid=((q_calib_high+q_calib_low)/2)[:int(num_pts*validation_ratio), :].tolist()
        q_calib = ((q_calib_high+q_calib_low)/2)[int(num_pts*validation_ratio):, :].tolist()
        
        if project_on_plane:
            tcp_calib_high = np.array(tcp_calib_high)
            normal, d = fit_plane_pca(tcp_calib_high)
            projected_points = project_points_onto_plane(tcp_calib_high, normal, d)
            tcp_calib_high= np.array(projected_points)[:, [1, 2]].tolist()
            
            tcp_calib_low = np.array(tcp_calib_low)
            normal, d = fit_plane_pca(tcp_calib_low)
            projected_points = project_points_onto_plane(tcp_calib_low, normal, d)
            tcp_calib_low= np.array(projected_points)[:, [1, 2]].tolist()
            
        else:
            tcp_calib_high= np.array(tcp_calib_high)[:, [1, 2]].tolist()
            tcp_calib_low= np.array(tcp_calib_low)[:, [1, 2]].tolist()
            
        tcp_low = tcp_calib_low
            
        tcp_valid_high= torch.tensor(tcp_calib_high)[:int(num_pts*validation_ratio),:]     
        tcp_calib_high= torch.tensor(tcp_calib_high)[int(num_pts*validation_ratio):, :]
        
        tcp_valid_low= torch.tensor(tcp_calib_low)[:int(num_pts*validation_ratio), :]           
        tcp_calib_low= torch.tensor(tcp_calib_low)[int(num_pts*validation_ratio):, :]
        
        # L = model_high.compute_planar_3r_link_lengths()
        # print("The three link lengths (m) =", L)

        # lengths, angles = model_high.compute_planar_3r_link_lengths_and_angles()

        # print("Lengths = [L1, L2, L3] =", lengths, " (meters)")
        # print("Angles = [alpha12, alpha23] =", angles, " (radians)")

        # # 判断正负号: alpha12>0 => v2相对v1逆时针
        # # 若想看度数:
        # angles_deg = angles * 180.0 / np.pi
        # print("Angles (deg) = ", angles_deg)
        
        lengths, angles, p2_np, p3_np ,p5_np, tcp_np = model_high.compute_planar_4links_3angles_zero_pose()
        
        print("\n--- Zero Pose Joint Positions (x,y,z) ---")
        print("p2  =", p2_np)
        print("p3  =", p3_np)
        print("p5  =", p5_np)
        print("TCP =", tcp_np)

        print("四段长度 [L0, L1, L2, L3] =", lengths, " (米)")
        print("三个关节角 [alpha2, alpha3, alpha5] =", angles, " (弧度)")

        # 若要转换为度
        angles_deg = angles * 180.0 / np.pi
        print("关节角 (度) =", angles_deg)
        
        
        
        
    ###################################################################################   
    # 假设这里 lengths 已经由:
    #    lengths, angles, p2_np, p3_np, p5_np, tcp_np = model_high.compute_planar_4links_3angles_zero_pose()
    # 得到, 形如 [L0, L1, L2, L3]
    L0, L1, L2, L3 = lengths
    A1, A2, A3 = angles

    # ========== 继续处理验证集的 q, tcp ==========

    # q_valid 已经是 shape (N,3)，每行是 [q2, q3, q5]
    q_valid_np = np.array(q_low)  # 转成numpy便于后续运算


    tcp_valid_np = tcp_low

    # ========== 4R平面模型预测 ==========

    pred_valid = []
    for i in range(len(q_valid_np)):
        q2, q3, q5 = q_valid_np[i]

        # 第1轴固定 pi/2
        theta0 = math.pi/2
        # 依次累加关节角 q2, q3, q5
        theta1 = theta0 + A1  - q2
        theta2 = theta1 + A2 - q3
        theta3 = theta2 + A3 - q5

        # (y,z) = sum of cos / sin
        y_val = (L0*math.cos(theta0) +
                L1*math.cos(theta1) +
                L2*math.cos(theta2) +
                L3*math.cos(theta3))
        z_val = (L0*math.sin(theta0) +
                L1*math.sin(theta1) +
                L2*math.sin(theta2) +
                L3*math.sin(theta3))

        pred_valid.append([y_val, z_val])

    pred_valid = np.array(pred_valid)  # shape (N,2)

    # ========== 计算误差 ==========
    if isinstance(pred_valid, torch.Tensor):
        pred_valid = pred_valid.detach().cpu().numpy()

    if isinstance(tcp_valid_np, torch.Tensor):
        tcp_valid_np = tcp_valid_np.detach().cpu().numpy()

    err = pred_valid - tcp_valid_np     # (N,2)
    dist_err = np.linalg.norm(err, axis=1)  # 每个样本的欧氏距离
    mean_err = dist_err.mean()
    max_err = dist_err.max()

    print("\n[4R Model Validation]")
    print(f"Mean distance error: {mean_err:.6f} m")
    print(f"Max distance error:  {max_err:.6f} m")
    print(f"Total samples: {len(dist_err)}")
        
    
    
    
    
    
    
    ###################################################################################
    
    
    # Setting all the data as torch tensor before starting the training
    tau_calib_high=torch.tensor(model_high.get_gravity_torque(q_calib)[:, [1, 2, 4]])
    
    tau_calib_low=torch.tensor(model_low.get_gravity_torque(q_calib)[:, [1, 2, 4]])
    
    tau_valid_high=torch.tensor(model_high.get_gravity_torque(q_valid)[:, [1, 2, 4]])
    
    tau_valid_low=torch.tensor(model_low.get_gravity_torque(q_valid)[:, [1, 2, 4]])
    
    q_calib=torch.tensor(q_calib)
    q_valid=torch.tensor(q_valid)
    
    X_tr=torch.concat((q_calib,tau_calib_low,tau_calib_high),dim=1)
    X_te=torch.concat((q_valid,tau_valid_low,tau_valid_high),dim=1)
    
    y_tr=torch.tensor(tcp_calib_high-tcp_calib_low)
    y_te=torch.tensor(tcp_valid_high-tcp_valid_low)
    
    tau_tr=np.max(abs(model_high.get_gravity_torque(q_calib.detach().numpy())),axis=0)
    
    figure1, axis = plt.subplots(3, 2) 
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.5) 
    
    if use_fake_data:
        model_low.plot_compliance(axis, calib_params=err_params_gt,tau_tr=tau_tr)
        
    
    # print("=== Data Overview ===")

    # # 检查关节角输入
    # print("q_calib shape:", q_calib.shape)
    # print("q_calib sample (first 5 rows):\n", q_calib[:5].cpu().numpy())

    # # 检查 TCP 输出
    # print("tcp_calib_high shape:", tcp_calib_high.shape)
    # print("tcp_calib_high sample (first 5 rows):\n", tcp_calib_high[:5].cpu().numpy())

    # # 验证集
    # print("q_valid sample (first 5 rows):\n", q_valid[:5].cpu().numpy())
    # print("tcp_valid_high sample (first 5 rows):\n", tcp_valid_high[:5].cpu().numpy())

    # # 检查最大最小值，判断单位
    # print("Max joint angle (train):", torch.max(q_calib).item())
    # print("Min joint angle (train):", torch.min(q_calib).item())
  
    ################ STAGE 1: Recovering the Compliance model with NN #################

    
    model = NNMultiLoadComplianceModel()
    
    # Initializing the parameters with a Kaiming distribution as we are using relu as an activation function
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.data.size()) > 1:  
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            else:
                nn.init.zeros_(param.data)
        else:
            nn.init.zeros_(param.data)
    
    # Scaling the intial parameters to help with convergence       
    scale_factor=1e-3
    
    for layer in model.compliance_network2.children():
        if isinstance(layer, nn.Linear):
            layer.weight.data *= scale_factor
    
    for layer in model.compliance_network3.children():
        if isinstance(layer, nn.Linear):
            layer.weight.data *= scale_factor
            
    for layer in model.compliance_network5.children():
        if isinstance(layer, nn.Linear):
            layer.weight.data *= scale_factor

    #Setting the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
        
    loss_values = []
    loss_values_te = []
    
    # Normalizing the input torques between -1 and 1 for better convergence (q is nor normalized as it doesn't go through the NN)
    tau_max=torch.max(torch.abs(X_tr[:,6:]),dim=0).values
    normalization=torch.concatenate((torch.ones(3),tau_max,tau_max))
    X_tr=X_tr/normalization
    
    tau_max=torch.max(torch.abs(X_te[:,6:]),dim=0).values
    
    normalization=torch.concatenate((torch.ones(3),tau_max,tau_max))
    X_te=X_te/normalization
    
    
    # training loop
    for epoch in range(num_epochs):
        for i in range(0, len(X_tr), batch_size):
            #preparing the batch
            Xbatch = torch.tensor(X_tr[i:i+batch_size,:],dtype=torch.float32)
            ybatch = torch.tensor(y_tr[i:i+batch_size,:],dtype=torch.float32)
            #Forward pass
            y_pred_tr = model(Xbatch)
            #Computing the loss
            loss = loss_fn(ybatch, y_pred_tr)
            
            #Calculating the validation loss
            model.eval()
            if i<len(X_te):
                with torch.no_grad():
                    Xbatch_te = torch.tensor(X_te[i:i+batch_size,:],dtype=torch.float32)
                    y_pred_te= model(Xbatch_te)
                    ybatch_te = torch.tensor(y_te[i:i+batch_size,:],dtype=torch.float32)
                    loss_te = loss_fn(y_pred_te, ybatch_te) 
                loss_values_te.append(loss_te.item())
            
            model.train()
            #logging the losses
            loss_values.append(loss.item())
            
            
            #Back propagation
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
        if epoch%10 == 0:
            print(f'Finished epoch {epoch}, train loss {loss}, test loss {loss_te}')
    
    
    ####################### STAGE 2: RECOVERING THE KINEMATIC ERROR TERMS ######################
    
    # compliance_model="Lin" #The kinematic model is just set trivialy, in this step we will use the NN model instead of the regular model. The coefficient will be forced to zero and won't affect calibration
    
    # model_phase2_high= CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_high, comp_model=compliance_model,tau_tr=tau_tr)
    # model_phase2_low= CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_low, comp_model=compliance_model,tau_tr=tau_tr)
    # num_comp_param=1
    # err_params_zero = np.zeros((12+num_comp_param*4,)) 
    
    # err_par_switch = [
    #     [True, True, True, False],
    #     [False, True, True, False],
    #     [False, False, False, False],
    #     [False, True, False, False],
    # ]
    
    # model_phase2_high.set_error_par_switch(err_par_switch)
    # model_phase2_low.set_error_par_switch(err_par_switch)
    
    # # check error of uncalirated model
    # q_calib=q_calib.detach().numpy()
    # tcp_calib_high=tcp_calib_high.detach().numpy()
    # tcp_calib_low=tcp_calib_low.detach().numpy()
    
    # q_valid=q_valid.detach().numpy()
    # tcp_valid_high=tcp_valid_high.detach().numpy()
    # tcp_valid_low=tcp_valid_low.detach().numpy()
    
    # #Calculating the uncalibrated errors
    # mean_err_uncalib_high, max_err_uncalib_high = model_phase2_high.get_error(
    #     q=q_calib, meas=tcp_calib_high, calib_params=err_params_zero
    # ) 
    
    # mean_err_uncalib_low, max_err_uncalib_low = model_phase2_low.get_error(
    #     q=q_calib, meas=tcp_calib_low, calib_params=err_params_zero
    # ) 
    # mean_err_uncalib = (mean_err_uncalib_high+mean_err_uncalib_low)/2
    # max_err_uncalib = max(max_err_uncalib_high,max_err_uncalib_low)
    
    # #TODO: In order to incorporate the NN approach without modifying too much on the previously used structure, we will calculate the deviation due to compliance beforehand (just by passing the torque through the trained NN)
    # #The dq need to be added to the joint targets before using them in the second stage and later on in calibration/validation error computation
    
    # #This might not be the neatest way to code it so you might wan to incorporate it properly in the structure or use a new class for it
    # dq_calib_high= model.get_dq(tau_calib_high,tau_tr)
    # dq_calib_low= model.get_dq(tau_calib_low,tau_tr)

    # solver = CalibrationSolverIpopt()
    # solved_params = solver.solve_calibration_dual_2(
    #     model_high=model_phase2_high,
    #     model_low=model_phase2_low,
    #     q_high=q_calib+dq_calib_high,
    #     q_low=q_calib+dq_calib_low,
    #     meas_high=tcp_calib_high,
    #     meas_low=tcp_calib_low,
    #     initial_guess=err_params_zero,
    #     use_high_load= calib_with_high
    # )
    
    # mean_err_low, max_err_low = model_phase2_low.get_error(q=q_calib+dq_calib_low, meas=tcp_calib_low, calib_params=solved_params)
    
    # if calib_with_high:
    #     mean_err_high, max_err_high = model_phase2_high.get_error(q=q_calib+dq_calib_high, meas=tcp_calib_high, calib_params=solved_params)
    #     mean_err=(mean_err_low+mean_err_high)/2
    #     max_err=max(max_err_high,max_err_low)
    # else:
    #     mean_err=mean_err_low
    #     max_err=max_err_low
        
    # print(solved_params)
    
    # print(f"mean err (uncalib) = {mean_err_uncalib*1000:.3f} mm")
    # print(f"max err (uncalib) = {max_err_uncalib*1000:.3f} mm")
    # print(f"mean err (calib) = {mean_err*1000:.3f} mm")
    # print(f"max err (calib) = {max_err*1000:.3f} mm")
    
    # dq_valid_high= model.get_dq(tau_valid_high,tau_tr)
    # dq_valid_low= model.get_dq(tau_valid_low,tau_tr)
    
    # mean_err_low, max_err_low = model_phase2_low.get_error(q=q_valid+dq_valid_low, meas=tcp_valid_low, calib_params=solved_params)

    #  #If the high load was used in the second step, it should be used in evaluating the error and in validation
    # if calib_with_high:
    #     mean_err_high, max_err_high = model_phase2_high.get_error(q=q_valid+dq_valid_high, meas=tcp_valid_high, calib_params=solved_params)
    #     mean_err=(mean_err_low+mean_err_high)/2
    #     max_err=max(max_err_high,max_err_low)
    # else:
    #     mean_err=mean_err_low
    #     max_err=max_err_low
    
    # print(f"mean err (valid) = {mean_err*1000:.3f} mm")
    # print(f"max err (valid) = {max_err*1000:.3f} mm")
    
    # #plotting the output model from the NN trained earlier
    # model.plot_compliance_NN(axis, tau_tr)   
    
    # #In case the ground truth compliance is linear and we are training single neuron networks, we can compare the weights to the linear coefficient directly
    # if use_fake_data and compliance_model_gt=="Lin":
    #     print(f'True linear compliance coefficient of joint 2 is {err_params_gt[3]} , the estimated one is {model.compliance_network2.weight.data[0][0]}')
    #     print(f'True linear compliance coefficient of joint 3 is {err_params_gt[7]} , the estimated one is {model.compliance_network3.weight.data[0][0]}')
    #     print(f'True linear compliance coefficient of joint 5 is {err_params_gt[11]} , the estimated one is {model.compliance_network5.weight.data[0][0]}') 
            
    # plt.show()
    
    # print("done")
    
    print("====== Running Neural Network FK Model (Stage 2 replacement) ======")
    # run_stage2_nn_model(
    #     q_calib=q_calib.numpy(),               # 记得转换为 numpy 类型
    #     tcp_calib_high=tcp_calib_high.numpy(),
    #     q_valid=q_valid.numpy(),
    #     tcp_valid_high=tcp_valid_high.numpy(),
    #     num_epochs=50000,
    #     batch_size=32,
    #     learning_rate=1e-3
    # )
    from calibration.run_4r_stage2 import run_stage2_nn_model
    model_4r, train_losses, val_losses = run_stage2_nn_model(
    q_calib         = q_calib,
    tcp_calib_high  = tcp_calib_high,
    q_valid         = q_valid,
    tcp_valid_high  = tcp_valid_high,
    init_lengths    = lengths,   # 来自 compute_planar_4links_3angles_zero_pose
    init_angles     = angles,    # same
    num_epochs      = 20000,
    batch_size      = 32,
    learning_rate   = 1e-3
    )


if __name__ == "__main__":
    main()
