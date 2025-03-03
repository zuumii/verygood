from calibration.calib_models import (
    CalibrationModelPlanar3RComplNl,
)
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import random


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
    calib_with_load = True
    compliance_model = "Lin" # The options are Lin or Quad or Cubic 
    
    #If using real dataset
    use_small_dataset = False 
    validation_ratio= 0.2 #size of validation part of the dataset / size of all the dataset
    project_on_plane=True #Option to fit and Project the datapoints on a plane before using them in the optimization loop
    
    #If using 2 splines
    use_two_splines = False 
    alpha=0.5 #the fraction tau_transition/tau_max
    
    #If using synthetic data   
    use_fake_data = False
    num_pts=500
    noise_level=0.0000
    seed=1
    
    if compliance_model == "Lin":
        num_comp_param=1
    elif compliance_model == "Quad":
        num_comp_param=2
    elif compliance_model == "Cubic":
        num_comp_param=3
    else: 
        raise ValueError(f"Unsupported Compliance Model: {compliance_model}")
    
    
    if calib_with_load:
        filename_load_fct = "load_func_holder_5kg.pkl"
        if use_small_dataset:        
            filename_data = "data/JT100_holder_with_load.tri"
        else:
            filename_data = "data/JT500_holder_with_load.tri"
            
    else:        
        filename_load_fct = "load_func_holder_only.pkl"
        if use_small_dataset: 
            filename_data = "data/JT100_holder_only.tri" 
        else:
            filename_data = "data/JT500_holder_only.tri"   
    
    wrench_fct = None
    with open(filename_load_fct, "rb") as f:
        wrench_fct = pickle.load(f)
        
    err_params_zero = np.zeros((12+num_comp_param*4,)) 

    if use_fake_data:
        np.random.seed(seed+10)
        noise_calib=np.random.normal(0,noise_level,(num_pts,2))
        np.random.seed(seed+30)
        noise_valid=np.random.normal(0,noise_level,(num_pts,2))
        
        if compliance_model == "Cubic":
            err_params_gt= [-6.36236267e-03, -5.49660343e-04, -4.69902594e-04,  5.40976174e-05, -1.12037179e-05,  1.20539419e-06,
                            4.28820339e-35,  7.07404136e-05,  2.49662289e-03,  6.31024159e-05, -1.25767623e-05, -4.79546170e-06,
                            -3.32923954e-29,  9.79377506e-31, -6.70531769e-30,  4.47019452e-05, -1.98230925e-04,  2.17380214e-04,
                            4.97167260e-29, -2.89458957e-04, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00] 
        elif compliance_model == "Lin":
            err_params_gt= [-1.66730280e-02,5.94729485e-05,-6.64014605e-04,3.90348425e-05,
                        1.25820949e-30,3.43967774e-05,1.93361196e-03,4.49819275e-05,
                        -9.70674805e-32,0.00000000e+00,0.00000000e+00,1.79762791e-05,
                        -4.31288877e-35,-1.37306817e-03,0.00000000e+00,0.00000000e+00]
        else:
            err_params_gt= [-6.36236267e-03, -5.49660343e-04, -4.69902594e-04,  5.40976174e-05, -1.12037179e-05,
                            4.28820339e-35,  7.07404136e-05,  2.49662289e-03,  6.31024159e-05, -1.25767623e-05, 
                            -3.32923954e-29,  9.79377506e-31, -6.70531769e-30,  4.47019452e-05, -1.98230925e-04,
                            4.97167260e-29, -2.89458957e-04, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00] 
            
            
        model_gt = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model)

        # generate fake data
        q_calib_fake =  angle_generator(num_pts=num_pts,joint_limits=joint_limits,seed=1)
        
        tcp_calib_fake = model_gt.generate_fake_data(q=q_calib_fake, err_params=err_params_gt) + noise_calib
        
        q_calib = np.array(q_calib_fake)
        tcp_calib = np.array(tcp_calib_fake)
        
        q_valid_fake =  angle_generator(num_pts=num_pts,joint_limits=joint_limits,seed=10)
        
        tcp_valid_fake = model_gt.generate_fake_data(q=q_valid_fake, err_params=err_params_gt) + noise_valid
        
        q_valid = np.array(q_valid_fake)
        tcp_valid = np.array(tcp_valid_fake)
    else:   
        # load measurement data
        q_calib, tcp_calib = parse_tri_file(filename_data)
        q_calib = np.array(q_calib)[:, [1, 2, 4]]
    
        if project_on_plane:
            tcp_calib= np.array(tcp_calib)
            normal, d = fit_plane_pca(tcp_calib)
            projected_points = project_points_onto_plane(tcp_calib, normal, d)                
            tcp_calib= np.array(projected_points)[:, [1, 2]].tolist()
        else:
            tcp_calib = np.array(tcp_calib)[:, [1, 2]].tolist()   
             
        tcp_valid= np.array(tcp_calib)[:int(num_pts*validation_ratio), :]
        tcp_calib= np.array(tcp_calib)[int(num_pts*validation_ratio):, :]

        q_valid = (q_calib)[:int(num_pts*validation_ratio), :].tolist()
        q_calib=(q_calib)[int(num_pts*validation_ratio):, :].tolist()
        
            
    #To help in convergence in the case where the compliance model is of higher order, we start solving the linear case and increment the order step by step until we reach the desired one
    # After each optimization stage, the output of it is set as the initial guess for the next stage
    
    #Preliminatry optimization
    if compliance_model != "Lin": #For the higher order model only
    
        compliance_model_in = "Lin"
        # set error par switch
        #  per axis
        #  [err trans y, err trans z, err rot about x, compliance terms]
        err_par_switch_in = [
                [True, True, True, True],
                [False, True, True, True],
                [False, False, False, True],
                [False, True, False, False],
            ]
        model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model_in)
        
        model.set_error_par_switch(err_par_switch_in)

        # perform calibration
        solver = CalibrationSolverIpopt()
        solved_params = solver.solve_calibration(
            model=model,
            q=q_calib,
            meas=tcp_calib,
            comp_model=compliance_model_in,
            initial_guess = np.zeros(16) 
        )

        print(solved_params)

        # check result
        mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
        print(f"mean err (initial linear calib) = {mean_err*1000:.3f} mm")
        print(f"max err (initial linear calib) = {max_err*1000:.3f} mm")
        
        if compliance_model == "Cubic": #If it is cubic we need to optimize for the quadratic case before doing the main optimization loop
            
            compliance_model_in = "Quad"
            # set error par switch
            #  per axis
            #  [err trans y, err trans z, err rot about x, compliance terms]
            err_par_switch_in = [
            [True, True, True, True, True],
            [False, True, True, True, True],
            [False, False, False, True, True],
            [False, True, False, False, False],
            ]
            
            solved_params=np.reshape(np.hstack((np.reshape(solved_params,(4,4)),np.zeros((4,1)))),(20,)) #The 4 extra new terms (quadratic terms) are set to zero for initial condition
            
            model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model_in)
            
            model.set_error_par_switch(err_par_switch_in)

            # perform calibration
            solver = CalibrationSolverIpopt()
            solved_params = solver.solve_calibration(
                model=model,
                q=q_calib,
                meas=tcp_calib,
                comp_model=compliance_model_in,
                initial_guess = solved_params 
            )

            print(solved_params)

            # check result
            mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
            print(f"mean err (initial quadratic calib) = {mean_err*1000:.3f} mm")
            print(f"max err (initial quadratic calib) = {max_err*1000:.3f} mm")
    
    else: 
        
        solved_params=err_params_zero
    
    #The main calibration loop where the compliance model is the one chosen at the begining of the code
    model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model)
    
    # check error of uncalirated model
    mean_err_uncalib, max_err_uncalib = model.get_error(
        q=q_calib, meas=tcp_calib, calib_params=err_params_zero
    ) 
    # set error par switch
    #  per axis
    #  [err trans y, err trans z, err rot about x, compliance terms]
    if compliance_model=="Lin":    
        err_par_switch = [
            [True, True, True, True],
            [False, True, True, True],
            [False, False, False, True],
            [False, True, False, False],
        ]
        
    elif compliance_model=="Quad":
        err_par_switch = [
            [True, True, True, True, True],
            [False, True, True, True, True],
            [False, False, False, True, True],
            [False, True, False, False, False],
        ]
        
        solved_params=np.reshape(np.hstack((np.reshape(solved_params,(4,4)),np.zeros((4,1)))),(20,))
        
    elif compliance_model=="Cubic":
        err_par_switch = [
            [True, True, True, True, True, True],
            [False, True, True, True, True, True],
            [False, False, False, True, True, True],
            [False, True, False, False, False, False],
        ]
        
        solved_params=np.reshape(np.hstack((np.reshape(solved_params,(4,5)),np.zeros((4,1)))),(24,))
        
    model.set_error_par_switch(err_par_switch)
    
    solver = CalibrationSolverIpopt()
    solved_params = solver.solve_calibration(
        model=model,
        q=q_calib,
        meas=tcp_calib,
        comp_model=compliance_model,
        initial_guess=solved_params
    )
    
    mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
    
    print(solved_params)
    
    print(f"mean err (uncalib) = {mean_err_uncalib*1000:.3f} mm")
    print(f"max err (uncalib) = {max_err_uncalib*1000:.3f} mm")
    print(f"mean err (calib) = {mean_err*1000:.3f} mm")
    print(f"max err (calib) = {max_err*1000:.3f} mm")
    
    
    #In case the 2 spline option is selected we need to run a fourth optimization loop that builds on the single cubic spline result
    if compliance_model=="Cubic" and use_two_splines:
        tau_tr=np.max(abs(model.get_gravity_torque(q_calib)*alpha),axis=0)
        model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model="Cubic2",tau_tr=tau_tr)
        
        err_par_switch = [
            [True, True, True, True, True, True, True, True, True, True],
            [False, True, True, True, True, True, True, True, True, True],
            [False, False, False, True, True, True, True, True, True, True],
            [False, True, False, False, False, False, False, False, False, False],
        ]
        
        solved_params=np.reshape(solved_params,(4,6))
        solved_params=np.reshape(np.hstack((solved_params,np.zeros((4,1)),solved_params[:,3:])),(40,))
        
        model.set_error_par_switch(err_par_switch)
         
        solver = CalibrationSolverIpopt()
        solved_params = solver.solve_calibration(
            model=model,
            q=q_calib,
            meas=tcp_calib,
            comp_model="Cubic2",
            initial_guess=solved_params
        )
        
        mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
    
        print(solved_params)
        
        print(f"mean err (uncalib) = {mean_err_uncalib*1000:.3f} mm")
        print(f"max err (uncalib) = {max_err_uncalib*1000:.3f} mm")
        print(f"mean err (calib) = {mean_err*1000:.3f} mm")
        print(f"max err (calib) = {max_err*1000:.3f} mm")
        
    mean_err, max_err = model.get_error(q=q_valid, meas=tcp_valid, calib_params=solved_params)

    print(f"mean err (valid) = {mean_err*1000:.3f} mm")
    print(f"max err (valid) = {max_err*1000:.3f} mm")
    
    tau_calib = np.array([model.get_gravity_torque(q) for q in q_calib])
    tau_valid = np.array([model.get_gravity_torque(q) for q in q_valid])
    
    fk_cmd = model.get_symbolic_meas_fct()   
    
    # # Plotting the compliance curves   
    # tau_tr=np.max(abs(model.get_gravity_torque(q_calib)),axis=0) #max tau for each joint
    # figure1, axis = plt.subplots(3, 2)
    
    
    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1, 
    #                 right=0.9, 
    #                 top=0.9, 
    #                 wspace=0.3, 
    #                 hspace=0.5) 
    
    # model.plot_compliance(axis, calib_params=solved_params,tau_tr=tau_tr)
    
    # if use_fake_data:
    #     model_gt.plot_compliance(axis, calib_params=err_params_gt,tau_tr=tau_tr)

   
    # plt.show()
    
    print("done")
    
    


    # =====================================================
    #   1. 配置：阈值 (betas)、PSO 超参数、角度单位
    # =====================================================

    NUM_BETAS = 3
    betas_2 = np.linspace(0, 2, NUM_BETAS)
    betas_3 = np.linspace(0, 2, NUM_BETAS)
    betas_5 = np.linspace(0, 2, NUM_BETAS)

    POP_SIZE = 30
    MAX_ITER = 50
    W_INERTIA = 0.7
    C1, C2 = 1.5, 1.5
    MIN_VAL, MAX_VAL = -0.5, 0.5

    # =====================================================
    #   2. Play 算子与 PI 迟滞模型
    # =====================================================

    def play_operator(x_now, r_prev, beta):
        """
        单步Play算子。
        x_now: 当前角度
        r_prev: 上次输出
        beta: 阈值
        """
        lower, upper = x_now - beta, x_now + beta
        if lower <= r_prev <= upper:
            return r_prev
        return lower if r_prev < lower else upper

    def compute_pi_correction(q_cmd_axis, w_axis, betas_axis):
        """
        针对单个关节的命令角序列 q_cmd_axis，用多个play算子(阈值 betas_axis)加权叠加得到修正量。
        """
        q_cmd_axis = np.array(q_cmd_axis)
        T, K = len(q_cmd_axis), len(betas_axis)
        r_states = np.zeros((K, T))

        # 初始化
        for k in range(K):
            r_states[k, 0] = play_operator(q_cmd_axis[0], 0.0, betas_axis[k])

        # 逐时刻更新
        for t in range(1, T):
            for k in range(K):
                r_states[k, t] = play_operator(q_cmd_axis[t], r_states[k, t - 1], betas_axis[k])

        # 线性加权
        return np.dot(w_axis, r_states)

    def pi_model(q_cmd_full, param):
        """
        对 (N,3) 的命令角序列，使用 param 的 (w2,w3,w5) 分别对 2/3/5 轴做PI修正。
        返回 (N,3) 的修正后角度。
        """
        q_cmd_full = np.array(q_cmd_full)
        q2_cmd, q3_cmd, q5_cmd = q_cmd_full[:, 0], q_cmd_full[:, 1], q_cmd_full[:, 2]

        dq2 = compute_pi_correction(q2_cmd, param['w2'], betas_2)
        dq3 = compute_pi_correction(q3_cmd, param['w3'], betas_3)
        dq5 = compute_pi_correction(q5_cmd, param['w5'], betas_5)

        return np.column_stack([q2_cmd + dq2, q3_cmd + dq3, q5_cmd + dq5])

    # =====================================================
    #   3. 目标函数: 计算 "末端位置误差"
    # =====================================================

    def objective_function(param, q_cmd_data, tcp_data, fk_cmd, solved_params):
        """
        1) 用 pi_model 修正命令角
        2) 用 fk_cmd(修正角, solved_params) 计算末端位置
        3) 与实测 tcp_data 做差，返回 SSE(平方和)
        """
        q_cmd_data = np.array(q_cmd_data)
        tcp_data = np.array(tcp_data)

        q_corrected = pi_model(q_cmd_data, param)
        tcp_model = []

        for q_i in q_corrected:
            tcp_model.append(fk_cmd(q_i, solved_params))

        tcp_model = np.array(tcp_model).squeeze()  # 去除 (N,2,1) 这种多余维度

        diff = tcp_data - tcp_model
        sse = np.sum(np.sum(diff ** 2, axis=1))
        return sse

    # =====================================================
    #   4. 粒子群算法 (PSO) 用于搜索 PI 权重
    # =====================================================

    def pso_optimize(q_calib, tcp_calib, fk_cmd, solved_params):
        """
        粒子群算法搜索 w2,w3,w5 (各自是 NUM_BETAS 维度)，共 3*NUM_BETAS 维。
        返回最佳参数 best_param 和最优误差 gbest_score
        """
        q_calib = np.array(q_calib)
        tcp_calib = np.array(tcp_calib)

        dim = 3 * NUM_BETAS
        pop = MIN_VAL + (MAX_VAL - MIN_VAL) * np.random.rand(POP_SIZE, dim)
        vel = np.zeros_like(pop)

        pbest = pop.copy()
        pbest_scores = np.full(POP_SIZE, np.inf)
        gbest, gbest_score = None, np.inf

        def vec_to_param(vec):
            return {
                'w2': vec[0:NUM_BETAS],
                'w3': vec[NUM_BETAS:2*NUM_BETAS],
                'w5': vec[2*NUM_BETAS:3*NUM_BETAS]
            }

        print("PSO 迭代过程:")
        for it in range(MAX_ITER):
            # 评估当前种群
            for i in range(POP_SIZE):
                param_i = vec_to_param(pop[i])
                score_i = objective_function(param_i, q_calib, tcp_calib, fk_cmd, solved_params)

                # 更新个体最优
                if score_i < pbest_scores[i]:
                    pbest_scores[i] = score_i
                    pbest[i] = pop[i].copy()

                # 更新全局最优
                if score_i < gbest_score:
                    gbest_score = score_i
                    gbest = pop[i].copy()

            # 更新速度和位置
            for i in range(POP_SIZE):
                r1, r2 = random.random(), random.random()
                vel[i] = (W_INERTIA * vel[i]
                        + C1 * r1 * (pbest[i] - pop[i])
                        + C2 * r2 * (gbest - pop[i]))
                pop[i] += vel[i]

            # 每 5 次迭代打印一次日志
            if (it + 1) % 5 == 0:
                print(f"  迭代 {it + 1}/{MAX_ITER}, 当前最佳误差: {gbest_score:.6f}")

        return vec_to_param(gbest), gbest_score

    # =====================================================
    #   5. 主函数: 标定 + 验证
    # =====================================================

    def main_calibrate_pi(q_calib, tcp_calib, q_valid, tcp_valid, fk_cmd, solved_params):
        """
        1) 在 (q_calib, tcp_calib) 上使用 pso_optimize 搜索最佳 PI 权重
        2) 在验证集 (q_valid, tcp_valid) 上评估
        3) 计算并打印 mean / max 误差
        """
        q_calib = np.array(q_calib)
        tcp_calib = np.array(tcp_calib)
        q_valid = np.array(q_valid)
        tcp_valid = np.array(tcp_valid)

        best_param, best_score = pso_optimize(q_calib, tcp_calib, fk_cmd, solved_params)
        valid_error = objective_function(best_param, q_valid, tcp_valid, fk_cmd, solved_params)

        # 计算误差统计
        # 标定集
        q_calib_corrected = pi_model(q_calib, best_param)
        tcp_model_calib = np.array([fk_cmd(q_i, solved_params) for q_i in q_calib_corrected]).squeeze()
        err_calib = np.linalg.norm(tcp_calib - tcp_model_calib, axis=1)
        mean_err_calib = np.mean(err_calib)
        max_err_calib = np.max(err_calib)

        # 验证集
        q_valid_corrected = pi_model(q_valid, best_param)
        tcp_model_valid = np.array([fk_cmd(q_i, solved_params) for q_i in q_valid_corrected]).squeeze()
        err_valid = np.linalg.norm(tcp_valid - tcp_model_valid, axis=1)
        mean_err_valid = np.mean(err_valid)
        max_err_valid = np.max(err_valid)

        print(f"mean err (calib) = {mean_err_calib:.3f} mm")
        print(f"max err (calib) = {max_err_calib:.3f} mm")
        print(f"mean err (valid) = {mean_err_valid:.3f} mm")
        print(f"max err (valid) = {max_err_valid:.3f} mm")

        return best_param

    try:
        best_param_solved = main_calibrate_pi(q_calib, tcp_calib, q_valid, tcp_valid, fk_cmd, solved_params)
    except Exception as e:
        print(f"PI 标定时发生错误: {e}")
        best_param_solved = None

    if best_param_solved:
        print("最优 PI 参数:")
        print("  轴2权重 w2:", best_param_solved['w2'])
        print("  轴3权重 w3:", best_param_solved['w3'])
        print("  轴5权重 w5:", best_param_solved['w5'])
    else:
        print("未能找到最优 PI 参数，请检查输入数据或 PI 计算逻辑。")
    def plot_q_correction(q_cmd, q_corrected):
        """
        分别绘制关节2、3、5的命令角(q_cmd)与矫正后角度(q_corrected)，
        每个关节单独一张图，用两条不同颜色的线进行对比。

        参数:
        q_cmd:        shape (N, 3) - 原始命令角序列(关节2,3,5)
        q_corrected:  shape (N, 3) - 经PI模型修正后的关节角序列
        """
        # 时间步(或采样顺序)
        t = np.arange(len(q_cmd))

        # 关节 2
        plt.figure()
        plt.plot(t, q_cmd[:, 0], color='red', label='q_cmd')
        plt.plot(t, q_corrected[:, 0], color='blue', label='q_corrected')
        plt.xlabel('Time Step')
        plt.ylabel('Joint2 Angle (deg)')
        plt.title('Joint 2')
        plt.legend()

        # 关节 3
        plt.figure()
        plt.plot(t, q_cmd[:, 1], color='red', label='q_cmd')
        plt.plot(t, q_corrected[:, 1], color='blue', label='q_corrected')
        plt.xlabel('Time Step')
        plt.ylabel('Joint3 Angle (deg)')
        plt.title('Joint 3')
        plt.legend()

        # 关节 5
        plt.figure()
        plt.plot(t, q_cmd[:, 2], color='red', label='q_cmd')
        plt.plot(t, q_corrected[:, 2], color='blue', label='q_corrected')
        plt.xlabel('Time Step')
        plt.ylabel('Joint5 Angle (deg)')
        plt.title('Joint 5')
        plt.legend()

        # 显示三张图
        plt.show()
        
    def plot_q_correction_diff(q_cmd, q_corrected):
        """
        绘制关节2、3、5的命令角(q_cmd)与矫正后角度(q_corrected)，
        并绘制它们的差值(误差)曲线以增强可视化效果。

        参数:
        q_cmd:        shape (N, 3) - 原始命令角序列(关节2,3,5)
        q_corrected:  shape (N, 3) - 经PI模型修正后的关节角序列
        """
        q_cmd = np.array(q_cmd)
        q_corrected = np.array(q_corrected)
        t = np.arange(len(q_cmd))
        q_diff = q_corrected - q_cmd  # 计算误差

        fig, axs = plt.subplots(3, 2, figsize=(12, 8))  # 创建 3x2 子图

        joint_labels = ["Joint 2", "Joint 3", "Joint 5"]
        for i in range(3):
            # 原始命令角 vs. 矫正角
            axs[i, 0].plot(t, q_cmd[:, i], color='red', label='q_cmd')
            axs[i, 0].plot(t, q_corrected[:, i], color='blue', label='q_corrected')
            axs[i, 0].set_xlabel('Time Step')
            axs[i, 0].set_ylabel('Angle (deg)')
            axs[i, 0].set_title(f'{joint_labels[i]} (Command vs. Corrected)')
            axs[i, 0].legend()

            # 误差曲线
            axs[i, 1].plot(t, q_diff[:, i], color='green', linestyle='dashed', label='Correction Error')
            axs[i, 1].set_xlabel('Time Step')
            axs[i, 1].set_ylabel('Error (deg)')
            axs[i, 1].set_title(f'{joint_labels[i]} Correction Error')
            axs[i, 1].legend()

        plt.tight_layout()
        plt.show()
        
    # 1) 使用最优参数，对 q_calib 进行 PI 矫正
    q_calib = np.array(q_calib)
    q_calib_corrected = pi_model(q_calib, best_param_solved)
    q_valid = np.array(q_valid)
    q_valid_corrected = pi_model(q_valid, best_param_solved)
    # 2) 调用绘图函数
    # plot_q_correction(q_calib, q_calib_corrected)
    plot_q_correction_diff(q_valid, q_valid_corrected)
    
if __name__ == "__main__":
    main()
