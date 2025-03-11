from calibration.calib_models import (
    CalibrationModelPlanar3RComplNl,
)
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt




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
    compliance_model = "Cubic" # The options are Lin or Quad or Cubic
    
    do_pi_cali = True 
    # do_pi_cali = False 
    
    #If using real dataset
    use_small_dataset = False 
    validation_ratio= 0.2 #size of validation part of the dataset / size of all the dataset
    project_on_plane=True #Option to fit and Project the datapoints on a plane before using them in the optimization loop
    
    # #If using 2 splines
    # use_two_splines = False 
    # alpha=0.5 #the fraction tau_transition/tau_max
    
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
    T_calib = len(q_calib)
    
     # ##PI funcitons
    def play_operator(x_now, r_prev, beta):
        
        lower, upper = x_now - beta, x_now + beta
        if lower <= r_prev <= upper:
            return r_prev
        return lower if r_prev < lower else upper

    def compute_pi_correction(q_cmd_axis, w_axis, betas_axis):
        
        q_cmd_axis = np.array(q_cmd_axis)
        T, K = len(q_cmd_axis), len(betas_axis)
        r_states = np.zeros((K, T))

        
        for k in range(K):
            r_states[k, 0] = play_operator(q_cmd_axis[0], q_cmd_axis[0], betas_axis[k])

        
        for t in range(1, T):
            for k in range(K):
                r_states[k, t] = play_operator(q_cmd_axis[t], r_states[k, t - 1], betas_axis[k])

       
        out = np.dot(w_axis, r_states)   # shape (T,)

        
        sum_w = np.sum(w_axis)         

        # Δq(t) = out(t) - x_cmd(t) * sum_w
        delta = out - q_cmd_axis * sum_w
        return delta

    def pi_model(q_cmd_full, param):
        
        q_cmd_full = np.array(q_cmd_full)
        q2_cmd, q3_cmd, q5_cmd = q_cmd_full[:, 0], q_cmd_full[:, 1], q_cmd_full[:, 2]
        betas_2 = [0.088, 0.218, 0.349]
        betas_3 = [0.088, 0.218, 0.349]
        betas_5 = [0.00, 0.00, 0.00]
        dq2 = compute_pi_correction(q2_cmd, param['w2'], betas_2)
        dq3 = compute_pi_correction(q3_cmd, param['w3'], betas_3)
        dq5 = compute_pi_correction(q5_cmd, param['w5'], betas_5)

        # q_corrected = q_cmd + Δq
        return np.column_stack([q2_cmd + dq2, 
                                q3_cmd + dq3, 
                                q5_cmd + dq5])
    
    def plot_q_correction_diff(q_cmd, q_corrected):
        
        q_cmd = np.array(q_cmd)
        q_corrected = np.array(q_corrected)
        t = np.arange(len(q_cmd))
        q_diff = q_corrected - q_cmd 

        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        joint_labels = ["Joint 2", "Joint 3", "Joint 5"]

        for i in range(3):
            
            axs[i, 0].plot(t, q_corrected[:, i], color='blue', alpha=0.7, linewidth=2, label='q_corrected')
            axs[i, 0].plot(t, q_corrected[:, i], color='blue', linestyle='dotted', linewidth=2.5, zorder=3, label='q_corrected') 
            # axs[i, 0].scatter(t[::5], q_cmd[::5, i], color='red', marker='o', s=10, alpha=0.8, label='q_cmd points')  
            axs[i, 0].plot(t, q_cmd[:, i], color='red', linestyle='dotted', linewidth=2.5, zorder=3, label='q_cmd') 
            # axs[i, 0].scatter(t[::5], q_cmd[::5, i], color='red', marker='o', s=10, alpha=0.8, label='q_cmd points')  

            axs[i, 0].set_xlabel('Time Step')
            axs[i, 0].set_ylabel('Angle (deg)')
            axs[i, 0].set_title(f'{joint_labels[i]} (Command vs. Corrected)')
            axs[i, 0].legend()

            
            axs[i, 1].plot(t, q_diff[:, i], color='green', linestyle='dashed', linewidth=2, label='Correction Error')
            axs[i, 1].set_xlabel('Time Step')
            axs[i, 1].set_ylabel('Error (deg)')
            axs[i, 1].set_title(f'{joint_labels[i]} Correction Error')
            axs[i, 1].legend()

        plt.tight_layout()
        plt.show()
   
                
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
        model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model_in, T_sequence=T_calib)
        
        model.set_error_par_switch(err_par_switch_in)

        # perform calibration
        solver = CalibrationSolverIpopt()
        if do_pi_cali:
            
            solved_params, w2_sol, w3_sol, w5_sol = solver.solve_calibration(
                model=model,
                q=q_calib,
                meas=tcp_calib,
                comp_model=compliance_model_in,
                initial_guess = np.zeros(16),
                use_pi_model=True,         
                num_betas_for_pi=3,
                initial_w2=None,         
                initial_w3=None,
                initial_w5=None 
            )
        else:
            solved_params = solver.solve_calibration(
                model=model,
                q=q_calib,
                meas=tcp_calib,
                comp_model=compliance_model_in,
                initial_guess=np.zeros(16),
                use_pi_model=False
            )
            w2_sol = w3_sol = w5_sol = None
        

        print(solved_params)
        
        
        if (w2_sol is not None) and (w3_sol is not None) and (w5_sol is not None):
            best_param_solved = {
                'w2': w2_sol,
                'w3': w3_sol,
                'w5': w5_sol
            }
            # Offline PI correction
            q_corr_np = pi_model(q_valid, best_param_solved)  # shape (T,3)

            mean_err, max_err = model.get_error(q_corr_np, tcp_valid, solved_params)

            print(f"[with PI] mean err = {mean_err*1000:.3f} mm, max err = {max_err*1000:.3f} mm")
            print("Final solved_params =", solved_params)
            print("Final w2_sol =", w2_sol)
            print("Final w3_sol =", w3_sol)
            print("Final w5_sol =", w5_sol)


        else:
            mean_err, max_err = model.get_error(q_valid, tcp_valid, solved_params)
            print(f"[no PI] mean err = {mean_err*1000:.3f} mm, max err = {max_err*1000:.3f} mm")

            # # check result
            # mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
            # print(f"mean err (initial linear calib) = {mean_err*1000:.3f} mm")
            # print(f"max err (initial linear calib) = {max_err*1000:.3f} mm")
        
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
            
            model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model_in, T_sequence=T_calib)
            
            model.set_error_par_switch(err_par_switch_in)

            # # perform calibration
            # solver = CalibrationSolverIpopt()
            # solved_params = solver.solve_calibration(
            #     model=model,
            #     q=q_calib,
            #     meas=tcp_calib,
            #     comp_model=compliance_model_in,
            #     initial_guess = solved_params 
            # )

            # print(solved_params)

            # # check result
            # mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
            # print(f"mean err (initial quadratic calib) = {mean_err*1000:.3f} mm")
            # print(f"max err (initial quadratic calib) = {max_err*1000:.3f} mm")
            solver = CalibrationSolverIpopt()
            if do_pi_cali:
                
                solved_params, w2_sol, w3_sol, w5_sol = solver.solve_calibration(
                    model=model,
                    q=q_calib,
                    meas=tcp_calib,
                    comp_model=compliance_model_in,
                    initial_guess = solved_params,
                    use_pi_model=True,        
                    num_betas_for_pi=3,
                    initial_w2=w2_sol,          
                    initial_w3=w3_sol,
                    initial_w5=w5_sol 
                )
            else:
                solved_params = solver.solve_calibration(
                    model=model,
                    q=q_calib,
                    meas=tcp_calib,
                    comp_model=compliance_model_in,
                    initial_guess= solved_params,
                    use_pi_model=False
                )
                w2_sol = w3_sol = w5_sol = None
            

            print(solved_params)
            
            
            if (w2_sol is not None) and (w3_sol is not None) and (w5_sol is not None):
                best_param_solved = {
                    'w2': w2_sol,
                    'w3': w3_sol,
                    'w5': w5_sol
                }
                # Offline PI correction
                q_corr_np = pi_model(q_valid, best_param_solved)  # shape (T,3)

                mean_err, max_err = model.get_error(q_corr_np, tcp_valid, solved_params)

                print(f"[with PI] mean err = {mean_err*1000:.3f} mm, max err = {max_err*1000:.3f} mm")
                print("Final solved_params =", solved_params)
                print("Final w2_sol =", w2_sol)
                print("Final w3_sol =", w3_sol)
                print("Final w5_sol =", w5_sol)


            else:
                mean_err, max_err = model.get_error(q_valid, tcp_valid, solved_params)
                print(f"[no PI] mean err = {mean_err*1000:.3f} mm, max err = {max_err*1000:.3f} mm")

    
    else: 
        
        solved_params=err_params_zero
    
    #The main calibration loop where the compliance model is the one chosen at the begining of the code
    model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model=compliance_model, T_sequence=T_calib)
    
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
    
    if do_pi_cali:
        solved_params, w2_sol, w3_sol, w5_sol = solver.solve_calibration(
            model=model,
            q=q_calib,
            meas=tcp_calib,
            comp_model=compliance_model,
            initial_guess=solved_params,
            use_pi_model=True,        
            num_betas_for_pi=3,
            initial_w2=w2_sol,         
            initial_w3=w3_sol,
            initial_w5=w5_sol
        )
    else:
        solved_params = solver.solve_calibration(
            model=model,
            q=q_calib,
            meas=tcp_calib,
            comp_model=compliance_model,
            initial_guess=solved_params,
            use_pi_model=False
        )
        w2_sol = w3_sol = w5_sol = None
        
    if (w2_sol is not None) and (w3_sol is not None) and (w5_sol is not None):
        best_param_solved = {
            'w2': w2_sol,
            'w3': w3_sol,
            'w5': w5_sol
        }
        # Offline PI correction
        q_corr_np_calib = pi_model(q_calib, best_param_solved)  # shape (T,3)
        q_corr_np_valid = pi_model(q_valid, best_param_solved)  # shape (T,3)

        mean_err_calib, max_err_calib = model.get_error(q_corr_np_calib, tcp_calib, solved_params)
        mean_err_valid, max_err_valid = model.get_error(q_corr_np_valid, tcp_valid, solved_params)
        

        print(f"[with PI] mean err calib = {mean_err_calib*1000:.3f} mm, max err vaild= {max_err_calib*1000:.3f} mm")
        print(f"[with PI] mean err vaild = {mean_err_valid*1000:.3f} mm, max err vaild= {max_err_valid*1000:.3f} mm")
        print("Final solved_params =", solved_params)
        print("Final w2_sol =", w2_sol)
        print("Final w3_sol =", w3_sol)
        print("Final w5_sol =", w5_sol)
        plot_q_correction_diff(q_valid, q_corr_np_valid)


    else:
        mean_err, max_err = model.get_error(q_valid, tcp_valid, solved_params)
        print(f"[no PI] mean err = {mean_err*1000:.3f} mm, max err = {max_err*1000:.3f} mm")

    
    
    
   

 
    # def objective_function(param, q_cmd_data, tcp_data, fk_cmd, solved_params):
       
    #     q_cmd_data = np.array(q_cmd_data)
    #     tcp_data = np.array(tcp_data)

    #     q_corrected = pi_model(q_cmd_data, param)
    #     tcp_model = []

    #     for q_i in q_corrected:
    #         tcp_model.append(fk_cmd(q_i, solved_params))

    #     tcp_model = np.array(tcp_model).squeeze() 

    #     diff = tcp_data - tcp_model
    #     mae = np.mean(np.linalg.norm(diff, axis=1)) * 1000
    #     return mae

    

    # def pso_optimize(q_calib, tcp_calib, fk_cmd, solved_params, prev_best_param=None):
    #     q_calib = np.array(q_calib)
    #     tcp_calib = np.array(tcp_calib)

    #     dim = 3 * NUM_BETAS
    #     pop = MIN_VAL + (MAX_VAL - MIN_VAL) * np.random.rand(POP_SIZE, dim)

    #    
    #     if prev_best_param is not None:
    #         pop[0] = np.concatenate([prev_best_param['w2'], prev_best_param['w3'], prev_best_param['w5']])

    #     vel = np.zeros_like(pop)
    #     pbest = pop.copy()
    #     pbest_scores = np.full(POP_SIZE, np.inf)
    #     gbest, gbest_score = None, np.inf

    #     def vec_to_param(vec):
    #         return {
    #             'w2': vec[0:NUM_BETAS],
    #             'w3': vec[NUM_BETAS:2*NUM_BETAS],
    #             'w5': vec[2*NUM_BETAS:3*NUM_BETAS]
    #         }

    #     print("PSO iteration process:")
    #     for it in range(MAX_ITER):
    #         for i in range(POP_SIZE):
    #             param_i = vec_to_param(pop[i])
    #             score_i = objective_function(param_i, q_calib, tcp_calib, fk_cmd, solved_params)

    #             if score_i < pbest_scores[i]:
    #                 pbest_scores[i] = score_i
    #                 pbest[i] = pop[i].copy()

    #             if score_i < gbest_score:
    #                 gbest_score = score_i
    #                 gbest = pop[i].copy()

    #         for i in range(POP_SIZE):
    #             r1, r2 = random.random(), random.random()
    #             vel[i] = (W_INERTIA * vel[i]
    #                     + C1 * r1 * (pbest[i] - pop[i])
    #                     + C2 * r2 * (gbest - pop[i]))
    #             pop[i] += vel[i]
    #             pop[i] = np.clip(pop[i], MIN_VAL, MAX_VAL)

    #         if (it + 1) % 5 == 0:
    #             print(f"  Iteration {it + 1}/{MAX_ITER}, Current Best Error: {gbest_score:.6f}")

    #    
    #     if gbest is None:
    #         print("Warning: PSO did not find a valid solution, using default parameters.")
    #         gbest = pop[0]  

    #     return vec_to_param(gbest), gbest_score


    # def main_calibrate_pi(q_calib, tcp_calib, q_valid, tcp_valid, fk_cmd, solved_params, prev_best_param=None):
        
    #     q_calib = np.array(q_calib)
    #     tcp_calib = np.array(tcp_calib)
    #     q_valid = np.array(q_valid)
    #     tcp_valid = np.array(tcp_valid)

    #     best_param, best_score = pso_optimize(q_calib, tcp_calib, fk_cmd, solved_params, prev_best_param)
    #     valid_error = objective_function(best_param, q_valid, tcp_valid, fk_cmd, solved_params)

       
    #     q_calib_corrected = pi_model(q_calib, best_param)
    #     tcp_model_calib = np.array([fk_cmd(q_i, solved_params) for q_i in q_calib_corrected]).squeeze()
    #     err_calib = np.linalg.norm(tcp_calib - tcp_model_calib, axis=1)
    #     mean_err_calib = np.mean(err_calib)
    #     max_err_calib = np.max(err_calib)


    #     q_valid_corrected = pi_model(q_valid, best_param)
    #     tcp_model_valid = np.array([fk_cmd(q_i, solved_params) for q_i in q_valid_corrected]).squeeze()
    #     err_valid = np.linalg.norm(tcp_valid - tcp_model_valid, axis=1)
    #     mean_err_valid = np.mean(err_valid)
    #     max_err_valid = np.max(err_valid)

    #     mean_err_calib = mean_err_calib * 1000
    #     max_err_calib = max_err_calib * 1000
    #     mean_err_valid= mean_err_valid * 1000
    #     max_err_valid = max_err_valid * 1000

    #     # np.set_printoptions(precision=10, suppress=False)  #debug

    #     # for i in range(5):  
    #     #     print(f"Index {i}:")
    #     #     print(f"  tcp_valid[{i}]      = {tcp_valid[i]}")
    #     #     print(f"  tcp_model_valid[{i}] = {tcp_model_valid[i]}")
    #     #     print(f"  Difference[{i}]      = {tcp_valid[i] - tcp_model_valid[i]}\n") #debug

    #     print(f"mean err (calib) = {mean_err_calib:.3f} mm")
    #     print(f"max err (calib) = {max_err_calib:.3f} mm")
    #     print(f"mean err (valid) = {mean_err_valid:.3f} mm")
    #     print(f"max err (valid) = {max_err_valid:.3f} mm")

    #     return best_param
    
    # fk_cmd = model.get_symbolic_meas_fct()
    # ITER = 100 
    # TOL = 0.01
    # q_calib_cor=q_calib
    # q_valid_cor=q_valid
    # prev_err = float("inf")
    # best_param_now=None
    
    
    # for iter in range(ITER):
    #     print(f"=== Iteration {iter + 1} ===")
    #     print("Running IPOPT Calibration...")
    #     solver = CalibrationSolverIpopt()
    #     solved_params = solver.solve_calibration(
    #         model=model,
    #         q=q_calib_cor,
    #         meas=tcp_calib,
    #         comp_model=compliance_model,
    #         initial_guess=solved_params
    #     )
        
    #     mean_err, max_err = model.get_error(q=q_calib_cor, meas=tcp_calib, calib_params=solved_params)
        
    #     print(solved_params)
        
    #     print(f"mean err (uncalib) = {mean_err_uncalib*1000:.3f} mm")
    #     print(f"max err (uncalib) = {max_err_uncalib*1000:.3f} mm")
    #     print(f"mean err (calib) = {mean_err*1000:.3f} mm")
    #     print(f"max err (calib) = {max_err*1000:.3f} mm")
        
        
    #     # #In case the 2 spline option is selected we need to run a fourth optimization loop that builds on the single cubic spline result
    #     # if compliance_model=="Cubic" and use_two_splines:
    #     #     tau_tr=np.max(abs(model.get_gravity_torque(q_calib)*alpha),axis=0)
    #     #     model = CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct, comp_model="Cubic2",tau_tr=tau_tr)
            
    #     #     err_par_switch = [
    #     #         [True, True, True, True, True, True, True, True, True, True],
    #     #         [False, True, True, True, True, True, True, True, True, True],
    #     #         [False, False, False, True, True, True, True, True, True, True],
    #     #         [False, True, False, False, False, False, False, False, False, False],
    #     #     ]
            
    #     #     solved_params=np.reshape(solved_params,(4,6))
    #     #     solved_params=np.reshape(np.hstack((solved_params,np.zeros((4,1)),solved_params[:,3:])),(40,))
            
    #     #     model.set_error_par_switch(err_par_switch)
            
    #     #     solver = CalibrationSolverIpopt()
    #     #     solved_params = solver.solve_calibration(
    #     #         model=model,
    #     #         q=q_calib,
    #     #         meas=tcp_calib,
    #     #         comp_model="Cubic2",
    #     #         initial_guess=solved_params
    #     #     )
            
    #     #     mean_err, max_err = model.get_error(q=q_calib, meas=tcp_calib, calib_params=solved_params)
        
    #     #     print(solved_params)
            
    #     #     print(f"mean err (uncalib) = {mean_err_uncalib*1000:.3f} mm")
    #     #     print(f"max err (uncalib) = {max_err_uncalib*1000:.3f} mm")
    #     #     print(f"mean err (calib) = {mean_err*1000:.3f} mm")
    #     #     print(f"max err (calib) = {max_err*1000:.3f} mm")
            
    #     mean_err, max_err = model.get_error(q=q_valid_cor, meas=tcp_valid, calib_params=solved_params)

    #     print(f"mean err (valid) = {mean_err*1000:.3f} mm")
    #     print(f"max err (valid) = {max_err*1000:.3f} mm")   
        
        
    #     # solved_params = err_params_zero #debug
    
    #     # # Plotting the compliance curves   
    #     # tau_tr=np.max(abs(model.get_gravity_torque(q_calib)),axis=0) #max tau for each joint
    #     # figure1, axis = plt.subplots(3, 2)
        
    #     # plt.subplots_adjust(left=0.1,
    #     #                 bottom=0.1, 
    #     #                 right=0.9, 
    #     #                 top=0.9, 
    #     #                 wspace=0.3, 
    #     #                 hspace=0.5) 
        
    #     # model.plot_compliance(axis, calib_params=solved_params,tau_tr=tau_tr)
        
    #     # if use_fake_data:
    #     #     model_gt.plot_compliance(axis, calib_params=err_params_gt,tau_tr=tau_tr)

    
    #     # plt.show()
        
    #     # print("done")

    #         # =====================================================

    #     NUM_BETAS = 3
    #     betas_2 = np.linspace(0.088, 0.349, NUM_BETAS)
    #     betas_3 = np.linspace(0.088, 0.349, NUM_BETAS)
    #     betas_5 = np.linspace(0, 0, NUM_BETAS)

    #     POP_SIZE = 500
    #     MAX_ITER = 20
    #     W_INERTIA = 0.7
    #     C1, C2 = 1.5, 1.5
    #     MIN_VAL, MAX_VAL = -0.2, -0.001
        

    #     print("Running PSO Optimization...")
    #     try:
    #         best_param_solved = main_calibrate_pi(q_calib, tcp_calib, q_valid, tcp_valid, fk_cmd, solved_params, best_param_now)
    #     except Exception as e:
    #         print(f"Error in PI calibration: {e}")
    #         best_param_solved = None
            
    #     best_param_now=best_param_solved
    #     q_valid_cor = pi_model(q_valid, best_param_solved)
    #     q_calib_cor = pi_model(q_calib, best_param_solved)

    #     mean_err, max_err = model.get_error(q=q_valid_cor, meas=tcp_valid, calib_params=solved_params)
    #     error_valid=mean_err*1000
    #     print(f"mean error vaild = {error_valid:.3f} mm")
    #     err_diff = abs(prev_err - error_valid)
        
    #     print(f"Error difference = {err_diff:.3f} mm")
    #     if err_diff < TOL:
    #         print(f"Converged at iteration {iter + 1}, stopping optimization.")
    #         break
    #     prev_err=error_valid
    
    
    # if best_param_solved:
    #     print("Best PI Parameters:")
    #     print("  Axis 2 weights w2:", best_param_solved['w2'])
    #     print("  Axis 3 weights w3:", best_param_solved['w3'])
    #     print("  Axis 5 weights w5:", best_param_solved['w5'])
    # else:
    #     print("Best PI Parameters not found!")

    

    
    
    # if best_param_solved is not None:
    #     q_valid_corrected = pi_model(q_valid, best_param_solved)
    #     q_calib_corrected = pi_model(q_calib, best_param_solved)
    #     plot_q_correction_diff(q_valid, q_valid_corrected)
    

if __name__ == "__main__":
    main()
