
import random


    
    


    # =====================================================

    NUM_BETAS = 10
    betas_2 = np.linspace(0, 0.78, NUM_BETAS)
    betas_3 = np.linspace(0, 0.78, NUM_BETAS)
    betas_5 = np.linspace(0, 0, NUM_BETAS)

    POP_SIZE = 100
    MAX_ITER = 50
    W_INERTIA = 0.7
    C1, C2 = 1.5, 1.5
    MIN_VAL, MAX_VAL = -0.5, 0.0

    

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
            r_states[k, 0] = play_operator(q_cmd_axis[0], 0.0, betas_axis[k])

        
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

        dq2 = compute_pi_correction(q2_cmd, param['w2'], betas_2)
        dq3 = compute_pi_correction(q3_cmd, param['w3'], betas_3)
        dq5 = compute_pi_correction(q5_cmd, param['w5'], betas_5)

        # q_corrected = q_cmd + Δq
        return np.column_stack([q2_cmd + dq2, 
                                q3_cmd + dq3, 
                                q5_cmd + dq5])

 
    def objective_function(param, q_cmd_data, tcp_data, fk_cmd, solved_params):
       
        q_cmd_data = np.array(q_cmd_data)
        tcp_data = np.array(tcp_data)

        q_corrected = pi_model(q_cmd_data, param)
        tcp_model = []

        for q_i in q_corrected:
            tcp_model.append(fk_cmd(q_i, solved_params))

        tcp_model = np.array(tcp_model).squeeze() 

        diff = tcp_data - tcp_model
        sse = np.sum(np.sum(diff ** 2, axis=1))
        return sse

    

    def pso_optimize(q_calib, tcp_calib, fk_cmd, solved_params):
        
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

        print("PSO iteration process:")
        for it in range(MAX_ITER):
           
            for i in range(POP_SIZE):
                param_i = vec_to_param(pop[i])
                score_i = objective_function(param_i, q_calib, tcp_calib, fk_cmd, solved_params)

                
                if score_i < pbest_scores[i]:
                    pbest_scores[i] = score_i
                    pbest[i] = pop[i].copy()

                
                if score_i < gbest_score:
                    gbest_score = score_i
                    gbest = pop[i].copy()

            
            for i in range(POP_SIZE):
                r1, r2 = random.random(), random.random()
                vel[i] = (W_INERTIA * vel[i]
                        + C1 * r1 * (pbest[i] - pop[i])
                        + C2 * r2 * (gbest - pop[i]))
                pop[i] += vel[i]
                
                
                pop[i] = np.clip(pop[i], MIN_VAL, MAX_VAL)

            
            if (it + 1) % 5 == 0:
                print(f"  Iteration {it + 1}/{MAX_ITER}, Current Best Error: {gbest_score:.6f}")

        return vec_to_param(gbest), gbest_score

    # =====================================================
    #   5. 主函数: 标定 + 验证
    # =====================================================

    def main_calibrate_pi(q_calib, tcp_calib, q_valid, tcp_valid, fk_cmd, solved_params):
        
        q_calib = np.array(q_calib)
        tcp_calib = np.array(tcp_calib)
        q_valid = np.array(q_valid)
        tcp_valid = np.array(tcp_valid)

        best_param, best_score = pso_optimize(q_calib, tcp_calib, fk_cmd, solved_params)
        valid_error = objective_function(best_param, q_valid, tcp_valid, fk_cmd, solved_params)

       
        q_calib_corrected = pi_model(q_calib, best_param)
        tcp_model_calib = np.array([fk_cmd(q_i, solved_params) for q_i in q_calib_corrected]).squeeze()
        err_calib = np.linalg.norm(tcp_calib - tcp_model_calib, axis=1)
        mean_err_calib = np.mean(err_calib)
        max_err_calib = np.max(err_calib)


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
        print(f"Error in PI calibration: {e}")
        best_param_solved = None

    if best_param_solved:
        print("Best PI Parameters:")
        print("  Axis 2 weights w2:", best_param_solved['w2'])
        print("  Axis 3 weights w3:", best_param_solved['w3'])
        print("  Axis 5 weights w5:", best_param_solved['w5'])
    else:
        print("Best PI Parameters not found!")

    

    def plot_q_correction_diff(q_cmd, q_corrected):
        
        q_cmd = np.array(q_cmd)
        q_corrected = np.array(q_corrected)
        t = np.arange(len(q_cmd))
        q_diff = q_corrected - q_cmd 

        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        joint_labels = ["Joint 2", "Joint 3", "Joint 5"]

        for i in range(3):
            
            axs[i, 0].plot(t, q_corrected[:, i], color='blue', alpha=0.7, linewidth=2, label='q_corrected')  
            axs[i, 0].plot(t, q_cmd[:, i], color='red', linestyle='dotted', linewidth=2.5, zorder=3, label='q_cmd') 
            axs[i, 0].scatter(t[::5], q_cmd[::5, i], color='red', marker='o', s=10, alpha=0.8, label='q_cmd points')  

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

    
    if best_param_solved is not None:
        q_valid_corrected = pi_model(q_valid, best_param_solved)
        q_calib_corrected = pi_model(q_calib, best_param_solved)
        plot_q_correction_diff(q_valid, q_valid_corrected)
    
