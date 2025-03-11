from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import List
import casadi as cs
import numpy as np
from calibration.calib_models import CalibrationModel


class CalibrationSolver(ABC):

    @abstractmethod
    def solve_calibration(
        self,
        model: CalibrationModel,
        q: np.ndarray,
        meas: np.ndarray,
    ) -> np.ndarray:
        pass


class CalibrationSolverIpopt(CalibrationSolver):
    

    def solve_calibration(
        self,
        model: CalibrationModel,
        q: List[List[float]],        # shape (N,3)
        meas: List[List[float]],     # shape (N,2)
        comp_model: str,
        initial_guess: List[float] = None,
        
        use_pi_model: bool = False,
        num_betas_for_pi: int = 3,
        initial_w2=None,
        initial_w3=None,
        initial_w5=None
    ) -> np.ndarray:
        
        
        q_np = np.array(q)        # (N,3)
        meas_np = np.array(meas)  # (N,2)
        N = q_np.shape[0]        

        
        opti = cs.Opti()

        
        
        q_data_param  = opti.parameter(N, 3)  # (N,3)
        meas_data_param = opti.parameter(N, 2) # (N,2)

      
        opti.set_value(q_data_param, q_np)
        opti.set_value(meas_data_param, meas_np)

       
        err_par_switch = model.get_error_par_switch()
        num_ax = len(err_par_switch)
        num_params_per_ax = len(err_par_switch[0])
        x_dim = num_ax * num_params_per_ax
        x = opti.variable(x_dim)

        if initial_guess is None:
            initial_guess = np.zeros(x_dim)
        opti.set_initial(x, initial_guess)

       
        if use_pi_model:
            w2 = opti.variable(num_betas_for_pi)
            w3 = opti.variable(num_betas_for_pi)
            w5 = opti.variable(num_betas_for_pi)
            if initial_w2 is None: 
                initial_w2 = np.zeros(num_betas_for_pi)
            if initial_w3 is None: 
                initial_w3 = np.zeros(num_betas_for_pi)
            if initial_w5 is None: 
                initial_w5 = np.zeros(num_betas_for_pi)
            opti.set_initial(w2, initial_w2)
            opti.set_initial(w3, initial_w3)
            opti.set_initial(w5, initial_w5)

            # ============ w2,w3,w5<= 0 ============
            opti.subject_to(w2 <= 0)
            opti.subject_to(w3 <= 0)
            opti.subject_to(w5 == 0)
            # ====================================================
        else:
            w2 = None
            w3 = None
            w5 = None

       
        cost_fct = 0.0
        fwkin_fct = model.get_symbolic_meas_fct()  # (q_single, x)=> (2×1)

        if use_pi_model:
            # =>  (N×3) => Q_corr
            Q_corr = model._symbolic_fct_qcorr(q_data_param, w2, w3, w5)  # shape (N,3)
            
            for i in range(N):
                q_i = Q_corr[i,:]  # shape(1,3)
                tcp_pred_i = fwkin_fct(q_i, x)  # (2×1)
                tcp_pred_i_row = tcp_pred_i.T   # => (1×2)
                diff_i = tcp_pred_i_row - meas_data_param[i,:]  # (1×2)
                cost_fct += cs.dot(diff_i, diff_i)
        else:
            
            for i in range(N):
                q_i = q_data_param[i,:]      # (1,3)
                tcp_pred_i = fwkin_fct(q_i, x)  # (2×1)
                tcp_pred_i_row = tcp_pred_i.T   # (1×2)
                diff_i = tcp_pred_i_row - meas_data_param[i,:]  # (1×2)
                cost_fct += cs.dot(diff_i, diff_i)

        opti.minimize(cost_fct)

       
       
        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_par_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == 0.0)

        ax = [1,2,4]
        tau=cs.SX.sym("tau")
        t=cs.SX.sym("t")
        C=cs.SX.sym("C",model.num_comp_param)

        #Setting the symbolic functions of the first and second derivative of the compliance model that will be used to define the constraints
        if comp_model == "Cubic2":
            der=cs.jacobian(model.comp_fct(tau,C,t),tau)  
            der_comp=cs.Function("der_comp",[tau,C,t],[der]) 
            con=cs.jacobian(der,tau)*cs.sign(tau)       
            concavity=cs.Function("concavity",[tau,C,t],[con])  
        else:
            der=cs.jacobian(model.comp_fct(tau,C),tau)  
            der_comp=cs.Function("der_comp",[tau,C],[der]) 
            con=cs.jacobian(der,tau)*cs.sign(tau)       
            concavity=cs.Function("concavity",[tau,C],[con])

       
        for i_data in range(N):
            

            taul_full = np.squeeze(model.get_gravity_torque(q_np[i_data]))  # => shape(3,)
            c=0
            for j in ax:
                subC = x[c*num_params_per_ax+3 : (c+1)*num_params_per_ax]
                if comp_model == "Cubic2":
                    t_val=model.tau_t[j]
                    opti.subject_to(der_comp(taul_full[j],subC,t_val)>=0)
                    if comp_model != "Lin":
                        opti.subject_to(concavity(taul_full[j],subC,t_val)<=0)
                    if j<3:
                        opti.subject_to(der_comp(taul_full[j],subC,t_val)<=6.45161e-05)
                    else:
                        opti.subject_to(der_comp(taul_full[j],subC,t_val)<=0.00005)
                else:
                    opti.subject_to(der_comp(taul_full[j],subC)>=0)
                    if comp_model != "Lin":
                        opti.subject_to(concavity(taul_full[j],subC)<=0)
                    if j<3:
                        opti.subject_to(der_comp(taul_full[j],subC)<=6.45161e-05)
                    else:
                        opti.subject_to(der_comp(taul_full[j],subC)<=0.00005)
                c+=1

        #Continuity constraints for the 2 spline model at the transition torque
        if comp_model == "Cubic2":
            c=0
            for j in ax:
                subC = x[c*num_params_per_ax+3 : (c+1)*num_params_per_ax]
                tau_t=model.tau_t[j]
                opti.subject_to(subC[0]*tau_t+1e-2*subC[1]*tau_t**2+1e-4*subC[2]*tau_t**3==subC[3]+subC[4]*tau_t+1e-2*subC[5]*tau_t**2+1e-4*subC[6]*tau_t**3)
                opti.subject_to(subC[0]+1e-2*2*subC[1]*tau_t+1e-4*3*subC[2]*tau_t**2==subC[4]+1e-2*2*subC[5]*tau_t+1e-4*3*subC[6]*tau_t**2)
                c=c+1

       
        opti.solver("ipopt")
        sol = opti.solve()

       
        if use_pi_model:
            # => (x, w2, w3, w5)
            return sol.value(x), sol.value(w2), sol.value(w3), sol.value(w5)
        else:
            return sol.value(x)
    
    #Method to solve the first step of the dual load calibration problem (Recovering only the compliance terms)
    def solve_calibration_dual(
        self,
        model_low: CalibrationModel,
        model_high: CalibrationModel,
        q: List[List[float]],
        meas_low: List[List[float]],
        meas_high: List[List[float]],
        comp_model: str,
        initial_guess=List[float]
    ) -> np.ndarray:
        
        err_par_switch = model_low.get_error_par_switch() #the error switch of both model (high/low load) are identical
        num_ax = len(err_par_switch)
        num_params_per_ax = len(err_par_switch[0])
        # set up opti
        opti = cs.Opti()
        # create variables according to err par switch
        x = opti.variable(num_ax * num_params_per_ax)
        # set up cost function
        cost_fct = 0.0
        for q_, tcp_l, tcp_h in zip(q, meas_low, meas_high):
            fwkin_fct_low = model_low.get_symbolic_meas_fct() #get the symbolic formulation for the end effector position with low load
            fwkin_fct_high = model_high.get_symbolic_meas_fct()  #get the symbolic formulation for the end effector position with high load
            tcp_pred = fwkin_fct_high(q_, x)-fwkin_fct_low(q_, x) 
            cost_fct += cs.dot((tcp_pred - np.array(tcp_h) +np.array(tcp_l)), (tcp_pred - np.array(tcp_h) +np.array(tcp_l))) #Comparing the symbolic deviation to the actual one in the cost function
            
            
        opti.minimize(cost_fct)
        # Add constraints according to error par switch
        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_par_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == 0.0)

        
        ax=[1,2,4]
        
        tau=cs.SX.sym("tau")
        t=cs.SX.sym("t")
        C=cs.SX.sym("C",model_high.num_comp_param)
        
        #Setting the symbolic functions of the first and second derivative of the compliance model for the high and low load options
        
        if comp_model == "Cubic2":
            der=cs.jacobian(model_high.comp_fct(tau,C,t),tau)  
            der_comp_high=cs.Function("der_comp",[tau,C,t],[der]) 
            con=cs.jacobian(der,tau)*cs.sign(tau)       
            concavity_high=cs.Function("concavity",[tau,C,t],[con])  
        else:
            der=cs.jacobian(model_high.comp_fct(tau,C),tau)  
            der_comp_high=cs.Function("der_comp",[tau,C],[der]) 
            con=cs.jacobian(der,tau)*cs.sign(tau)       
            concavity_high=cs.Function("concavity",[tau,C],[con])
            
        if comp_model == "Cubic2":
            der=cs.jacobian(model_low.comp_fct(tau,C,t),tau)  
            der_comp_low=cs.Function("der_comp",[tau,C,t],[der]) 
            con=cs.jacobian(der,tau)*cs.sign(tau)       
            concavity_low=cs.Function("concavity",[tau,C,t],[con])  
        else:
            der=cs.jacobian(model_low.comp_fct(tau,C),tau)  
            der_comp_low=cs.Function("der_comp",[tau,C],[der]) 
            con=cs.jacobian(der,tau)*cs.sign(tau)       
            concavity_low=cs.Function("concavity",[tau,C],[con])
        
        for l in range(2):  # The same constraint of the single load calibration are repeated twice, once for high and once for low load   
            for q_ in q:
                q_full=[0,q_[0],q_[1],0,q_[2],0]
                qd = cs.MX.zeros(6, 1)
                qdd = cs.MX.zeros(6, 1)
                if l== 1:
                    load_trq_fct = -model_low.load_fct(q_full, qd, qdd)[0:6, 0:3]
                    der_comp=der_comp_low
                    concavity=concavity_low
                else:
                    load_trq_fct = -model_high.load_fct(q_full, qd, qdd)[0:6, 0:3]
                    der_comp=der_comp_high
                    concavity=concavity_high
                taul_full = cs.MX.zeros(6, 1)
                taul_full[1] = load_trq_fct[1, 1]
                taul_full[2] = load_trq_fct[2, 1]
                taul_full[4] = load_trq_fct[4, 1]
                c=0
                for j in ax:                    
                    C=x[c*num_params_per_ax+3:(c+1)*num_params_per_ax]
                    if comp_model == "Cubic2":
                        t=model_high.tau_t[j]
                        opti._subject_to(der_comp(taul_full[j],C,t)>=0)
                        if comp_model != "Lin":
                            opti._subject_to(concavity(taul_full[j],C,t)<=0)
                        if j<3:
                            opti._subject_to(der_comp(taul_full[j],C,t)<=(6.45161e-05))
                        else:
                            opti._subject_to(der_comp(taul_full[j],C,t)<=0.00005)
                    else:
                        opti._subject_to(der_comp(taul_full[j],C)>=0)
                        if comp_model != "Lin":
                            opti._subject_to(concavity(taul_full[j],C)<=0)
                        if j<3:
                            opti._subject_to(der_comp(taul_full[j],C)<=(6.45161e-05))
                        else:
                            opti._subject_to(der_comp(taul_full[j],C)<=0.00005)   
                    c=c+1
                
        # Adding continuity constraints in case of 2 splines 
        if comp_model == "Cubic2":
            c=0
            for j in ax:
                C=x[c*num_params_per_ax+3:(c+1)*num_params_per_ax]
                tau_t=model_high.tau_t[j]
                opti._subject_to(C[0]*tau_t+1e-2*C[1]*tau_t**2+1e-4*C[2]*tau_t**3==C[3]+C[4]*tau_t+1e-2*C[5]*tau_t**2+1e-4*C[6]*tau_t**3)
                opti._subject_to(C[0]+1e-2*2*C[1]*tau_t+1e-4*3*C[2]*tau_t**2==C[4]+1e-2*2*C[5]*tau_t+1e-4*3*C[6]*tau_t**2)
                c=c+1
            
        # solve problem
        opti.solver("ipopt")

        opti.set_initial(x,initial_guess)
        sol = opti.solve()
        return sol.value(x)
    
    # Method to solve the second stage of the dual load calibration problem (solving only for the kinematic errors) 
    def solve_calibration_dual_2(
        self,
        model_low: CalibrationModel,
        model_high: CalibrationModel,
        q_high: List[List[float]],
        q_low: List[List[float]],
        meas_low: List[List[float]],
        meas_high: List[List[float]],
        initial_guess: List[float],
        use_high_load: bool
    ) -> np.ndarray:
        
        err_par_switch = model_low.get_error_par_switch()
        num_ax = len(err_par_switch)
        num_params_per_ax = len(err_par_switch[0])
        # set up opti
        opti = cs.Opti()
        # create variables according to err par switch
        x = opti.variable(num_ax * num_params_per_ax)
        # set up cost function
        cost_fct = 0.0
        
        for q_l, q_h, tcp_l, tcp_h in zip(q_low, q_high, meas_low, meas_high):
            fwkin_fct_low = model_low.get_symbolic_meas_fct()
            fwkin_fct_high = model_high.get_symbolic_meas_fct()
            # We can use both the datasets in the cost function (in this case we compare symbolic end effector position to the measured one instead of comparinf the deviation like in stage 1)   
            tcp_pred = fwkin_fct_low(q_l, x)
            cost_fct += cs.dot((tcp_pred - np.array(tcp_l)), (tcp_pred - np.array(tcp_l)))
            if use_high_load:   
                tcp_pred = fwkin_fct_high(q_h, x)
                cost_fct += cs.dot((tcp_pred - np.array(tcp_h)), (tcp_pred - np.array(tcp_h)))
            
            
        opti.minimize(cost_fct)
        # add constraints according to error par switch
        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_par_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == 0.0)
                if j>2: # All the compliance terms are forced to be equal to the output result of the first stage (inputed here as an initial guess)
                    opti.subject_to(x[i * num_params_per_ax + j] == initial_guess[i * num_params_per_ax + j])

        # No need to include constraints related to the compliance as we are not solving for it here
        opti.solver("ipopt")
        opti.set_initial(x,initial_guess)
        sol = opti.solve()
        return sol.value(x)
