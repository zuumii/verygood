
from typing import List
import casadi as cs
import numpy as np
from calibration.calib_models import CalibrationModel


class CalibrationSolverIpopt:


    def solve_calibration(
        self,
        model: CalibrationModel,
        q: np.ndarray,
        meas: np.ndarray,
        comp_model: str,
        initial_guess: np.ndarray,
    ) -> np.ndarray:
        err_switch = model.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])

        # ---------- IPOPT opti ----------
        opti = cs.Opti()
        x = opti.variable(num_ax * num_params_per_ax)

        # ---------- cost ----------
        cost = 0.0
        fwkin = model.get_symbolic_meas_fct()
        for qi, mi in zip(q, meas):
            pred = fwkin(qi, x)
            cost += cs.sumsqr(pred - mi)
        opti.minimize(cost)

        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == 0)


        if num_params_per_ax > 6:
            for i in range(6):  
                c0_idx = i * num_params_per_ax + 6  
                opti.subject_to(x[c0_idx] > 0.000000000001)

        # ---------- solve ----------
        opti.solver("ipopt")
        opti.set_initial(x, initial_guess)
        sol = opti.solve()
        return np.asarray(sol.value(x))
    
    
    
    def solve_double_calibration(
        self,
        model_high: CalibrationModel,
        model_low: CalibrationModel,
        q_high: np.ndarray,
        meas_high: np.ndarray,
        q_low: np.ndarray,
        meas_low: np.ndarray,
        comp_model: str,
        initial_guess: np.ndarray,
    ) -> np.ndarray:

        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        num_params_per_ax = len(err_switch[0])

        assert model_low.get_error_par_switch() == err_switch, "err_switch diff!"

        # ---------- IPOPT opti ----------
        opti = cs.Opti()
        x = opti.variable(num_ax * num_params_per_ax)

        # ---------- cost  ----------
        cost = 0.0
        fwkin_high = model_high.get_symbolic_meas_fct()
        fwkin_low = model_low.get_symbolic_meas_fct()

        for qi, mi in zip(q_high, meas_high):
            pred = fwkin_high(qi, x)
            cost += cs.sumsqr(pred - mi)

        for qi, mi in zip(q_low, meas_low):
            pred = fwkin_low(qi, x)
            cost += cs.sumsqr(pred - mi)

        opti.minimize(cost)


        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == 0)


        if num_params_per_ax > 6:
            for i in range(6):
                c0_idx = i * num_params_per_ax + 6
                opti.subject_to(x[c0_idx] > 0.000000000001)

        # ---------- solve ----------
        opti.solver("ipopt")
        opti.set_initial(x, initial_guess)
        sol = opti.solve()

        return np.asarray(sol.value(x))
    
    
    def solve_compliance_alignment(
        self,
        model_high: CalibrationModel,
        model_low:  CalibrationModel,
        q_shared:   np.ndarray,        # (N,6) 
        tau_high:   np.ndarray,        # (N,6)  
        tau_low:    np.ndarray,        # (N,6) 
        meas_high:  np.ndarray,        # (N,3)  
        meas_low:   np.ndarray,        # (N,3)
        comp_model: str,
        initial_guess: np.ndarray,
        lambda1: float = 1.0,           
        lambda2: float = 10.0           
    ) -> np.ndarray:

        err_switch = model_high.get_error_par_switch()
        num_ax = len(err_switch)
        geom_param = 6
        nC = {"Lin":1, "Quad":2, "Cubic":3}[comp_model]
        num_params_per_ax = geom_param + nC

        # ---------- CasADi ----------
        opti = cs.Opti()
        x = opti.variable(num_ax * num_params_per_ax)

        fk_high = model_high.get_symbolic_meas_fct()
        fk_low  = model_low .get_symbolic_meas_fct()

        cost = 0
        for q_i, τh, τl, tcp_m_h, tcp_m_l in zip(q_shared, tau_high, tau_low,
                                                meas_high, meas_low):
    
            dq_h, dq_l = [], []
            for j in range(6):
                base = j * num_params_per_ax + geom_param
                if   nC == 1:
                    C1 = x[base]
                    dq_h.append(C1 * τh[j]);           dq_l.append(C1 * τl[j])
                elif nC == 2:
                    C1, C2 = x[base], x[base+1]
                    dq_h.append(C1*τh[j] + C2*τh[j]**2)
                    dq_l.append(C1*τl[j] + C2*τl[j]**2)
                else:
                    C1,C2,C3 = x[base],x[base+1],x[base+2]
                    dq_h.append(C1*τh[j]+C2*τh[j]**2+C3*τh[j]**3)
                    dq_l.append(C1*τl[j]+C2*τl[j]**2+C3*τl[j]**3)

            dq_h = cs.vertcat(*dq_h)
            dq_l = cs.vertcat(*dq_l)

            q_corr_h = q_i + dq_h
            q_corr_l = q_i + dq_l

            tcp_pred_h = fk_high(q_corr_h, x)
            tcp_pred_l = fk_low (q_corr_l, x)

            delta_pred  = tcp_pred_h - tcp_pred_l         
            delta_meas  = tcp_m_h  - tcp_m_l              
            pred_dist2 = cs.sumsqr(tcp_pred_h - tcp_pred_l)  
            meas_dist2 = cs.sumsqr(tcp_m_h  - tcp_m_l)   
            
            cost += lambda1 * cs.sumsqr(delta_pred - delta_meas)   
            cost += lambda2 * cs.sumsqr(pred_dist2 - meas_dist2)   

        opti.minimize(cost)

        
        for j in range(num_ax):
           
            for g in range(geom_param):
                opti.subject_to(x[j*num_params_per_ax + g] == 0)
         
            for k in range(nC):
                local_idx = geom_param + k
                if not err_switch[j][local_idx]:
                    opti.subject_to(x[j*num_params_per_ax + local_idx] == 0)

        # C1 > 0
        for j in range(6):
            opti.subject_to(x[j*num_params_per_ax + geom_param] > 1e-12)

        # ---------- solve ----------
        
        # ---------------- IPOPT ----------------
        ipopt_opts = {
            "max_iter":        15000,     
            "tol":             1e-10,
            "acceptable_tol":  1e-7,
            "acceptable_iter": 0,
            "constr_viol_tol": 1e-10,
            "dual_inf_tol":    1e-10,
            "print_level":     5,
        }



        opti.solver("ipopt", {}, ipopt_opts)
        opti.set_initial(x, initial_guess)
        sol = opti.solve()
        return np.asarray(sol.value(x))