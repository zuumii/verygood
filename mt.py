from abc import ABC, abstractmethod
from typing import List, Tuple
import casadi as cs
import numpy as np
import utils.utils as ut
import random
import matplotlib.pyplot as plt


class CalibrationModel(ABC):
    _symbolic_meas_fct: cs.Function
    _error_param_jacobian_fct: cs.Function
    _err_par_switch: List[List[bool]]

    def __init__(self) -> None:
        self._err_par_switch = []
        self._generate_symbolic_functions()

    @abstractmethod
    def _generate_symbolic_functions(self) -> None:
        pass

    @abstractmethod
    def generate_fake_data(
        self,
        q: List[List[float]],
        err_params: np.ndarray,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        tcp=np.zeros((np.shape(q)[0],2))
        for i in range(np.shape(q)[0]):
            tcp[i] = self._symbolic_meas_fct(q[i], err_params)
        return tcp 
    
    # The par_switch is a boolean matrix that shows which calibration parameters are we going to solve for and which will be forced to zero
    def set_error_par_switch(self, err_par_switch: List[List[bool]]) -> None:
        self._err_par_switch = err_par_switch

    def get_error_par_switch(self) -> List[List[bool]]:
        return self._err_par_switch

    def get_symbolic_meas_fct(self) -> cs.Function:
        return self._symbolic_meas_fct

    def get_error_param_jacobian_fct(self) -> cs.Function:
        return self._error_param_jacobian_fct


class CalibrationModelPlanar3RComplNl(CalibrationModel):
    kinvec: List[List[float]]
    rotation_axes: List[ut.RotationAxis]
    tcp: List[float]
    load_fct: cs.Function
    comp_fct: cs.Function
    tau_t: List[float]
    num_comp_param: int
    num_betas: int
    T_sequence: int
    
    def __init__(
        self,
        kinvec: List[List[float]],
        tcp: List[float],
        load_fct: cs.Function,
        comp_model: str,
        tau_tr=np.zeros(6),
        num_betas=3,
        T_sequence=250,
        betas_2=None,
        betas_3=None,
        betas_5=None
        ) -> None:
        self.kinvec = kinvec
        self.rotation_axes = [
            ut.RotationAxis.Z,
            ut.RotationAxis.Y,
            ut.RotationAxis.Y,
            ut.RotationAxis.X,
            ut.RotationAxis.Y,
            ut.RotationAxis.X,
        ]
        
        if betas_2 is None:
            betas_2 = [0.044, 0.109, 0.175]  
        if betas_3 is None:
            betas_3 = [0.044, 0.109, 0.175]
        if betas_5 is None:
            betas_5 = [0.00, 0.00, 0.00]
        self.betas_2 = cs.DM(betas_2)  # shape (num_betas,1) or (num_betas,)
        self.betas_3 = cs.DM(betas_3)
        self.betas_5 = cs.DM(betas_5)
        
        self.tcp = tcp
        self.load_fct = load_fct
        self.num_betas = num_betas  
        self.T_sequence = T_sequence 
        # Setting the compliance model as a casadi function depending on the degree needed (Lin/Quad/Cubic/2 cubic splines)
        tau=cs.SX.sym("tau")
        t=cs.SX.sym("t") #the torque where we transition between the 2 splines
        if comp_model == "Lin":
            C=cs.SX.sym("C")
            q=C*tau
            self.num_comp_param=1
            self.comp_fct=cs.Function("comp",[tau,C],[q])
        elif comp_model == "Quad":
            C=cs.SX.sym("C",2)
            q=C[0]*tau+1e-2*C[1]*cs.sign(tau)*(tau)**2
            self.num_comp_param=2
            self.comp_fct=cs.Function("comp",[tau,C],[q])
        elif comp_model == "Cubic":
            C=cs.SX.sym("C",3)
            q=C[0]*tau+1e-2*C[1]*cs.sign(tau)*(tau)**2+1e-4*C[2]*(tau)**3
            self.num_comp_param=3
            self.comp_fct=cs.Function("comp",[tau,C],[q])            
        elif comp_model == "Cubic2":
            C=cs.SX.sym("C",7)
            # The first spline is cenetered around the origin betwen -t and +t. The other 2 splines are for torques higher than t and are symetrical with respect to the origin
            q=cs.if_else(cs.fabs(tau)<t,C[0]*tau+1e-2*C[1]*cs.sign(tau)*(tau)**2+1e-4*C[2]*(tau)**3,C[3]*cs.sign(tau)+C[4]*tau+1e-2*C[5]*cs.sign(tau)*(tau)**2+1e-4*C[6]*(tau)**3)
            self.num_comp_param=7
            self.comp_fct=cs.Function("comp",[tau,C,t],[q])
            self.tau_t=tau_tr
        
        super().__init__()
        
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    def play_operator_sym(self, x_now, r_prev, beta):
       
        lower = x_now - beta
        upper = x_now + beta

       
        cond_in_range = ((r_prev >= lower) * (r_prev <= upper))

        return cs.if_else(
            cond_in_range,
            r_prev,
            cs.if_else(r_prev < lower, lower, upper)
        )
    
    def pi_model_sym(self, Q_cmd, w2, w3, w5):
        T = Q_cmd.size1() 
        q2_cmd = Q_cmd[:, 0]
        q3_cmd = Q_cmd[:, 1]
        q5_cmd = Q_cmd[:, 2]

        dq2 = self.compute_pi_correction_sym(q2_cmd, w2, self.betas_2)
        dq3 = self.compute_pi_correction_sym(q3_cmd, w3, self.betas_3)
        dq5 = self.compute_pi_correction_sym(q5_cmd, w5, self.betas_5)

        return cs.horzcat(q2_cmd + dq2, 
                        q3_cmd + dq3, 
                        q5_cmd + dq5)   
        
    
    def compute_pi_correction_sym(self, q_cmd_axis, w_axis, betas_axis):
        T = q_cmd_axis.size1()  
        K = w_axis.size1()     

      
        r_states_expr = [[None for _ in range(T)] for _ in range(K)]

       
        for k in range(K):
            val0 = self.play_operator_sym(q_cmd_axis[0], q_cmd_axis[0], betas_axis[k])
            r_states_expr[k][0] = val0 

       
        for t in range(1, T):
            for k in range(K):
                prev_val = r_states_expr[k][t-1]
                val_t = self.play_operator_sym(q_cmd_axis[t], prev_val, betas_axis[k])
                r_states_expr[k][t] = val_t  

        
        row_sx_list = []
        for k in range(K):
            
            row_sx_list.append(cs.vertcat(*r_states_expr[k]))  

       
        r_states_sx = cs.horzcat(*row_sx_list)  

      
        out_list = []
        for t_ in range(T):
            rs_col = r_states_sx[t_, :]            
            w_axis_col = cs.reshape(w_axis, (K,1)) 
            out_mat = cs.mtimes(rs_col, w_axis_col) 
            out_t = out_mat[0,0]                  
            out_list.append(out_t)

        out_symbol = cs.vertcat(*out_list)  # shape (T,1)

        # 5) delta(t) = out_symbol[t] - q_cmd_axis[t]* sum(w_axis)
        sum_w = cs.sum1(w_axis)
        delta = out_symbol - q_cmd_axis * sum_w
        return delta  # shape (T,)
    
    def pi_model_sym(self, Q_cmd, w2, w3, w5):
       
        T = Q_cmd.size1() 
        q2_cmd = Q_cmd[:, 0]
        q3_cmd = Q_cmd[:, 1]
        q5_cmd = Q_cmd[:, 2]

        dq2 = self.compute_pi_correction_sym(q2_cmd, w2, self.betas_2)
        dq3 = self.compute_pi_correction_sym(q3_cmd, w3, self.betas_3)
        dq5 = self.compute_pi_correction_sym(q5_cmd, w5, self.betas_5)

        return cs.horzcat(q2_cmd + dq2, 
                          q3_cmd + dq3, 
                          q5_cmd + dq5)
        
    # Creating a symbolic function for the forward kinematics in the general case along with a function for the jacobian (if needed)
    def _generate_symbolic_functions(self) -> None:
        q = cs.SX.sym("q", 3)
        taul = cs.SX.sym("taul", 3)
        err_params = cs.SX.sym("err_params", 12+self.num_comp_param*4)

        # treat as special case of 6R with joint compliance
        err_params_full = cs.SX.zeros(42+self.num_comp_param*7, 1)
        
        ax=[1,2,4,6] #axis considered in the planar case: 2, 3 5 and end effector
        param_considered=[0,2,4] #the kinematic parameters in the planar case are dly dlz and dq
        
        t_f=6+self.num_comp_param #6 kinematic parameters in the full 6D case
        t_p=3+self.num_comp_param #3 kinematic parameters in the planar case
        
        #Matching the planar case to the more general 6D case 
        c_i=0
        for i in ax:
            c_j=0
            for j in range(t_f):
                if j in param_considered or j>=6:
                    err_params_full[i*t_f+j]= err_params[c_i*t_p+c_j]
                    c_j=c_j+1
            c_i=c_i+1       
        
        # Augmenting q to be able to feed it into the load function that requires the angles of all the 6 joints             
        q_full = cs.SX.zeros(6, 1)
        q_full[0] = cs.pi / 2.0
        q_full[1] = q[0]
        q_full[2] = q[1]
        q_full[4] = q[2]

        # the first and second derivative of joint angles are set to zero as we calibrate in the static case
        qd = cs.SX.zeros(6, 1)
        qdd = cs.SX.zeros(6, 1)
        load_trq_fct = -self.load_fct(q_full, qd, qdd)[0:6, 0:3]

        # Only considering the tau on the actuated axis of each joint. The torques on the non planar joints are set to 0
        taul[0] = load_trq_fct[1, 1]
        taul[1] = load_trq_fct[2, 1]
        taul[2] = load_trq_fct[4, 1]

        taul_full = cs.SX.zeros(6, 1)
        taul_full[1] = taul[0]
        taul_full[2] = taul[1]
        taul_full[4] = taul[2]

        # Getting the 4x4 Tramsformation matrix in symbolic form before putting it in a Casadi function
        T_fk = self.get_fk(q_full, taul_full, err_params_full)
        
        # extract TCP yz pos 
        tcp_pos = T_fk[1:3, 3]
        self._symbolic_meas_fct = cs.Function(
            "tcp_pos",
            [q, err_params],
            [tcp_pos],
        )

        # Symbolic Jacobian matrix (not used in the code so far)
        jac_err = cs.jacobian(tcp_pos, err_params)
        self._error_param_jacobian_fct = cs.Function(
            "err_jac",
            [q, err_params],
            [jac_err],
        )
      
        T = self.T_sequence 
        Q_cmd = cs.SX.sym("Q_cmd", T, 3)  

        w2 = cs.SX.sym("w2", self.num_betas)
        w3 = cs.SX.sym("w3", self.num_betas)
        w5 = cs.SX.sym("w5", self.num_betas)

        Q_corr = self.pi_model_sym(Q_cmd, w2, w3, w5)
       
        self._symbolic_fct_qcorr = cs.Function(
            "q_corrected_sequence",
            [Q_cmd, w2, w3, w5],
            [Q_corr]
        )

    #Method to generate Sythetic data for simulations, The error params in this case are the synthetic ground truth parameters that the developer chose 
    def generate_fake_data(
        self,
        q: List[List[float]],
        err_params: np.ndarray,
    ) ->  List[List[float]]:
        num_points=np.shape(q)[0]
        measurements=np.zeros((num_points,2))
        for i in range(num_points):
            measurements[i]=np.squeeze(self._symbolic_meas_fct(q[i], err_params))
        return measurements

    # Method to combine all transformation matrix to get one forward kinematics matrix
    def get_fk(self, q: List[float], taul: List[float], err_params: np.ndarray) -> cs.DM:
        T_fk = cs.SX_eye(4)
        for k in range(6):
            ax_mat = self._get_joint_trafo(
                q=q[k], taul=taul[k], ax_no=k, err_params_joint=err_params[k * (6+self.num_comp_param) : (k + 1) * (6+self.num_comp_param)]
            )
            T_fk = cs.mtimes(T_fk, ax_mat)
        # add tool transformation
        tool_trans = ut.trans_mat(self.tcp)
        tool_err_trans = ut.trans_mat(err_params[-(6+self.num_comp_param):])
        tool_mat = cs.mtimes(tool_trans, tool_err_trans)
        T_fk = cs.mtimes(T_fk, tool_mat)
        return T_fk

    # Method to find the Transformation matrix for a joint
    def _get_joint_trafo(self, q: float, taul: float, ax_no: int, err_params_joint: np.ndarray):
        ax_trans = ut.trans_mat(self.kinvec[ax_no])
        ax_err_trans = ut.trans_mat(err_params_joint[0:3])
        
        # Computing the dq that is caused by compliance (in the case of 2 splines, the method needs an extra input, the transition torque)
        if self.num_comp_param == 7:
            q_compl = self.comp_fct(taul,err_params_joint[6:],self.tau_t[ax_no])
        else:
            q_compl = self.comp_fct(taul,err_params_joint[6:])
            
        if self.rotation_axes[ax_no] == ut.RotationAxis.X:
            ax_err_rot1 = ut.rot_mat_y(err_params_joint[4])
            ax_err_rot2 = ut.rot_mat_z(err_params_joint[5])
            ax_err_rot3 = ut.rot_mat_x(err_params_joint[3] + q + q_compl)
        elif self.rotation_axes[ax_no] == ut.RotationAxis.Y:
            ax_err_rot1 = ut.rot_mat_x(err_params_joint[3])
            ax_err_rot2 = ut.rot_mat_z(err_params_joint[5])
            ax_err_rot3 = ut.rot_mat_y(err_params_joint[4] + q + q_compl)
        elif self.rotation_axes[ax_no] == ut.RotationAxis.Z:
            ax_err_rot1 = ut.rot_mat_x(err_params_joint[3])
            ax_err_rot2 = ut.rot_mat_y(err_params_joint[4])
            ax_err_rot3 = ut.rot_mat_z(err_params_joint[5] + q + q_compl)
        else:
            raise ValueError(f"Unsupported RotationAxis {self.rotation_axes[ax_no]}")
        
        ax_rot = cs.mtimes(ax_err_rot1, cs.mtimes(ax_err_rot2, ax_err_rot3))
        ax_mat = cs.mtimes(ax_trans, cs.mtimes(ax_err_trans, ax_rot))
        
        return ax_mat

    # Method to find the mean and maximum error based on the found calibration parameters
    def get_error(
        self,
        q: List[List[float]],
        meas: List[List[float]],
        calib_params: np.ndarray,
    ) -> Tuple[float, float]:
        errors = []
        load_trq_23 = []
        errors_z = []
        for q_, tcp_ in zip(q, meas):
            load_trq = np.squeeze(self.get_gravity_torque(q_))
            load_trq_23.append([load_trq[1], load_trq[2]])
            tcp_pred = self._symbolic_meas_fct(q_, calib_params)
            errors.append(cs.norm_2(tcp_pred - np.array(tcp_)).full())
            errors_z.append(tcp_pred.full()[-1] - tcp_[-1])
            
        # Uncomment for plotting the z error and norm error as a function of the load torques
        """ load_trq_23 = np.array(load_trq_23)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(load_trq_23[:, 0], load_trq_23[:, 1], 1000 * np.array(errors))
        ax.set_xlabel("load trq ax2 in Nm")
        ax.set_ylabel("load trq ax3 in Nm")
        ax.set_zlabel("error norm in mm")

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection="3d")
        ax2.scatter(load_trq_23[:, 0], load_trq_23[:, 1], 1000 * np.array(errors_z))
        ax2.set_xlabel("load trq ax2 in Nm")
        ax2.set_ylabel("load trq ax3 in Nm")
        ax2.set_zlabel("z error in mm")

        fig3, axs = plt.subplots(2, 1, sharex=True)
        axs[0].scatter(load_trq_23[:, 0], 1000 * np.array(errors_z))
        axs[0].set_xlabel("load trq ax2 in Nm")
        axs[0].set_ylabel("z error in mm")
        axs[0].grid(True)
        axs[1].scatter(load_trq_23[:, 1], 1000 * np.array(errors_z))
        axs[1].set_xlabel("load trq ax3 in Nm")
        axs[1].set_ylabel("z error in mm")
        axs[1].grid(True) """
        # plt.show()
        return np.mean(errors), np.max(errors)
    
    # Method to plot the compliance curves, tau_tr is the maximum torque that we can get on each joint in the dataset
    def plot_compliance(self, axis, calib_params: np.ndarray, tau_tr: np.ndarray):
        
        ax=[1,2,4] #The indices of the planar joints                  
        num_pts=1000 #The number of datapoints per plot can be increased/decreased based on the desired smoothness the plot 
        
        test_torques=np.zeros((6,num_pts))

        for i in range(6):
            test_torques[i]=np.linspace(-tau_tr[i],tau_tr[i],num_pts)
            
        test_torques=np.transpose(test_torques)

        q_est=np.zeros((num_pts,6))
        
        for i in range(num_pts):
            c=0
            for j in ax:
                C_est=calib_params[c*(3+self.num_comp_param)+3:(c+1)*(3+self.num_comp_param)] #Isolating the compliance parameters from the set of all calibration parameters
                
                if self.num_comp_param == 7: # in case we have 2 splines, the method requires the tau at which we transition from one to the other
                    q_est[i][j]=self.comp_fct(test_torques[i][j],C_est,self.tau_t[j])
                else:
                    q_est[i][j]=self.comp_fct(test_torques[i][j],C_est)
                c=c+1
            
        q_est=np.transpose(q_est)
        test_torques=np.transpose(test_torques)

        for i in range(6):
            axis[i%3,int(i/3)].plot(test_torques[i],q_est[i])
            
            axis[i%3,int(i/3)].set_title("Estimated compliance curves for joint " + str(i+1))
            
            axis[i%3,int(i/3)].set_xlabel("tau in Nm")
            
            axis[i%3,int(i/3)].set_ylabel("dq in rad")
     
    #Method to get the gravity torques on the actuated axis of 6 joints (for the non planar one joints, joint 1/4/6, the value is forced to zero)        
    def get_gravity_torque(self,q):
        if (np.size(q))==3:
            q=np.expand_dims(q, axis=0)
            
        taul_full=np.zeros((np.shape(q)[0],6))
        i=0        
        for q_ in (q):
            q_full=[0,q_[0],q_[1],0,q_[2],0]
            qd = np.zeros((6, 1))
            qdd = np.zeros((6, 1))
            load_trq_fct = -self.load_fct(q_full, qd, qdd)[0:6, 0:3]

            taul_full[i][1] = load_trq_fct[1, 1]
            taul_full[i][2] = load_trq_fct[2, 1]
            taul_full[i][4] = load_trq_fct[4, 1]
            i=i+1
        
        return taul_full