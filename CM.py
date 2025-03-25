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

    def __init__(
        self,
        kinvec: List[List[float]],
        tcp: List[float],
        load_fct: cs.Function,
        comp_model: str,
        tau_tr=np.zeros(6)
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
        self.tcp = tcp
        self.load_fct = load_fct
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
    
    def compute_planar_3r_link_lengths(self) -> np.ndarray:
        """
        计算“平面3R”三段连杆长度 [L1, L2, L3]，
        其中 L1=dist(关节2, 关节3), L2=dist(关节3, 关节5), L3=dist(关节5, TCP)。

        注意：
        - 这里对关节1/4/6的角度设为0，关节2/3/5也设为0，且不考虑任何err_params和扭矩(taul)。
        - 因此得到的是最理想情况下的标称长度。
        """

        import casadi as cs
        import numpy as np

        # (1) 先准备一个全零的 err_params
        # 在代码里，每个关节有 (6 + self.num_comp_param) 个参数，
        # 整个 6 关节就是 6*(6+self.num_comp_param)
        total_err_dim = 6 * (6 + self.num_comp_param)
        err_params_zero = cs.SX.zeros(total_err_dim, 1)

        # (2) 关节角度全部置0:
        #    [joint1, joint2, joint3, joint4, joint5, joint6]
        #    在“Planar3R”概念里只动2/3/5，这里也都设它们=0
        q_full = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # (3) 不考虑扭矩 => taul=0
        taul_full = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # (4) 循环计算各关节的变换矩阵，并存储每个关节末端坐标
        T_k = cs.SX.eye(4)
        joint_positions = []  # 用来保存每个关节末端在基坐标系下的位置 p_k

        for k in range(6):
            # 取每个关节对应的参数片段
            start_idx = k*(6+self.num_comp_param)
            end_idx   = (k+1)*(6+self.num_comp_param)
            err_params_joint = err_params_zero[start_idx:end_idx]

            # ================= 构造单个关节的 4x4 变换矩阵 =================
            # 1) 固定平移
            ax_trans = ut.trans_mat(self.kinvec[k])
            # 2) 误差平移(此处为0)
            ax_err_trans = ut.trans_mat(err_params_joint[0:3])

            # 3) 计算关节柔性修正(扭矩=0 => dq=0)
            if self.num_comp_param == 7:
                q_compl = self.comp_fct(taul_full[k], err_params_joint[6:], self.tau_t[k])
            else:
                q_compl = self.comp_fct(taul_full[k], err_params_joint[6:])

            # 4) 关节旋转(实际 = err_params_joint[3..5] + q + q_compl)，但这里都=0
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

            # 5) 更新全局变换
            T_k = cs.mtimes(T_k, ax_mat)

            # 保存此关节末端的 3D 位置
            p_k = T_k[0:3, 3]
            joint_positions.append(p_k)

        # (5) 最后再乘上末端工具 => 得到 TCP 位置
        tool_err_dim = 6 + self.num_comp_param  # 最后那一段
        tool_err = err_params_zero[-tool_err_dim:]  # 这里也全0

        tool_trans = ut.trans_mat(self.tcp)
        tool_err_trans = ut.trans_mat(tool_err)
        tool_mat = cs.mtimes(tool_trans, tool_err_trans)

        T_tcp = cs.mtimes(T_k, tool_mat)
        p_tcp = T_tcp[0:3, 3]

        # joint_positions[k] => 第k关节末端的位置
        # k=1 => 关节2,  k=2 => 关节3,  k=4 => 关节5
        p2 = joint_positions[1]
        p3 = joint_positions[2]
        p5 = joint_positions[4]

        # (6) 分别计算三段距离
        L1_sx = cs.norm_2(p3 - p2)      # 关节2 -> 关节3
        L2_sx = cs.norm_2(p5 - p3)      # 关节3 -> 关节5
        L3_sx = cs.norm_2(p_tcp - p5)   # 关节5 -> TCP

        # 由于 L1_sx / L2_sx / L3_sx 是 SX 符号，需要转成 float
        L1_val = float(cs.evalf(L1_sx))
        L2_val = float(cs.evalf(L2_sx))
        L3_val = float(cs.evalf(L3_sx))

        # 返回 numpy 数组
        return np.array([L1_val, L2_val, L3_val])
    
    
    
    
    def compute_planar_3r_link_lengths_and_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算“平面3R”情况下:
        1) 三段连杆长度: [L1, L2, L3]
            其中 L1=dist(joint2, joint3), L2=dist(joint3, joint5), L3=dist(joint5, TCP)。
        2) 两个相邻连杆之间的夹角: [alpha12, alpha23] (单位:弧度),
            - alpha12 = v1相对v2的有向夹角 (关节3处的夹角),
            - alpha23 = v2相对v3的有向夹角 (关节5处的夹角).
            * 若 >0 代表逆时针, <0 代表顺时针.

        注意:
        - 默认将关节1~6都设为0°, 不考虑任何误差/补偿/扭矩. 
        - 在 'Planar3R' 模型中, 
            joint2 => k=1, 
            joint3 => k=2, 
            joint5 => k=4.
        - 如果此时机械臂确实运动在 yz 平面, 那么计算夹角时我们仅取 (y,z) 分量.
        """
        import casadi as cs
        import numpy as np

        # ---------- 1) 准备全零 err_params 和关节角、扭矩 -----------
        total_err_dim = 6 * (6 + self.num_comp_param)
        err_params_zero = cs.SX.zeros(total_err_dim, 1)

        q_full = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        taul_full = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ---------- 2) 依次计算每个关节末端的变换矩阵, 并保存位置 ----------
        T_k = cs.SX.eye(4)
        joint_positions = []
        for k in range(6):
            start_idx = k*(6+self.num_comp_param)
            end_idx   = (k+1)*(6+self.num_comp_param)
            err_params_joint = err_params_zero[start_idx:end_idx]

            # 平移
            ax_trans = ut.trans_mat(self.kinvec[k])
            # 误差平移(=0)
            ax_err_trans = ut.trans_mat(err_params_joint[0:3])

            # 关节柔性修正(=0)
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
            p_k = T_k[0:3, 3]
            joint_positions.append(p_k)

        # 再乘工具末端
        tool_err_dim = 6 + self.num_comp_param
        tool_err = err_params_zero[-tool_err_dim:]
        tool_trans = ut.trans_mat(self.tcp)
        tool_err_trans = ut.trans_mat(tool_err)
        tool_mat = cs.mtimes(tool_trans, tool_err_trans)

        T_tcp = cs.mtimes(T_k, tool_mat)
        p_tcp = T_tcp[0:3, 3]

        # ---------- 3) 提取关节2,3,5 以及 TCP 的位置 ----------
        p2 = joint_positions[1]  # joint2
        p3 = joint_positions[2]  # joint3
        p5 = joint_positions[4]  # joint5

        # 三段向量 v1,v2,v3
        v1 = p3 - p2
        v2 = p5 - p3
        v3 = p_tcp - p5

        # ---------- 4) 计算长度 ----------
        import math
        L1_sx = cs.norm_2(v1)
        L2_sx = cs.norm_2(v2)
        L3_sx = cs.norm_2(v3)

        L1 = float(cs.evalf(L1_sx))
        L2 = float(cs.evalf(L2_sx))
        L3 = float(cs.evalf(L3_sx))
        lengths = np.array([L1, L2, L3], dtype=float)

        # ---------- 5) 在 yz 平面上 计算夹角(逆时针为正) ----------
        #    定义一个 2D 叉乘 / 点乘 的函数:
        def angle_between_2d(a: np.ndarray, b: np.ndarray) -> float:
            """
            返回向量a到向量b的有向夹角(弧度),
            a->b逆时针>0, 顺时针<0, 范围(-pi, pi).
            """
            cross = a[0]*b[1] - a[1]*b[0]  # 2D cross
            dot   = a[0]*b[0] + a[1]*b[1]
            return math.atan2(cross, dot)

        # 把 CasADi SX 转成数值, 并只取 yz 分量
        v1_np = np.array([float(cs.evalf(v1[1])), float(cs.evalf(v1[2]))])
        v2_np = np.array([float(cs.evalf(v2[1])), float(cs.evalf(v2[2]))])
        v3_np = np.array([float(cs.evalf(v3[1])), float(cs.evalf(v3[2]))])

        alpha12 = angle_between_2d(v1_np, v2_np)
        alpha23 = angle_between_2d(v2_np, v3_np)

        angles = np.array([alpha12, alpha23], dtype=float)

        return lengths, angles
    
    
    
    
    # def compute_4links_3angles_zero_pose(self):
    #     """
    #     计算:
    #     1) 四段长度 [L0, L1, L2, L3]:
    #         - L0:  基座(0,0,0) → 关节2
    #         - L1:  关节2 → 关节3
    #         - L2:  关节3 → 关节5
    #         - L3:  关节5 → TCP (工具尖端)

    #     2) 三个关节处的夹角 [alpha2, alpha3, alpha5], 单位: 弧度
    #         - alpha2: 在关节2处 (v0, v1) 的有向夹角
    #         - alpha3: 在关节3处 (v1, v2) 的有向夹角
    #         - alpha5: 在关节5处 (v2, v3) 的有向夹角
    #         (逆时针>0, 顺时针<0)

    #     同时打印并返回关节2、3、5以及 TCP 的 (x, y, z) 坐标，方便排查角度问题。

    #     注意：
    #     - 所有关节(1~6)都设 0°
    #     - 不考虑任何误差/扭矩
    #     - 仅在“理想刚体”下计算
    #     """

    #     import casadi as cs
    #     import numpy as np
    #     import math

    #     # ====== 1) 全零 err_params, 关节角, 扭矩 ======
    #     total_err_dim = 6 * (6 + self.num_comp_param)
    #     err_params_zero = cs.SX.zeros(total_err_dim, 1)

    #     q_full = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     taul_full = [0.0]*6

    #     # ====== 2) 计算每个关节末端的位姿 ======
    #     T_k = cs.SX_eye(4)
    #     joint_positions = []  # 存储关节k的坐标

    #     for k in range(6):
    #         start_idx = k*(6+self.num_comp_param)
    #         end_idx   = (k+1)*(6+self.num_comp_param)
    #         err_params_joint = err_params_zero[start_idx:end_idx]

    #         # 固定平移
    #         ax_trans = ut.trans_mat(self.kinvec[k])
    #         # 误差平移(=0)
    #         ax_err_trans = ut.trans_mat(err_params_joint[0:3])

    #         # 柔性修正(=0)
    #         if self.num_comp_param == 7:
    #             q_compl = self.comp_fct(taul_full[k], err_params_joint[6:], self.tau_t[k])
    #         else:
    #             q_compl = self.comp_fct(taul_full[k], err_params_joint[6:])

    #         # 旋转(=0)
    #         if self.rotation_axes[k] == ut.RotationAxis.X:
    #             ax_err_rot1 = ut.rot_mat_y(0)
    #             ax_err_rot2 = ut.rot_mat_z(0)
    #             ax_err_rot3 = ut.rot_mat_x(0)
    #         elif self.rotation_axes[k] == ut.RotationAxis.Y:
    #             ax_err_rot1 = ut.rot_mat_x(0)
    #             ax_err_rot2 = ut.rot_mat_z(0)
    #             ax_err_rot3 = ut.rot_mat_y(0)
    #         else:  # Z
    #             ax_err_rot1 = ut.rot_mat_x(0)
    #             ax_err_rot2 = ut.rot_mat_y(0)
    #             ax_err_rot3 = ut.rot_mat_z(0)

    #         ax_rot = cs.mtimes(ax_err_rot1, cs.mtimes(ax_err_rot2, ax_err_rot3))
    #         ax_mat = cs.mtimes(ax_trans, cs.mtimes(ax_err_trans, ax_rot))

    #         T_k = cs.mtimes(T_k, ax_mat)
    #         p_k = T_k[0:3, 3]  # 关节 k 的坐标
    #         joint_positions.append(p_k)

    #     # ====== 3) 工具末端 ======
    #     tool_err_dim = 6 + self.num_comp_param
    #     tool_err = err_params_zero[-tool_err_dim:]
    #     tool_trans = ut.trans_mat(self.tcp)
    #     tool_err_trans = ut.trans_mat(tool_err)
    #     tool_mat = cs.mtimes(tool_trans, tool_err_trans)

    #     T_tcp = cs.mtimes(T_k, tool_mat)
    #     p_tcp = T_tcp[0:3, 3]  # TCP坐标

    #     # ====== 4) 取出关节2,3,5 和基座、TCP ======
    #     # 在你的索引:
    #     #   joint2 => p2 = joint_positions[1]
    #     #   joint3 => p3 = joint_positions[2]
    #     #   joint5 => p5 = joint_positions[4]
    #     # 基座 => p0 = [0,0,0]
    #     p0 = cs.SX([0.0, 0.0, 0.0])
    #     p2 = joint_positions[1]
    #     p3 = joint_positions[2]
    #     p5 = joint_positions[4]

    #     # ====== 5) 四段向量 v0,v1,v2,v3 ======
    #     v0 = p2 - p0             # 基座→关节2
    #     v1 = p3 - p2             # 关节2→关节3
    #     v2 = p5 - p3             # 关节3→关节5
    #     v3 = p_tcp - p5          # 关节5→TCP

    #     # ====== 6) 计算四段长度 ======
    #     L0 = float(cs.norm_2(v0))
    #     L1 = float(cs.norm_2(v1))
    #     L2 = float(cs.norm_2(v2))
    #     L3 = float(cs.norm_2(v3))
    #     lengths = np.array([L0, L1, L2, L3], dtype=float)

    #     # ====== 7) 计算三个关节处的夹角(逆时针为正) ======
    #     def angle_between_2d(a: np.ndarray, b: np.ndarray) -> float:
    #         """
    #         返回 a->b 的有向夹角(弧度),
    #         若 >0 => 逆时针, <0 => 顺时针.
    #         """
    #         cross = a[0]*b[1] - a[1]*b[0]  # 2D "叉乘"
    #         dot   = a[0]*b[0] + a[1]*b[1]  # 2D 点乘
    #         return math.atan2(cross, dot)

    #     # 只取 (y,z) 分量
    #     import math
    #     v0_yz = np.array([float(v0[1]),  float(v0[2])])
    #     v1_yz = np.array([float(v1[1]),  float(v1[2])])
    #     v2_yz = np.array([float(v2[1]),  float(v2[2])])
    #     v3_yz = np.array([float(v3[1]),  float(v3[2])])

    #     alpha2 = angle_between_2d(v0_yz, v1_yz)  # 关节2处
    #     alpha3 = angle_between_2d(v1_yz, v2_yz)  # 关节3处
    #     alpha5 = angle_between_2d(v2_yz, v3_yz)  # 关节5处
    #     angles = np.array([alpha2, alpha3, alpha5], dtype=float)

    #     # ====== 8) 打印各点坐标(供调试) ======
    #     def to_np3(p_sx):
    #         return np.array([float(p_sx[0]), float(p_sx[1]), float(p_sx[2])])

    #     p2_np  = to_np3(p2)
    #     p3_np  = to_np3(p3)
    #     p5_np  = to_np3(p5)
    #     tcp_np = to_np3(p_tcp)

    #     print("\n--- Zero Pose Joint Positions (x,y,z) ---")
    #     print("p2  =", p2_np)
    #     print("p3  =", p3_np)
    #     print("p5  =", p5_np)
    #     print("TCP =", tcp_np)

    #     print("\n--- Link Lengths (m) ---")
    #     print(f"L0={L0:.5f}, L1={L1:.5f}, L2={L2:.5f}, L3={L3:.5f}")

    #     print("\n--- Joint Angles (radians) [alpha2, alpha3, alpha5] ---")
    #     print(angles)
    #     print("Joint Angles (deg):", angles * 180.0 / math.pi)

    #     # ====== 9) 返回结果 ======
    #     return lengths, angles, p2_np, p3_np, p5_np, tcp_np
    
    
    
    def compute_planar_4links_3angles_zero_pose(self):
        """
        在“零姿态”下，仅考虑 yz 平面，计算:
        1) 四段长度 [L0, L1, L2, L3]:
            - L0: 基座(0,0) → 关节2(y,z)
            - L1: 关节2 → 关节3
            - L2: 关节3 → 关节5
            - L3: 关节5 → TCP
        2) 三个关节处的夹角 [alpha2, alpha3, alpha5], 单位: 弧度
            (逆时针>0, 顺时针<0)
        3) 打印并返回关节2、3、5和 TCP 的 2D 坐标 (y,z)，用于调试。

        其中与 _generate_symbolic_functions 一样:
        - 关节1 (ax_no=0) 设 π/2
        - 关节2,3,4,5,6 全=0
        - 不考虑任何误差/扭矩 => err_params=0, taul=0
        - 只看 yz 平面
        """

        import casadi as cs
        import numpy as np
        import math

        # 1) 全零的 err_params
        total_err_dim = 6 * (6 + self.num_comp_param)
        err_params_zero = cs.SX.zeros(total_err_dim, 1)

        # 2) 定义关节角(与 _generate_symbolic_functions 保持一致)
        #    关节1=π/2, 关节2/3/4/5/6=0
        q_full = [
            cs.pi/2,  # joint1
            0.0,      # joint2
            0.0,      # joint3
            0.0,      # joint4
            0.0,      # joint5
            0.0       # joint6
        ]

        # 全部扭矩为 0
        taul_full = [0.0]*6

        # 3) 循环计算每个关节的 4x4 变换矩阵, 并在关节2/3/5时保存坐标
        T_k = cs.SX_eye(4)
        p2_3d = None
        p3_3d = None
        p5_3d = None

        for k in range(6):
            # 取本关节对应的 err_params 片段
            start_idx = k*(6 + self.num_comp_param)
            end_idx   = (k+1)*(6 + self.num_comp_param)
            err_params_joint = err_params_zero[start_idx:end_idx]

            # 用 _get_joint_trafo(...) 求单个关节的变换
            T_joint = self._get_joint_trafo(
                q=q_full[k],
                taul=0,  # 不考虑重力扭矩
                ax_no=k,
                err_params_joint=err_params_joint
            )

            # 累乘到全局 T_k
            T_k = cs.mtimes(T_k, T_joint)

            # 若 k=1 => 关节2, k=2 => 关节3, k=4 => 关节5
            if k in [1,2,4]:
                p_k = T_k[0:3, 3]  # 取 (x,y,z)
                if k==1: p2_3d = p_k
                if k==2: p3_3d = p_k
                if k==4: p5_3d = p_k

        # 4) 再乘上 tool 矩阵 => 得到 TCP
        tool_err_dim = 6 + self.num_comp_param
        tool_err = err_params_zero[-tool_err_dim:]
        tool_trans = ut.trans_mat(self.tcp)
        tool_err_trans = ut.trans_mat(tool_err)
        tool_mat = cs.mtimes(tool_trans, tool_err_trans)

        T_tcp = cs.mtimes(T_k, tool_mat)
        p_tcp_3d = T_tcp[0:3, 3]

        # 5) 将 (关节2,3,5, TCP) 的 (x,y,z) 投影到 (y,z)
        def to_np_2d(p_sx):
            # 只取 (y,z)
            return np.array([float(p_sx[1]), float(p_sx[2])])

        # 基座当作 (0,0)
        p0_2d = np.array([0.0, 0.0])
        p2_2d = to_np_2d(p2_3d)
        p3_2d = to_np_2d(p3_3d)
        p5_2d = to_np_2d(p5_3d)
        tcp_2d= to_np_2d(p_tcp_3d)

        # 6) 计算四段向量
        v0 = p2_2d - p0_2d     # 基座->关节2
        v1 = p3_2d - p2_2d     # 关节2->关节3
        v2 = p5_2d - p3_2d     # 关节3->关节5
        v3 = tcp_2d - p5_2d    # 关节5->TCP

        # 长度
        def vec_len(a):
            return math.sqrt(a[0]*a[0] + a[1]*a[1])
        L0 = vec_len(v0)
        L1 = vec_len(v1)
        L2 = vec_len(v2)
        L3 = vec_len(v3)
        lengths = np.array([L0, L1, L2, L3], dtype=float)

        # 7) 计算三个关节处的夹角(逆时针>0, 顺时针<0)
        def angle_between_2d(a,b):
            cross = a[0]*b[1] - a[1]*b[0]
            dot   = a[0]*b[0] + a[1]*b[1]
            return math.atan2(cross, dot)

        alpha2 = angle_between_2d(v0, v1)  # 关节2
        alpha3 = angle_between_2d(v1, v2)  # 关节3
        alpha5 = angle_between_2d(v2, v3)  # 关节5
        angles = np.array([alpha2, alpha3, alpha5], dtype=float)

        # 8) 打印结果
        print("\n-- 2D Projected Positions (y,z) --")
        print("p2   =", p2_2d)
        print("p3   =", p3_2d)
        print("p5   =", p5_2d)
        print("pTCP =", tcp_2d)

        print("\n-- 2D Link Lengths (m) --")
        print(f"L0={L0:.5f}, L1={L1:.5f}, L2={L2:.5f}, L3={L3:.5f}")

        print("\n-- 2D Joint Angles [radians, alpha2, alpha3, alpha5] --")
        print(angles)
        print("Angles (deg) =", angles*180.0/math.pi)

        # 9) 返回
        return lengths, angles, p2_2d, p3_2d, p5_2d, tcp_2d