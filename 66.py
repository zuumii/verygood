"""
calib_models.py –  6-axis 3-D calibration model with joint compliance
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import casadi as cs
import numpy as np
import utils.utils as ut
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# 抽象基类 – 保留不变
# -----------------------------------------------------------
class CalibrationModel(ABC):
    _symbolic_meas_fct: cs.Function
    _error_param_jacobian_fct: cs.Function
    _err_par_switch: List[List[bool]]

    def __init__(self) -> None:
        self._err_par_switch = []
        self._generate_symbolic_functions()

    @abstractmethod
    def _generate_symbolic_functions(self) -> None:
        ...

    @abstractmethod
    def generate_fake_data(
        self,
        q: List[List[float]],
        err_params: np.ndarray,
    ) -> List[List[float]]:
        ...

    def set_error_par_switch(self, err_par_switch: List[List[bool]]) -> None:
        self._err_par_switch = err_par_switch

    def get_error_par_switch(self) -> List[List[bool]]:
        return self._err_par_switch

    def get_symbolic_meas_fct(self) -> cs.Function:
        return self._symbolic_meas_fct

    def get_error_param_jacobian_fct(self) -> cs.Function:
        return self._error_param_jacobian_fct


# ===================================================================
# 6-R / 3-D 位置误差 + 柔顺性模型
# ===================================================================
class CalibrationModel6RComplNl(CalibrationModel):
    """
    • 每个关节 6 几何误差 + N 个柔顺性参数 (N=1/2/3)。
    • 最后再附加 6 个工具坐标误差参数。
    err_params 结构：(ax1[6+N], ax2[6+N], … ax6[6+N], tool[6])
    """

    def __init__(
        self,
        kinvec: List[List[float]],
        tcp: List[float],
        load_fct: cs.Function,
        comp_model: str,
        tau_tr: np.ndarray = None,
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

        # ---------- 关节柔顺性模型 q_compl = f(tau, C) ----------
        tau = cs.SX.sym("tau")
        if comp_model == "Lin":
            C = cs.SX.sym("C")            # 1 参数
            q_c = C * tau
            self.num_comp_param = 1
            self.comp_fct = cs.Function("comp", [tau, C], [q_c])
        elif comp_model == "Quad":
            C = cs.SX.sym("C", 2)         # 2 参数
            q_c = C[0] * tau + 1e-2 * C[1] * cs.sign(tau) * tau**2
            self.num_comp_param = 2
            self.comp_fct = cs.Function("comp", [tau, C], [q_c])
        elif comp_model == "Cubic":
            C = cs.SX.sym("C", 3)         # 3 参数
            q_c = (
                C[0] * tau
                + 1e-2 * C[1] * cs.sign(tau) * tau**2
                + 1e-4 * C[2] * tau**3
            )
            self.num_comp_param = 3
            self.comp_fct = cs.Function("comp", [tau, C], [q_c])
        else:
            raise ValueError("Unsupported comp_model")

        super().__init__()

    # ----------------------------------------------------------
    # (I) 生成符号 TCP→3D 位置函数 & 误差对参数 Jacobian
    # ----------------------------------------------------------
    def _generate_symbolic_functions(self) -> None:
        q = cs.SX.sym("q", 6)  # 6 个关节角
        taul = cs.SX.sym("taul", 6)
        err_params = cs.SX.sym("err_params", (6 + self.num_comp_param) * 7)   # 7 = 6 关节 + tool


        # ---------------- 动力学 – 重力力矩 -----------------
        qd = cs.SX.zeros(6, 1)
        qdd = cs.SX.zeros(6, 1)
        load_trq = -self.load_fct(q, qd, qdd)[0:6, 0:3]
        for i in range(6):
            taul[i] = load_trq[i, 1]  # 只取关节驱动轴方向力矩

        # ---------------- 正向运动学 ----------------------
        T = cs.SX.eye(4)
        for joint in range(6):
            err_joint = err_params[
                joint * (6 + self.num_comp_param) : (joint + 1) * (6 + self.num_comp_param)
            ]
            T = cs.mtimes(T, self._get_joint_trafo(q[joint], taul[joint], joint, err_joint))

        # 工具坐标误差 (最后 6 个)
        tool_err = err_params[-(6 + self.num_comp_param) : -(self.num_comp_param)]
        T_tool = cs.mtimes(ut.trans_mat(self.tcp), ut.trans_mat(tool_err))
        T = cs.mtimes(T, T_tool)

        tcp_pos = T[0:3, 3]  # X,Y,Z
        self._symbolic_meas_fct = cs.Function("tcp3d", [q, err_params], [tcp_pos])

        jac = cs.jacobian(tcp_pos, err_params)
        self._error_param_jacobian_fct = cs.Function("jac_err", [q, err_params], [jac])

    # ----------------------------------------------------------
    # (II) 生成合成 / 仿真数据 (支持 Q 为 (N,6))
    # ----------------------------------------------------------
    def generate_fake_data(
        self,
        q: np.ndarray,
        err_params: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q)
        m = [self._symbolic_meas_fct(qi, err_params).full().ravel() for qi in q]
        return np.vstack(m)

    # ----------------------------------------------------------
    # (III) 单个关节误差变换矩阵
    # ----------------------------------------------------------
    def _get_joint_trafo(self, q_i, tau_i, ax_no, err_joint):
        """
        err_joint = [dLy, dLz, dLx, dRx, dRy, dRz, C...]  (len = 6+Nc)
        顺序保持与旧版一致（前三个平移 Y,Z,X；后三个转动 X,Y,Z）
        """
        trans_nom = ut.trans_mat(self.kinvec[ax_no])
        err_trans = ut.trans_mat(err_joint[:3])

        # 柔顺性 Δq
        if self.num_comp_param > 0:
            dq_c = self.comp_fct(tau_i, err_joint[6:])
        else:
            dq_c = 0.0

        # 三自由度误差转动
        rot_err1 = ut.rot_mat_x(err_joint[3])
        rot_err2 = ut.rot_mat_y(err_joint[4])
        rot_err3 = ut.rot_mat_z(err_joint[5])

        # 名义关节转动 (含柔顺)
        if self.rotation_axes[ax_no] == ut.RotationAxis.X:
            rot_joint = ut.rot_mat_x(q_i + dq_c)
        elif self.rotation_axes[ax_no] == ut.RotationAxis.Y:
            rot_joint = ut.rot_mat_y(q_i + dq_c)
        elif self.rotation_axes[ax_no] == ut.RotationAxis.Z:
            rot_joint = ut.rot_mat_z(q_i + dq_c)
        else:
            raise ValueError("Unknown axis type")

        return cs.mtimes([trans_nom, err_trans, rot_err1, rot_err2, rot_err3, rot_joint])

    # ----------------------------------------------------------
    # (IV) 误差评估 (返回均值/最大 欧氏距离)
    # ----------------------------------------------------------
    def get_error(
        self,
        q: np.ndarray,
        meas: np.ndarray,
        calib_params: np.ndarray,
    ) -> Tuple[float, float]:
        errs = [
            np.linalg.norm(self._symbolic_meas_fct(qi, calib_params).full().ravel() - mi)
            for qi, mi in zip(q, meas)
        ]
        return float(np.mean(errs)), float(np.max(errs))

    # ----------------------------------------------------------
    # (V) 关节 → 重力力矩 (数值)
    # ----------------------------------------------------------
    def get_gravity_torque(self, q: np.ndarray):
        q = np.atleast_2d(q)  # (N,6)
        qd = np.zeros((6, 1))
        qdd = np.zeros((6, 1))
        taus = []
        for qi in q:
            tau = -self.load_fct(qi, qd, qdd)[0:6, 0:3]
            taus.append(tau[:, 1])
        return np.asarray(taus)

    # ----------------------------------------------------------
    # (VI) 绘制柔顺性曲线 (每轴一图)
    # ----------------------------------------------------------
    def plot_compliance(self, axarr, calib_params: np.ndarray, tau_tr: np.ndarray):
        num_pts = 400
        taus = [np.linspace(-t, t, num_pts) for t in tau_tr]
        taus = np.asarray(taus)  # shape (6, num_pts)
        for j in range(6):
            C = calib_params[j * (6 + self.num_comp_param) + 6 : (j + 1) * (6 + self.num_comp_param)]
            dq = self.comp_fct(taus[j], C)
            row, col = j % 3, j // 3
            axarr[row, col].plot(taus[j], dq)
            axarr[row, col].set_title(f"Joint {j+1}")
            axarr[row, col].set_xlabel("τ [N·m]")
            axarr[row, col].set_ylabel("Δq [rad]")
            axarr[row, col].grid(True)