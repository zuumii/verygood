"""
calib_solvers.py – Least-squares calibration solver (IPOPT) – 通用维度
"""
from typing import List
import casadi as cs
import numpy as np
from calibration.calib_models import CalibrationModel


class CalibrationSolverIpopt:
    """
    • 将误差平方和作为目标函数。
    • err_par_switch 由模型提供 – 控制哪些参数可调。
    """

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

        # ---------- 参数固定 / 等式约束 ----------
        for i in range(num_ax):
            for j in range(num_params_per_ax):
                if not err_switch[i][j]:
                    opti.subject_to(x[i * num_params_per_ax + j] == 0)

        # ---------- 柔顺性单调/凹性约束（简化版） ----------
        # 仅对存在柔顺性参数的关节施加 C[0] ≥ 0 约束 (保证正刚度)
        if num_params_per_ax > 6:
            for i in range(6):  # 前 6 个关节
                c0_idx = i * num_params_per_ax + 6  # 第一个柔顺性参数
                opti.subject_to(x[c0_idx] >= 0)

        # ---------- solve ----------
        opti.solver("ipopt")
        opti.set_initial(x, initial_guess)
        sol = opti.solve()
        return np.asarray(sol.value(x))