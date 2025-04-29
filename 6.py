#!/usr/bin/env python3
"""
playground_2R.py  –  Six-axis / 3-D position-error calibration playground

▶ 主要变化
   • 数据读取不再裁剪到 3 个关节，而是完整保留 6 个关节角。
   • 测量向量改为 3-D 位置 (X,Y,Z)。
   • 采用新的 CalibrationModel6RComplNl 模型（见 calib_models.py）。
   • err_par_switch 自动根据关节数与 comp_model 阶次生成，免去手工写开关矩阵。
   • 误差评估、合规性绘图与 gravity-torque 接口均已扩展至 6 轴。
"""

from calibration.calib_models import CalibrationModel6RComplNl
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# ---------- 机器人名义 DH / Kin 参数 ----------
NDOF = 6
KINVEC = [
    [0, 0, 0.167899996],
    [0, -0.060899999, 0.0970999971],
    [0, 0, 0.444000006],
    [0.112999998, 0.060899999, 0.109999999],
    [0.356999993, 0.056499999, 0],
    [0.101000004, -0.056499999, 0.0799999982],
]
# TCP (非 RAPID 方向约定, x 轴朝外)
TCP = [0.135, 0.0, -0.07]

# 关节限位 (示例：全轴 ±180°)
JOINT_LIMITS = (np.ones((NDOF, 2)) * np.array([[-180, 180]])).astype(float) / 180 * np.pi

# ---------- 主流程 ----------
def main() -> None:
    # ★ 基本参数配置 ★ --------------------------------------------------
    calib_with_load = True           # 是否考虑末端负载
    compliance_model = "Cubic"       # "Lin" / "Quad" / "Cubic"
    use_small_dataset = False
    validation_ratio = 0.2           # 验证集占比
    use_fake_data = False            # 若为 True 则生成仿真数据
    num_pts = 500
    noise_level = 0.0000
    seed = 1
    # ------------------------------------------------------------------

    # 载荷文件、测量数据路径
    load_fct_file = (
        "load_func_holder_5kg.pkl" if calib_with_load else "load_func_holder_only.pkl"
    )
    data_file = os.path.join(
        "data",
        "JT100_holder_with_load.tri"
        if use_small_dataset and calib_with_load
        else ("JT500_holder_with_load.tri" if calib_with_load else
              ("JT100_holder_only.tri" if use_small_dataset else "JT500_holder_only.tri"))
    )

    with open(load_fct_file, "rb") as f:
        wrench_fct = pickle.load(f)

    # ------------ 计算各阶次需要的误差参数长度 -----------------
    order_map = {"Lin": 1, "Quad": 2, "Cubic": 3}
    if compliance_model not in order_map:
        raise ValueError("Unsupported compliance model: " + compliance_model)
    num_comp_param = order_map[compliance_model]

    # ----------- 加载 / 生成 数据 (q, tcp) ---------------------
    if use_fake_data:
        # ------ 生成虚拟数据 ------
        np.random.seed(seed)
        model_gt = CalibrationModel6RComplNl(
            kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct, comp_model=compliance_model
        )
        # 关节角
        q_calib = angle_generator(num_pts, JOINT_LIMITS, seed=seed)
        q_valid = angle_generator(num_pts, JOINT_LIMITS, seed=seed + 50)
        # ground-truth 随机误差参数
        rng = np.random.default_rng(seed + 100)
        err_params_gt = 1e-3 * rng.standard_normal(
            ( (6 + num_comp_param) * 7, )
        )  # 6 轴 + 工具
        # 测量值 (含噪声)
        tcp_calib = model_gt.generate_fake_data(q_calib, err_params_gt) + \
                    noise_level * rng.standard_normal((num_pts, 3))
        tcp_valid = model_gt.generate_fake_data(q_valid, err_params_gt) + \
                    noise_level * rng.standard_normal((num_pts, 3))
    else:
        # ------ 读取真实 tri 文件 ------
        q_all, tcp_all = parse_tri_file(data_file)       # q_all:(N,6)  tcp_all:(N,3)
        q_all, tcp_all = np.asarray(q_all), np.asarray(tcp_all)

        # 训练 / 验证拆分
        split = int((1 - validation_ratio) * len(q_all))
        q_calib, q_valid = q_all[:split], q_all[split:]
        tcp_calib, tcp_valid = tcp_all[:split], tcp_all[split:]

    print(f"[INFO] Calib poses: {len(q_calib)}   Valid poses: {len(q_valid)}")

    # --------- 构造 6R-3D 误差模型 ---------
    model = CalibrationModel6RComplNl(
        kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct, comp_model=compliance_model
    )

    # --------- 创建误差参数开关矩阵 (全轴/全参数可调) ----------
    num_params_per_ax = 6 + num_comp_param   # 关节: 6 几何 + Nc 柔顺
    def make_switch(geom_flags, keep_comp=False):
        """geom_flags: 长度6的 True/False 列表；keep_comp 决定是否保留柔顺性"""
        return geom_flags + ([True] * num_comp_param if keep_comp else [False] * num_comp_param)

    # —— 针对每个关节 ——（True=优化，False=锁 0）
    sw_ax1 = make_switch([True, True, False, False, False, True], keep_comp=False)
    sw_ax2 = make_switch([True, True, False, True, False, False], keep_comp=True)
    sw_ax3 = make_switch([True, True, False, True, False, False], keep_comp=True)
    sw_ax4 = make_switch([False, False, False, False, False, True], keep_comp=False)
    sw_ax5 = make_switch([True, True, False, True, False, False], keep_comp=True)
    sw_ax6 = make_switch([False, False, False, False, False, True], keep_comp=False)
    sw_tool = make_switch([True, True, True, False, False, False], keep_comp=False)

    err_par_switch = [
        sw_ax1, sw_ax2, sw_ax3, sw_ax4, sw_ax5, sw_ax6, sw_tool
    ]
    model.set_error_par_switch(err_par_switch)

    # --------- 误差初值 (0) ---------------
    init_guess = np.zeros((7 * num_params_per_ax,))

    # --------- 调用求解器 ---------------
    solver = CalibrationSolverIpopt()
    solved_params = solver.solve_calibration(
        model=model,
        q=q_calib,
        meas=tcp_calib,
        comp_model=compliance_model,
        initial_guess=init_guess,
    )

    # --------- 误差评估 -----------------
    def print_err(tag, mean_err, max_err):
        print(f"{tag:<10s}  mean={mean_err*1000:.2f} mm   max={max_err*1000:.2f} mm")

    mean_uncal, max_uncal = model.get_error(q_calib, tcp_calib, init_guess)
    mean_cal, max_cal = model.get_error(q_calib, tcp_calib, solved_params)
    mean_val, max_val = model.get_error(q_valid, tcp_valid, solved_params)

    print_err("uncalib", mean_uncal, max_uncal)
    print_err("calib",   mean_cal,  max_cal)
    print_err("valid",   mean_val,  max_val)

    # -------- 绘制柔顺性曲线 (可选) ---------
    tau_tr = np.max(np.abs(model.get_gravity_torque(q_calib)), axis=0)
    fig, axarr = plt.subplots(3, 2, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)
    model.plot_compliance(axarr, solved_params, tau_tr)
    plt.show()


if __name__ == "__main__":
    main()