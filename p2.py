#!/usr/bin/env python3
from calibration.calib_models import CalibrationModel6RComplNl
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np

import torch, torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os

# ---------- DH / Kin  ----------
NDOF = 6
KINVEC = [
    [0, 0, 0.167899996],
    [0, -0.060899999, 0.0970999971],
    [0, 0, 0.444000006],
    [0.112999998, 0.060899999, 0.109999999],
    [0.356999993, 0.056499999, 0],
    [0.101000004, -0.056499999, 0.0799999982],
]
# TCP 
# TOOL_FRAME \
# 7.700000e-002 -7.000000e-002 1.300000e-001
# TCP = [0.077, -0.07, 0.13]

TCP = [0.135, -0.09, -0.07]

# TCP=[0.135, 0.0, -0.07]

# joint limit
JOINT_LIMITS = (np.ones((NDOF, 2)) * np.array([[-180, 180]])).astype(float) / 180 * np.pi


def main() -> None:
   
    # calib_with_load = True          
    compliance_model = "Lin"       # "Lin" / "Quad" / "Cubic"
    
    validation_ratio = 0.2           
    use_fake_data = False            
    num_pts = 100
    noise_level = 0.0000
    seed = 1
    # ------------------------------------------------------------------

    # pkl file
    load_fct_file_high = (
        "load_func_holder_5kg.pkl"
    )
    load_fct_file_low = (
        "load_func_holder_only.pkl"
    )
    
    data_file_high = os.path.join(
        "data",
        "1500-multidir_JT120x60_5kg_v1000.tri"
    )
    
    data_file_low = os.path.join(
        "data",
        "1500-multidir_JT120x60_0kg_v1000.tri"
    )
    
    

    with open(load_fct_file_high, "rb") as f:
        wrench_fct_high = pickle.load(f)
        
    with open(load_fct_file_low, "rb") as f:
        wrench_fct_low = pickle.load(f)

    # ------------ Error para -----------------
    order_map = {"Lin": 1, "Quad": 2, "Cubic": 3}
    if compliance_model not in order_map:
        raise ValueError("Unsupported compliance model: " + compliance_model)
    num_comp_param = order_map[compliance_model]

    # ----------- fake(q, tcp) ---------------------
    # if use_fake_data:

    #     np.random.seed(seed)
    #     model_gt = CalibrationModel6RComplNl(
    #         kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct, comp_model=compliance_model
    #     )

    #     q_calib = angle_generator(num_pts, JOINT_LIMITS, seed=seed)
    #     q_valid = angle_generator(num_pts, JOINT_LIMITS, seed=seed + 50)
    #     # ground-truth 
    #     rng = np.random.default_rng(seed + 100)
    #     err_params_gt = 1e-3 * rng.standard_normal(
    #         ( (6 + num_comp_param) * 7, )
    #     )  

    #     tcp_calib = model_gt.generate_fake_data(q_calib, err_params_gt) + \
    #                 noise_level * rng.standard_normal((num_pts, 3))
    #     tcp_valid = model_gt.generate_fake_data(q_valid, err_params_gt) + \
    #                 noise_level * rng.standard_normal((num_pts, 3))
    # else:
        # ------ tri file ------
    q_all_high, tcp_all_high = parse_tri_file(data_file_high)       # q_all:(N,6)  tcp_all:(N,3)
    q_all_high, tcp_all_high = np.asarray(q_all_high), np.asarray(tcp_all_high)
    
    q_all_low, tcp_all_low = parse_tri_file(data_file_low)       # q_all:(N,6)  tcp_all:(N,3)
    q_all_low, tcp_all_low= np.asarray(q_all_low), np.asarray(tcp_all_low)
##############################
    max_samples = 300
    q_all_high, tcp_all_high = q_all_high[:max_samples], tcp_all_high[:max_samples]
    q_all_low,  tcp_all_low  = q_all_low[:max_samples], tcp_all_low[:max_samples]
    ####################################
    split_high = int((1 - validation_ratio) * len(q_all_high))
    q_calib_high, q_valid_high = q_all_high[:split_high], q_all_high[split_high:]
    tcp_calib_high, tcp_valid_high = tcp_all_high[:split_high], tcp_all_high[split_high:]
    
    split_low = int((1 - validation_ratio) * len(q_all_low))
    q_calib_low, q_valid_low = q_all_low[:split_low], q_all_low[split_low:]
    tcp_calib_low, tcp_valid_low = tcp_all_low[:split_low], tcp_all_low[split_low:]

    print(f"[INFO] Calib poses high: {len(q_calib_high)}   Valid poses high: {len(q_valid_high)}")
    print(f"[INFO] Calib poses low: {len(q_calib_low)}   Valid poses low: {len(q_valid_low)}")

    # --------- 6R-3D---------
    model_high = CalibrationModel6RComplNl(
        kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_high, comp_model=compliance_model
    )
    model_low = CalibrationModel6RComplNl(
        kinvec=KINVEC, tcp=TCP, load_fct=wrench_fct_low, comp_model=compliance_model
    )

    # ---------  para lock ----------
    num_params_per_ax = 6 + num_comp_param   # joint 6 kin and compliance
    def make_switch(geom_flags, keep_comp=False):
        return geom_flags + ([True] * num_comp_param if keep_comp else [False] * num_comp_param)

    # —— each joint ——


    # 0 FALSE FALSE FALSE FALSE FALSE FALSE \
    # 1 TRUE TRUE TRUE TRUE TRUE TRUE \
    # 2 TRUE FALSE FALSE TRUE TRUE FALSE \
    # 3 FALSE FALSE TRUE TRUE TRUE TRUE \
    # 4 FALSE TRUE TRUE TRUE FALSE TRUE \
    # 5 TRUE FALSE TRUE FALSE TRUE TRUE \
    # 6 FALSE TRUE TRUE TRUE FALSE TRUE \
    # 7 TRUE FALSE TRUE FALSE FALSE FALSE
    
    # 0 FALSE FALSE FALSE FALSE FALSE FALSE \
    # 1 TRUE TRUE TRUE TRUE TRUE TRUE \
    # 2 TRUE FALSE FALSE TRUE TRUE FALSE \
    # 3 FALSE FALSE TRUE TRUE TRUE TRUE \
    # 4 FALSE TRUE TRUE TRUE FALSE TRUE \
    # 5 TRUE FALSE TRUE FALSE TRUE TRUE \
    # 6 FALSE TRUE TRUE TRUE FALSE TRUE \
    # 7 TRUE FALSE TRUE FALSE FALSE FALSE

    sw_ax1 = make_switch([True, True, True, True, True, True], keep_comp=False)
    sw_ax2 = make_switch([True, False, False, True, True, False], keep_comp=True)
    sw_ax3 = make_switch([False, False, True, True, True, True], keep_comp=True)
    sw_ax4 = make_switch([False, True, True, True, False, True], keep_comp=True)
    sw_ax5 = make_switch([True, False, True, False, True, True], keep_comp=False)
    sw_ax6 = make_switch([False, True, True, True, False, False], keep_comp=False)
    sw_tool = make_switch([True, False, True, False, False, False], keep_comp=False)


    # sw_ax1 = make_switch([False, False, False, False, False, False,], keep_comp=False)
    # sw_ax2 = make_switch([False, False, False, False, False, False,], keep_comp=False)
    # sw_ax3 = make_switch([False, False, False, False, False, False,], keep_comp=False)
    # sw_ax4 = make_switch([False, False, False, False, False, False], keep_comp=False)
    # sw_ax5 = make_switch([False, False, False, False, False, False], keep_comp=False)
    # sw_ax6 = make_switch([False, False, False, False, False, False], keep_comp=False)
    # sw_tool = make_switch([True, True, True, False, False, False], keep_comp=False)

    err_par_switch = [
        sw_ax1, sw_ax2, sw_ax3, sw_ax4, sw_ax5, sw_ax6, sw_tool
    ]
    
    
    # ==================
    err_par_switch_comp_only = [     
        [False, False, False, False, False, False, False],  # J1
        [False, False, False, False, False, False,  True],  # J2
        [False, False, False, False, False, False,  True],  # J3
        [False, False, False, False, False, False,  True],  # J4
        [False, False, False, False, False, False,  False],  # J5
        [False, False, False, False, False, False, False],  # J6
        [False, False, False, False, False, False, False],  # TCP
    ]

    err_par_switch_geom_only = [       
        [ True,  True,  True,  True,  True,  True, False],  # J1
        [ True, False, False,  True,  True, False, False],  # J2
        [False, False,  True,  True,  True,  True, False],  # J3
        [False,  True,  True,  True, False,  True, False],  # J4
        [ True, False,  True, False,  True,  True, False],  # J5
        [False,  True,  True,  True, False,  True, False],  # J6
        [ True, False,  True, False, False, False, False],  # TCP
    ]

    def print_err(tag, mean_err, max_err):
        print(f"{tag:<20s}  mean={mean_err * 1000:.2f} mm   max={max_err * 1000:.2f} mm")

    
    
    def run_calibration(tag, err_switch):
        """对一套 error‐switch 在高 / 低载数据集上分别做标定 + 评估"""
        # ---------- ----------
        mH = CalibrationModel6RComplNl(kinvec=KINVEC, tcp=TCP,
                                    load_fct=wrench_fct_high, comp_model=compliance_model)
        mL = CalibrationModel6RComplNl(kinvec=KINVEC, tcp=TCP,
                                    load_fct=wrench_fct_low , comp_model=compliance_model)
        mH.set_error_par_switch(err_switch)
        mL.set_error_par_switch(err_switch)

        # --------- ----------
        solver = CalibrationSolverIpopt()
        pH = solver.solve_calibration(mH, q_calib_high, tcp_calib_high,
                                    compliance_model, np.zeros(7*num_params_per_ax))
        pL = solver.solve_calibration(mL, q_calib_low , tcp_calib_low ,
                                    compliance_model, np.zeros(7*num_params_per_ax))

        # ---------- ----------
        def err(model, q_val, tcp_val, pars):
            return model.get_error(q_val, tcp_val, pars)

        mh_c, xh_c = err(mH, q_calib_high, tcp_calib_high, pH)
        mh_v, xh_v = err(mH, q_valid_high, tcp_valid_high, pH)
        ml_c, xl_c = err(mL, q_calib_low , tcp_calib_low , pL)
        ml_v, xl_v = err(mL, q_valid_low , tcp_valid_low , pL)

        print(f"\n===== {tag} =====")
        print_err("High | Calib", mh_c, xh_c)
        print_err("High | Valid", mh_v, xh_v)
        print_err("Low  | Calib", ml_c, xl_c)
        print_err("Low  | Valid", ml_v, xl_v)

        # ---------- ----------
        stride = 6 + num_comp_param
        print(f"\n--- {tag} Parameters (High-load) ---")
        for i in range(0, len(pH), stride):
            print("[" + " ".join(f"{x:.4g}" for x in pH[i:i+stride]) + "]")
        print(f"\n--- {tag} Parameters (Low-load) ---")
        for i in range(0, len(pL), stride):
            print("[" + " ".join(f"{x:.4g}" for x in pL[i:i+stride]) + "]")

    # ========= =========
    run_calibration("Comp-Only  (几何锁死)", err_par_switch_comp_only)
    run_calibration("Geom-Only  (柔顺锁死)", err_par_switch_geom_only)
        
        
        
        
        
    
    
    ################################################################
    model_high.set_error_par_switch(err_par_switch)
    model_low.set_error_par_switch(err_par_switch)

    # --------- initial reeor para ---------------
    init_guess = np.zeros((7 * num_params_per_ax,))

    # --------- Solver ---------------
    solver_high = CalibrationSolverIpopt()
    solved_params_high = solver_high.solve_calibration(
        model=model_high,
        q=q_calib_high,
        meas=tcp_calib_high,
        comp_model=compliance_model,
        initial_guess=init_guess,
    )
    solver_low = CalibrationSolverIpopt()
    solved_params_low = solver_low.solve_calibration(
        model=model_high,
        q=q_calib_low,
        meas=tcp_calib_low,
        comp_model=compliance_model,
        initial_guess=init_guess,
    )
    # --------- error eval -----------------
    def print_err(tag, mean_err, max_err):
        print(f"{tag:<10s}  mean={mean_err*1000:.2f} mm   max={max_err*1000:.2f} mm")

    mean_uncal_high, max_uncal_high = model_high.get_error(q_calib_high, tcp_calib_high, init_guess)
    mean_cal_high, max_cal_high = model_high.get_error(q_calib_high, tcp_calib_high, solved_params_high)
    mean_val_high, max_val_high = model_high.get_error(q_valid_high, tcp_valid_high, solved_params_high)
    
    mean_uncal_low, max_uncal_low = model_low.get_error(q_calib_low, tcp_calib_low, init_guess)
    mean_cal_low, max_cal_low = model_low.get_error(q_calib_low, tcp_calib_low, solved_params_low)
    mean_val_low, max_val_low = model_low.get_error(q_valid_low, tcp_valid_low, solved_params_low)
    

    mean_cal_high_in_low_paras, max_cal_high_in_low_paras = model_high.get_error(q_calib_high, tcp_calib_high, solved_params_low)
    mean_val_high_in_low_paras, max_val_high_in_low_paras = model_high.get_error(q_valid_high, tcp_valid_high, solved_params_low)
    
    mean_cal_low_in_high_paras, max_cal_low_in_high_paras = model_low.get_error(q_calib_low, tcp_calib_low, solved_params_high)
    mean_val_low_in_high_paras, max_val_low_in_high_paras = model_low.get_error(q_valid_low, tcp_valid_low, solved_params_high)

    # --------  --------
    print_err("High | Uncalibrated", mean_uncal_high, max_uncal_high)
    print_err("High | Calibrated", mean_cal_high, max_cal_high)
    print_err("High | Validation", mean_val_high, max_val_high)

    # -------- --------
    print_err("Low  | Uncalibrated", mean_uncal_low, max_uncal_low)
    print_err("Low  | Calibrated", mean_cal_low, max_cal_low)
    print_err("Low  | Validation", mean_val_low, max_val_low)

    # -------- --------
    print_err("High (in Low's Paras) | Calib", mean_cal_high_in_low_paras, max_cal_high_in_low_paras)
    print_err("High (in Low's Paras) | Valid", mean_val_high_in_low_paras, max_val_high_in_low_paras)

    print_err("Low  (in High's Paras) | Calib", mean_cal_low_in_high_paras, max_cal_low_in_high_paras)
    print_err("Low  (in High's Paras) | Valid", mean_val_low_in_high_paras, max_val_low_in_high_paras)
    
    
    # ==================
    stride = 6 + num_comp_param        
    print("\n=== Compliance parameters per joint ===")
    print("          High-load                     Low-load")
    for j in range(7):                
        C_high = solved_params_high[j*stride+6 : (j+1)*stride]
        C_low  = solved_params_low [j*stride+6 : (j+1)*stride]
        print(f"Joint {j+1}: {C_high}   {C_low}")
        
        
    # ==================
    def zero_compliance(params: np.ndarray, num_comp_param: int) -> np.ndarray:
        """返回一个新数组，把每个关节最后 num_comp_param 个柔顺参数置 0"""
        stride = 6 + num_comp_param
        out = params.copy()
        for j in range(7):
            out[j*stride+6 : (j+1)*stride] = 0.0
        return out

    params_high_nocomp = zero_compliance(solved_params_high, num_comp_param)
    params_low_nocomp  = zero_compliance(solved_params_low , num_comp_param)


    mean_nc_high, max_nc_high = model_high.get_error(q_valid_high, tcp_valid_high, params_high_nocomp)
    mean_nc_low , max_nc_low  = model_low .get_error(q_valid_low , tcp_valid_low , params_low_nocomp)

    print("\n=== Error WITHOUT compliance parameters ===")
    print_err("High-load  (no-comp)", mean_nc_high, max_nc_high)
    print_err("Low-load   (no-comp)", mean_nc_low , max_nc_low )
    
    def extract_compliance_only_params(solved_params: np.ndarray, num_comp_param: int) -> np.ndarray:
        new_params = []
        stride = 6 + num_comp_param
        for i in range(7):  # 6 joints + tool
            block = solved_params[i * stride : (i + 1) * stride]
            zeros = np.zeros(6)
            C = block[6:]  
            new_params.append(np.concatenate([zeros, C]))
        return np.concatenate(new_params)
    

    componly_params_high = extract_compliance_only_params(solved_params_high, num_comp_param)
    componly_params_low  = extract_compliance_only_params(solved_params_low, num_comp_param)


    mean_componly_high, max_componly_high = model_high.get_error(q_valid_high, tcp_valid_high, componly_params_high)
    mean_componly_low,  max_componly_low  = model_low.get_error(q_valid_low,  tcp_valid_low,  componly_params_low)


    print_err("componly_high", mean_componly_high, max_componly_high)
    print_err("componly_low",  mean_componly_low,  max_componly_low)
    
    tau_calib_high = model_high.get_gravity_torque(q_calib_high)
    tau_valid_high = model_high.get_gravity_torque(q_valid_high)
    
    tau_valid_low = model_low.get_gravity_torque(q_valid_low)
    print("\n=== Compliance Calibration Evaluation ===")
    dq_valid_high = model_high.get_dq_from_tau(tau_valid_high, solved_params_high)
    dq_valid_low  = model_low.get_dq_from_tau(tau_valid_low, solved_params_low)
    def print_dq_per_joint(tag: str, dq: np.ndarray):
        dq_deg = dq * 180 / np.pi  # 转为度
        print(f"{tag}:")
        for j in range(dq.shape[1]):
            mean_j = np.mean(np.abs(dq_deg[:, j]))
            max_j = np.max(np.abs(dq_deg[:, j]))
            print(f"  Joint {j+1}: mean Δq = {mean_j:.4f} deg   max Δq = {max_j:.4f} deg")

    print_dq_per_joint("dq_high (compliance)", dq_valid_high)
    print_dq_per_joint("dq_low  (compliance)", dq_valid_low)
    
    
    # stride = 6 + num_comp_param
    # for i in range(0, len(solved_params), stride):
    #     print(solved_params[i:i+stride])
    stride = 6 + num_comp_param
    for i in range(0, len(solved_params_high), stride):
        line = solved_params_high[i:i+stride]
        print("[" + " ".join(f"{x:.4g}" for x in line) + "]", end=" \n")

    for i in range(0, len(solved_params_low), stride):
        line = solved_params_low[i:i+stride]
        print("[" + " ".join(f"{x:.4g}" for x in line) + "]", end=" \n")
        
    solver_double = CalibrationSolverIpopt()
    solved_params_joint = solver_double.solve_double_calibration(
        model_high=model_high,
        model_low=model_low,
        q_high=q_calib_high,
        meas_high=tcp_calib_high,
        q_low=q_calib_low,
        meas_low=tcp_calib_low,
        comp_model=compliance_model,
        initial_guess=np.zeros(7 * num_params_per_ax),
    )    
    
    mean_joint_high, max_joint_high = model_high.get_error(q_valid_high, tcp_valid_high, solved_params_joint)
    mean_joint_low, max_joint_low = model_low.get_error(q_valid_low, tcp_valid_low, solved_params_joint)
            
    print("\n=== Double Calibration Evaluation ===")
    print_err("JointCalib on High (Validation)", mean_joint_high, max_joint_high)
    print_err("JointCalib on Low  (Validation)", mean_joint_low, max_joint_low)
    
    for i in range(0, len(solved_params_joint), stride):
        line = solved_params_joint[i:i+stride]
        print("[" + " ".join(f"{x:.4g}" for x in line) + "]", end=" \n")
    

    # -------- plt compliance curves ---------
    # tau_tr = np.max(np.abs(model.get_gravity_torque(q_calib)), axis=0)
    # fig, axarr = plt.subplots(3, 2, figsize=(10, 12))
    # plt.subplots_adjust(hspace=0.4)
    # model.plot_compliance(axarr, solved_params, tau_tr, err_par_switch)
    # plt.show()
    
    
    # ========= just kin or compliance ==========
    def zero_kin_only(params: np.ndarray, stride: int) -> np.ndarray:

        out = params.copy()
        for j in range(7):
            out[j*stride : j*stride+6] = 0.0
        return out

    def zero_comp_only(params: np.ndarray, stride: int) -> np.ndarray:

        out = params.copy()
        for j in range(7):
            out[j*stride+6 : (j+1)*stride] = 0.0
        return out

    kin_zero_params  = zero_kin_only (solved_params_joint, stride)
    comp_zero_params = zero_comp_only(solved_params_joint, stride)

    def eval_and_print_split(tag: str, params: np.ndarray):

        m_ch, x_ch = model_high.get_error(q_calib_high, tcp_calib_high, params)
        m_vh, x_vh = model_high.get_error(q_valid_high, tcp_valid_high, params)

        m_cl, x_cl = model_low .get_error(q_calib_low , tcp_calib_low , params)
        m_vl, x_vl = model_low .get_error(q_valid_low , tcp_valid_low , params)

        print(f"\n--- {tag} ---")
        print_err("High | Calib", m_ch, x_ch)
        print_err("High | Valid", m_vh, x_vh)
        print_err("Low  | Calib", m_cl, x_cl)
        print_err("Low  | Valid", m_vl, x_vl)

    print("\n=== Parameter Stripping Evaluation ===")
    eval_and_print_split("Kin-zero (Keep Compliance)", kin_zero_params)
    eval_and_print_split("Comp-zero (Keep Geometry)", comp_zero_params)
    
    
    
    



if __name__ == "__main__":
    main() 