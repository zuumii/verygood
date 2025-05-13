#!/usr/bin/env python3
from calibration.calib_models import CalibrationModel6RComplNl
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np

import torch, torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os

import math
import torch
import torch.nn.functional as F

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
    max_samples = 1000
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
    
    
    # ========= 1) 定义两套开关 =========
    err_par_switch_comp_only = [       # “几何锁死、只标柔顺”
        [False, False, False, False, False, False, False],  # J1
        [False, False, False, False, False, False,  True],  # J2
        [False, False, False, False, False, False,  True],  # J3
        [False, False, False, False, False, False,  True],  # J4
        [False, False, False, False, False, False,  False],  # J5
        [False, False, False, False, False, False, False],  # J6
        [False, False, False, False, False, False, False],  # TCP
    ]

    err_par_switch_geom_only = [       # “柔顺锁死、只标几何”
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


    print_err("High | Uncalibrated", mean_uncal_high, max_uncal_high)
    print_err("High | Calibrated", mean_cal_high, max_cal_high)
    print_err("High | Validation", mean_val_high, max_val_high)


    print_err("Low  | Uncalibrated", mean_uncal_low, max_uncal_low)
    print_err("Low  | Calibrated", mean_cal_low, max_cal_low)
    print_err("Low  | Validation", mean_val_low, max_val_low)


    print_err("High (in Low's Paras) | Calib", mean_cal_high_in_low_paras, max_cal_high_in_low_paras)
    print_err("High (in Low's Paras) | Valid", mean_val_high_in_low_paras, max_val_high_in_low_paras)

    print_err("Low  (in High's Paras) | Calib", mean_cal_low_in_high_paras, max_cal_low_in_high_paras)
    print_err("Low  (in High's Paras) | Valid", mean_val_low_in_high_paras, max_val_low_in_high_paras)
    
    

    stride = 6 + num_comp_param         
    print("\n=== Compliance parameters per joint ===")
    print("          High-load                     Low-load")
    for j in range(7):                   
        C_high = solved_params_high[j*stride+6 : (j+1)*stride]
        C_low  = solved_params_low [j*stride+6 : (j+1)*stride]
        print(f"Joint {j+1}: {C_high}   {C_low}")
        
        

    def zero_compliance(params: np.ndarray, num_comp_param: int) -> np.ndarray:
        
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
        

    

    # -------- plt compliance curves ---------
    # tau_tr = np.max(np.abs(model.get_gravity_torque(q_calib)), axis=0)
    # fig, axarr = plt.subplots(3, 2, figsize=(10, 12))
    # plt.subplots_adjust(hspace=0.4)
    # model.plot_compliance(axarr, solved_params, tau_tr, err_par_switch)
    # plt.show()


  # ============================================================

    # ============================================================
    init_guess = np.zeros((7 * num_params_per_ax,))

    # ============================================================

    # ============================================================
    solver = CalibrationSolverIpopt()

    q_shared = q_calib_high                                # (N,6)
    tau_high = model_high.get_gravity_torque(q_shared)     # (N,6)
    tau_low  = model_low .get_gravity_torque(q_shared)

    params_compl = solver.solve_compliance_alignment(
        model_high    = model_high,
        model_low     = model_low,
        q_shared      = q_shared,
        tau_high      = tau_high,
        tau_low       = tau_low,
        meas_high     = tcp_calib_high,
        meas_low      = tcp_calib_low,
        comp_model    = compliance_model,                  # "Lin"/"Quad"/"Cubic"
        initial_guess = init_guess
    )

    # ============================================================

    # ============================================================
    def print_err(tag, mean_err, max_err):
        print(f"{tag:<20s}  mean={mean_err*1000:.2f} mm   max={max_err*1000:.2f} mm")

    mean_high_cal, max_high_cal = model_high.get_error(q_calib_high, tcp_calib_high, params_compl)
    mean_high_val, max_high_val = model_high.get_error(q_valid_high, tcp_valid_high, params_compl)
    mean_low_cal , max_low_cal  = model_low .get_error(q_calib_low , tcp_calib_low , params_compl)
    mean_low_val , max_low_val  = model_low .get_error(q_valid_low , tcp_valid_low , params_compl)

    print("\n=== Compliance-Alignment Result ===")
    print_err("High | Calib", mean_high_cal, max_high_cal)
    print_err("High | Valid", mean_high_val, max_high_val)
    print_err("Low  | Calib", mean_low_cal , max_low_cal )
    print_err("Low  | Valid", mean_low_val , max_low_val )

    # ============================================================

    # ============================================================
    stride = 6 + num_comp_param
    print("\n=== Learned Compliance parameters (C*) ===")
    for j in range(7):
        Cj = params_compl[j*stride+6 : (j+1)*stride]
        print(f"Joint {j+1}: {Cj}")

    # ============================================================

    # ============================================================
    tau_valid_high = model_high.get_gravity_torque(q_valid_high)
    tau_valid_low  = model_low .get_gravity_torque(q_valid_low)

    dq_valid_high = model_high.get_dq_from_tau(tau_valid_high, params_compl)
    dq_valid_low  = model_low .get_dq_from_tau(tau_valid_low , params_compl)

    def print_dq_per_joint(tag, dq):
        dq_deg = dq * 180 / np.pi
        print(f"{tag}:")
        for j in range(6):
            m = np.mean(np.abs(dq_deg[:, j]))
            M = np.max (np.abs(dq_deg[:, j]))
            print(f"  Joint {j+1}: mean Δq = {m:.4f} deg   max Δq = {M:.4f} deg")

    print("\n=== Δq Statistics after compliance alignment ===")
    print_dq_per_joint("High-valid", dq_valid_high)
    print_dq_per_joint("Low-valid ", dq_valid_low)

    # ============================================================

    # ============================================================
    print("\n=== High-Low TCP distance BEFORE vs AFTER compliance ===")


    dist_raw = np.linalg.norm(tcp_valid_high - tcp_valid_low, axis=1)
    print(f"Raw  distance   : mean = {dist_raw.mean()*1000:.2f} mm   "
        f"max = {dist_raw.max()*1000:.2f} mm")


    q_corr_high = q_valid_high + dq_valid_high
    q_corr_low  = q_valid_low  + dq_valid_low

    params_zero = np.zeros_like(params_compl)
    fk_high = model_high.get_symbolic_meas_fct()
    fk_low  = model_low .get_symbolic_meas_fct()

    tcp_corr_high = np.vstack([
        fk_high(qi, params_zero).full().ravel() for qi in q_corr_high
    ])
    tcp_corr_low  = np.vstack([
        fk_low (qi, params_zero).full().ravel() for qi in q_corr_low
    ])

    dist_after = np.linalg.norm(tcp_corr_high - tcp_corr_low, axis=1)
    print(f"After compliance: mean = {dist_after.mean()*1000:.2f} mm   "
        f"max = {dist_after.max()*1000:.2f} mm")
    
    
    



    tcp_corr_high = np.vstack([fk_high(qi, params_compl).full().ravel()
                                for qi in q_valid_high])
    tcp_corr_low  = np.vstack([fk_low (qi, params_compl).full().ravel()
                                for qi in q_valid_low])


    delta_pred = tcp_corr_high - tcp_corr_low
    delta_meas = tcp_valid_high - tcp_valid_low
    residual   = np.linalg.norm(delta_pred - delta_meas, axis=1)

    print(f"Residual after compliance (should → 0):  "
        f"mean = {residual.mean()*1000:.2f} mm   "
        f"max = {residual.max()*1000:.2f} mm")
    
    
    
    tcp_full_high = np.vstack([fk_high(qi, solved_params_high).full().ravel()
                                for qi in q_valid_high])
    tcp_full_low  = np.vstack([fk_low (qi, solved_params_low).full().ravel()
                                for qi in q_valid_low])


    delta_full_pred = tcp_full_high - tcp_full_low
    delta_meas = tcp_valid_high - tcp_valid_low
    full_residual   = np.linalg.norm(delta_full_pred - delta_meas, axis=1)

    print(f"full_Residual after compliance (should → 0):  "
        f"mean = {full_residual .mean()*1000:.2f} mm   "
        f"max = {full_residual .max()*1000:.2f} mm")
    
    
    
    
    params_zero = np.zeros_like(params_compl)
    fk_high = model_high.get_symbolic_meas_fct()
    fk_low  = model_low .get_symbolic_meas_fct()

    tcp_corr_high = np.vstack([
        fk_high(qi, params_zero).full().ravel() for qi in q_corr_high
    ])
    tcp_corr_low  = np.vstack([
        fk_low (qi, params_zero).full().ravel() for qi in q_corr_low
    ])

    dist_after = np.linalg.norm(tcp_corr_high - tcp_corr_low, axis=1)
    print(f"After compliance: mean = {dist_after.mean()*1000:.2f} mm   "
        f"max = {dist_after.max()*1000:.2f} mm")
    
    
    
######################################################################################################################################    
    
    
    print("\n=== NN Calibration Evaluation ===")
    

    dq_calib_high = model_high.get_dq_from_tau(tau_calib_high, params_compl)
    dq_valid_high = model_high.get_dq_from_tau(tau_valid_high, params_compl)


    q_corr_calib = q_calib_high + dq_calib_high       # shape (N,6)
    q_corr_valid = q_valid_high + dq_valid_high

    
        

    
    
##########################################################################

    class _HTMCell(nn.Module):
        def __init__(self, a_nom, d_nom):
            super().__init__()
            self.fc = nn.Linear(2, 9, bias=True)
            self.a  = nn.Parameter(torch.tensor([a_nom], dtype=torch.float32))
            self.d  = nn.Parameter(torch.tensor([d_nom], dtype=torch.float32))
            nn.init.zeros_(self.fc.bias)
            nn.init.normal_(self.fc.weight, 0, 1e-3)

        def forward(self, th):
            feat = torch.cat([torch.sin(th), torch.cos(th)], -1)
            R0 = self.fc(feat).view(-1, 3, 3)
            U, _, Vt = torch.linalg.svd(R0)
            R = U @ Vt
            det = torch.det(R)
            R[det < 0, :, 2] *= -1  

            sinθ, cosθ = torch.sin(th), torch.cos(th)
            t = torch.stack([self.a * cosθ, self.a * sinθ, self.d.expand_as(th)], -1).squeeze(-2)

            B = th.size(0)
            T = torch.eye(4, device=th.device).repeat(B, 1, 1)
            T[:, :3, :3] = R
            T[:, :3, 3]  = t
            return T

    class _ArmHTMNet(nn.Module):
        def __init__(self, dh_nom, tcp_init=[0.135, -0.09, -0.07]):
            super().__init__()
            self.cells = nn.ModuleList([_HTMCell(*dh) for dh in dh_nom])
            self.tcp = nn.Parameter(torch.tensor(tcp_init, dtype=torch.float32))

        def forward(self, q):
            B = q.size(0)
            T = torch.eye(4, device=q.device).repeat(B, 1, 1)
            for j in range(6):
                T = T @ self.cells[j](q[:, j:j+1])
            T[:, :3, 3] += self.tcp 
            return T[:, :3, 3]


    dh_nom = [
        (0.0,   0.1679),
        (0.0,   0.0971),
        (0.0,   0.4440),
        (0.113, 0.11),
        (0.357, 0.0),
        (0.101, 0.08)
    ]

    # ---------- HTM‑Net with LR Scheduler & Early Stopping ----------
    q_train = torch.tensor(q_corr_calib, dtype=torch.float32)
    p_train = torch.tensor(tcp_calib_high, dtype=torch.float32)
    q_val   = torch.tensor(q_corr_valid, dtype=torch.float32)
    p_val   = torch.tensor(tcp_valid_high, dtype=torch.float32)

    net = _ArmHTMNet(dh_nom)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, min_lr=1e-5, verbose=True
    )


    early_stop_patience = 5000    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    EPOCH = 30000
    for ep in range(EPOCH):
        net.train()
        loss = 1000*loss_fn(net(q_train), p_train)
        opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 100 == 0 or ep == 0:
            net.eval()
            val_l = 1000*loss_fn(net(q_val), p_val).item()
            scheduler.step(val_l) 
            print(f"Epoch {ep+1:5d}  train={loss.item():.6f}  val={val_l:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

            # Early Stopping
            if val_l < best_val_loss - 1e-6:  # improvement threshold
                best_val_loss = val_l
                epochs_no_improve = 0
            else:
                epochs_no_improve += 100

            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered at Epoch {ep+1}. Best Val Loss: {best_val_loss:.6f}")
                break


    with torch.no_grad():
        pred = net(q_val).cpu().numpy()
    err = np.linalg.norm(pred - p_val.cpu().numpy(), axis=1)
    print(f"\nAfter HTM‑Net  mean = {err.mean()*1000:.2f} mm   max = {err.max()*1000:.2f} mm")
######################################################################
 
    ##############################################################
# ---------------------------------------------------------------

# ---------------------------------------------------------------


    class _HTMCell(nn.Module):
        def __init__(self, a_nom, d_nom, b_nom=0.0):
            super().__init__()

            self.fc = nn.Linear(2, 9, bias=True)
            nn.init.zeros_(self.fc.bias)
            nn.init.normal_(self.fc.weight, 0, 1e-3)

            self.a = nn.Parameter(torch.tensor([a_nom], dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor([d_nom], dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor([b_nom], dtype=torch.float32))  

        def forward(self, th):                       # th: (B,1)

            feat = torch.cat([torch.sin(th), torch.cos(th)], -1)        # (B,2)
            R0   = self.fc(feat).view(-1, 3, 3)                         # (B,3,3)
            U, _, Vt = torch.linalg.svd(R0)
            R = U @ Vt
            det = torch.det(R);  R[det < 0, :, 2] *= -1                 # det(R)=+1

   
            sinθ, cosθ = torch.sin(th), torch.cos(th)
            x = self.a * cosθ
            y = self.a * sinθ + self.b.expand_as(th)                  
            z = self.d.expand_as(th)
            t = torch.stack([x, y, z], -1).squeeze(-2)                  # (B,3)

            B = th.size(0)
            T = torch.eye(4, device=th.device).repeat(B, 1, 1)
            T[:, :3, :3] = R
            T[:, :3, 3]  = t
            return T

    class _ArmHTMNet(nn.Module):
        def __init__(self, dh_nom, tcp_init=[0.135, -0.09, -0.07]):
            super().__init__()

            self.cells = nn.ModuleList([_HTMCell(*dh) for dh in dh_nom])
            self.tcp   = nn.Parameter(torch.tensor(tcp_init, dtype=torch.float32))

        def forward(self, q):                        # q: (B,6)
            B = q.size(0)
            T = torch.eye(4, device=q.device).repeat(B, 1, 1)
            for j in range(6):
                T = T @ self.cells[j](q[:, j:j+1])
            T[:, :3, 3] += self.tcp                 
            return T[:, :3, 3]

    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    dh_nom = [
        (0.0,   0.1679),
        (0.0,   0.0971),
        (0.0,   0.4440),
        (0.113, 0.11),
        (0.357, 0.0),
        (0.101, 0.08)
    ]

    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    q_train = torch.tensor(q_corr_calib, dtype=torch.float32)
    p_train = torch.tensor(tcp_calib_high, dtype=torch.float32)
    q_val   = torch.tensor(q_corr_valid, dtype=torch.float32)
    p_val   = torch.tensor(tcp_valid_high, dtype=torch.float32)

    net = _ArmHTMNet(dh_nom)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=500, min_lr=1e-6, verbose=True
    )

    early_stop_patience = 3000
    best_val_loss, epochs_no_improve = float('inf'), 0

    EPOCH = 30000
    for ep in range(EPOCH):
        net.train()
        loss = loss_fn(net(q_train), p_train)
        opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 100 == 0 or ep == 0:
            net.eval()
            val_l = loss_fn(net(q_val), p_val).item()
            scheduler.step(val_l)
            print(f"Epoch {ep+1:5d}  train={loss.item():.6f}  val={val_l:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

            if val_l < best_val_loss - 1e-6:
                best_val_loss = val_l
                epochs_no_improve = 0
            else:
                epochs_no_improve += 100
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered at Epoch {ep+1}. Best Val Loss: {best_val_loss:.6f}")
                break

    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    with torch.no_grad():
        pred = net(q_val).cpu().numpy()
    err = np.linalg.norm(pred - p_val.cpu().numpy(), axis=1)
    print(f"\nAfter HTM-Net  mean = {err.mean()*1000:.2f} mm   max = {err.max()*1000:.2f} mm")
    
    
    ###############################################################
    # ---------- dq ----------
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """四元数哈密顿积 q1*q2，q = (w,x,y,z)，形状 (...,4)。"""
        w1,x1,y1,z1 = q1.unbind(-1)
        w2,x2,y2,z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    # ---------- DQ-Cell ----------
    class _DQCell(nn.Module):

        def __init__(self, a_nom: float, d_nom: float):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([a_nom], dtype=torch.float32))
            self.d = nn.Parameter(torch.tensor([d_nom], dtype=torch.float32))
            self.fc = nn.Linear(2, 4, bias=True)    
            nn.init.zeros_(self.fc.bias)
            with torch.no_grad():                   
                self.fc.bias[0] = 1.0

        def forward(self, theta: torch.Tensor) -> torch.Tensor:   # theta: [B,1]
            feats = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)  # [B,2]
            q_raw = self.fc(feats)                               # [B,4]
            q_r   = F.normalize(q_raw, dim=-1)            

            sinθ, cosθ = torch.sin(theta), torch.cos(theta)
            t = torch.stack([self.a * cosθ,
                            self.a * sinθ,
                            self.d.expand_as(theta)], -1).squeeze(-2)       # [B,3]
            t_quat = torch.cat([theta.new_zeros(theta.size(0),1), t], dim=-1)# (0,t)

            #  q_d = 0.5 * t_quat * q_r
            q_d = 0.5 * quat_mul(t_quat, q_r)
            return torch.cat([q_r, q_d], dim=-1)                  # [B,8]

    # ----------  ----------
    class _ArmDQNet(nn.Module):
        def __init__(self, dh_nom, tcp_init=[0.135, -0.09, -0.07]):
            """
            dh_nom: 列表，每项为 (a_nom, d_nom)，alpha 视为 0
            """
            super().__init__()
            self.cells = nn.ModuleList([_DQCell(*dh) for dh in dh_nom])
            self.tcp = nn.Parameter(torch.tensor(tcp_init, dtype=torch.float32))

        def forward(self, q):
            """
            q: [B,6] 关节角 → 输出 TCP 位置 [B,3]
            """
            B = q.size(0)
            # (1 + ε0)
            dq_total = q.new_tensor([1.,0.,0.,0., 0.,0.,0.,0.]).expand(B,8).clone()

            for j, cell in enumerate(self.cells):
                dq_j = cell(q[:, j:j+1])           # [B,8]
                q1, q1e = dq_total[...,:4], dq_total[...,4:]
                q2, q2e = dq_j[...,:4], dq_j[...,4:]
                q_new  = quat_mul(q1, q2)
                q_eNew = quat_mul(q1, q2e) + quat_mul(q1e, q2)
                dq_total = torch.cat([q_new, q_eNew], dim=-1)

            q_rot   = dq_total[...,:4]
            q_trans = dq_total[...,4:]
            q_rot_conj = torch.cat([q_rot[...,:1], -q_rot[...,1:]], dim=-1)
            trans_quat = 2 * quat_mul(q_trans, q_rot_conj)        # (0, tx, ty, tz)
            tcp_pos = trans_quat[...,1:] + self.tcp               # [B,3]
            return tcp_pos

    # ----------  ----------
    dh_nom = [
        (0.0,   0.1679),
        (0.0,   0.0971),
        (0.0,   0.4440),
        (0.113, 0.11),
        (0.357, 0.0),
        (0.101, 0.08)
    ]


    q_train = torch.tensor(q_corr_calib, dtype=torch.float32)
    p_train = torch.tensor(tcp_calib_high, dtype=torch.float32)
    q_val   = torch.tensor(q_corr_valid, dtype=torch.float32)
    p_val   = torch.tensor(tcp_valid_high, dtype=torch.float32)

    net = _ArmDQNet(dh_nom)               
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, min_lr=1e-5, verbose=True
    )

    early_stop_patience = 5000
    best_val_loss = float('inf')
    epochs_no_improve = 0

    EPOCH = 30000
    for ep in range(EPOCH):
        net.train()
        loss = 1000 * loss_fn(net(q_train), p_train)
        opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 100 == 0 or ep == 0:
            net.eval()
            val_l = 1000 * loss_fn(net(q_val), p_val).item()
            scheduler.step(val_l)
            print(f"Epoch {ep+1:5d}  train={loss.item():.6f}  val={val_l:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

            if val_l < best_val_loss - 1e-6:
                best_val_loss = val_l
                epochs_no_improve = 0
            else:
                epochs_no_improve += 100
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered at Epoch {ep+1}. Best Val Loss: {best_val_loss:.6f}")
                break


    with torch.no_grad():
        pred = net(q_val).cpu().numpy()
    err = np.linalg.norm(pred - p_val.cpu().numpy(), axis=1)
    print(f"\nAfter DQ-Net  mean = {err.mean()*1000:.2f} mm   max = {err.max()*1000:.2f} mm")
    



if __name__ == "__main__":
    main() 