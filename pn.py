from calibration.calib_models import (
    CalibrationModel6RComplNl,
)
from calibration.calib_NN import *
from calibration.calib_solvers import CalibrationSolverIpopt
from utils.utils import *
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt



# GoFa 5 params
NDOF = 6
KINVEC = [
    [0, 0, 0.167899996],
    [0, -0.060899999, 0.0970999971],
    [0, 0, 0.444000006],
    [0.112999998, 0.060899999, 0.109999999],
    [0.356999993, 0.056499999, 0],
    [0.101000004, -0.056499999, 0.0799999982],
]

# note: TCP not in RAPID convention, but x axis pointing out of the flange
tcp=[0.135, -0.09, -0.07]

#joint limits for only the planar joint 2,3 and 5
joint_limits=np.array([[-180,180],[-225,85],[0,0]])/180*np.pi


def main() -> None:
    
    validation_ratio = 0.5
    num_pts = 100 
    compliance_model = "Lin"
    learning_rate=0.005
    num_epochs=100000
    batch_size=100000


    filename_data_high = "data/1500-multidir_JT120x60_5kg_v1000.tri"
    filename_data_low = "data/1500-multidir_JT120x60_0kg_v1000.tri"
    filename_load_fct_high = "load_func_holder_5kg.pkl"
    filename_load_fct_low = "load_func_holder_only.pkl"

    with open(filename_load_fct_high, "rb") as f:
        wrench_fct_high = pickle.load(f)
    with open(filename_load_fct_low, "rb") as f:
        wrench_fct_low = pickle.load(f)


    q_all_high, tcp_all_high = parse_tri_file(filename_data_high)
    q_all_low, tcp_all_low = parse_tri_file(filename_data_low)

    q_all_high = np.array(q_all_high)[:500]
    q_all_low  = np.array(q_all_low)[:500]
    tcp_all_high = np.array(tcp_all_high)[:500]
    tcp_all_low  = np.array(tcp_all_low)[:500]


    assert q_all_high.shape == q_all_low.shape
    assert tcp_all_high.shape == tcp_all_low.shape

    split = int((1 - validation_ratio) * len(q_all_high))
    q_calib = q_all_high[split:]
    q_valid = q_all_high[:split]
    tcp_calib_high = tcp_all_high[split:]
    tcp_valid_high = tcp_all_high[:split]
    tcp_calib_low = tcp_all_low[split:]
    tcp_valid_low = tcp_all_low[:split]


    model_high = CalibrationModel6RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_high, comp_model=compliance_model)
    model_low = CalibrationModel6RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_low, comp_model=compliance_model)

    tau_calib_high = model_high.get_gravity_torque(q_calib)
    tau_calib_low = model_low.get_gravity_torque(q_calib)
    tau_valid_high = model_high.get_gravity_torque(q_valid)
    tau_valid_low = model_low.get_gravity_torque(q_valid)


    q_calib = torch.tensor(q_calib, dtype=torch.float32)
    q_valid = torch.tensor(q_valid, dtype=torch.float32)


    X_tr = torch.cat([
        q_calib,  # (N, 6)
        torch.tensor(tau_calib_low, dtype=torch.float32).squeeze(-1),  # (N,6)
        torch.tensor(tau_calib_high, dtype=torch.float32).squeeze(-1)  # (N,6)
    ], dim=1)

    X_te = torch.cat([
        q_valid,
        torch.tensor(tau_valid_low, dtype=torch.float32).squeeze(-1),
        torch.tensor(tau_valid_high, dtype=torch.float32).squeeze(-1)
    ], dim=1)

    # y = tcp_high - tcp_low
    y_tr = torch.tensor(tcp_calib_high - tcp_calib_low, dtype=torch.float32)
    y_te = torch.tensor(tcp_valid_high - tcp_valid_low, dtype=torch.float32)


    tau_all = np.concatenate([tau_calib_low, tau_calib_high, tau_valid_low, tau_valid_high], axis=0)
    tau_tr = np.max(np.abs(tau_all), axis=0).flatten()  # shape: (6,)

    #  (6 + 6 + 6 = 18)
    normalization_tr = torch.cat([
        torch.ones(6, dtype=torch.float32),
        torch.tensor(tau_tr, dtype=torch.float32),
        torch.tensor(tau_tr, dtype=torch.float32)
    ])

    X_tr = X_tr / normalization_tr
    X_te = X_te / normalization_tr  
    
    ################ STAGE 1: Recovering the Compliance model with NN #################

    

    lock = [False, True, True, True, False, False]  


    model = NNSelectiveComplianceModel(lock=lock)

    
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.data.size()) > 1:
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            else:
                nn.init.zeros_(param.data)
        else:
            nn.init.zeros_(param.data)


    scale_factor = 1e-3
    for i in range(6):
        if lock[i]:
            for layer in model.compliance_networks[i]:
                if isinstance(layer, nn.Linear):
                    layer.weight.data *= scale_factor


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    tau_max_tr = torch.max(torch.abs(X_tr[:, 6:]), dim=0).values  # shape: (12,)
    normalization_tr = torch.cat((torch.ones(6), tau_max_tr))     # shape: (18,)
    X_tr = X_tr / normalization_tr

    tau_max_te = torch.max(torch.abs(X_te[:, 6:]), dim=0).values
    normalization_te = torch.cat((torch.ones(6), tau_max_te))
    X_te = X_te / normalization_te


    loss_values = []
    loss_values_te = []

    for epoch in range(num_epochs):
        for i in range(0, len(X_tr), batch_size):
            Xbatch = torch.tensor(X_tr[i:i + batch_size, :], dtype=torch.float32)
            ybatch = torch.tensor(y_tr[i:i + batch_size, :], dtype=torch.float32)

            y_pred_tr = model(Xbatch)
            loss = 1000000 * loss_fn(y_pred_tr, ybatch)

            model.eval()
            if i < len(X_te):
                with torch.no_grad():
                    Xbatch_te = torch.tensor(X_te[i:i + batch_size, :], dtype=torch.float32)
                    ybatch_te = torch.tensor(y_te[i:i + batch_size, :], dtype=torch.float32)
                    y_pred_te = model(Xbatch_te)
                    loss_te = 1000000* loss_fn(y_pred_te, ybatch_te)
                loss_values_te.append(loss_te.item())

            model.train()
            loss_values.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Finished epoch {epoch}, train loss {loss:.4f}, test loss {loss_te:.4f}')
            
    dq_valid_high = model.get_dq(torch.tensor(tau_valid_high, dtype=torch.float32), tau_tr)
    dq_valid_low  = model.get_dq(torch.tensor(tau_valid_low,  dtype=torch.float32), tau_tr)

    # dq_valid_high = dq_valid_high.squeeze(-1)
    # dq_valid_low  = dq_valid_low.squeeze(-1)
    num_params_per_ax = 7
    init_guess = np.zeros((7 * num_params_per_ax,))
    

    q_valid_np = q_valid.detach().numpy() if isinstance(q_valid, torch.Tensor) else q_valid
    tcp_valid_high_np = tcp_valid_high.detach().numpy() if isinstance(tcp_valid_high, torch.Tensor) else tcp_valid_high
    tcp_valid_low_np  = tcp_valid_low.detach().numpy()  if isinstance(tcp_valid_low,  torch.Tensor) else tcp_valid_low

    dq_valid_high_np = dq_valid_high.detach().numpy() if isinstance(dq_valid_high, torch.Tensor) else dq_valid_high
    dq_valid_low_np  = dq_valid_low.detach().numpy()  if isinstance(dq_valid_low,  torch.Tensor) else dq_valid_low


    mean_uncal_high, max_uncal_high = model_high.get_error(q_valid_np, tcp_valid_high_np, init_guess)
    mean_uncal_low,  max_uncal_low  = model_low.get_error(q_valid_np,  tcp_valid_low_np,  init_guess)
    mean_cal_high,   max_cal_high   = model_high.get_error(q_valid_np + dq_valid_high_np, tcp_valid_high_np, init_guess) 
    mean_cal_low,    max_cal_low    = model_low.get_error(q_valid_np + dq_valid_low_np, tcp_valid_low_np,  init_guess)
    
    def print_err(tag, mean_err, max_err):
        print(f"{tag:<10s}  mean={mean_err*1000:.2f} mm   max={max_err*1000:.2f} mm")

    
    print_err("uncalib_high", mean_uncal_high, max_uncal_high)
    print_err("uncalib_low", mean_uncal_low, max_uncal_low)
    print_err("calib_high",  mean_cal_high, max_cal_high)
    print_err("calib_low",  mean_cal_low, max_cal_low)
            
    
    
    
    ####################### STAGE 2: RECOVERING THE KINEMATIC ERROR TERMS ######################
    
    # compliance_model="Lin" #The kinematic model is just set trivialy, in this step we will use the NN model instead of the regular model. The coefficient will be forced to zero and won't affect calibration
    
    # model_phase2_high= CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_high, comp_model=compliance_model,tau_tr=tau_tr)
    # model_phase2_low= CalibrationModelPlanar3RComplNl(kinvec=KINVEC, tcp=tcp, load_fct=wrench_fct_low, comp_model=compliance_model,tau_tr=tau_tr)
    # num_comp_param=1
    # err_params_zero = np.zeros((12+num_comp_param*4,)) 
    
    # err_par_switch = [
    #     [True, True, True, False],
    #     [False, True, True, False],
    #     [False, False, False, False],
    #     [False, True, False, False],
    # ]
    
    # model_phase2_high.set_error_par_switch(err_par_switch)
    # model_phase2_low.set_error_par_switch(err_par_switch)
    
    # # check error of uncalirated model
    # q_calib=q_calib.detach().numpy()
    # tcp_calib_high=tcp_calib_high.detach().numpy()
    # tcp_calib_low=tcp_calib_low.detach().numpy()
    
    # q_valid=q_valid.detach().numpy()
    # tcp_valid_high=tcp_valid_high.detach().numpy()
    # tcp_valid_low=tcp_valid_low.detach().numpy()
    
    # #Calculating the uncalibrated errors
    # mean_err_uncalib_high, max_err_uncalib_high = model_phase2_high.get_error(
    #     q=q_calib, meas=tcp_calib_high, calib_params=err_params_zero
    # ) 
    
    # mean_err_uncalib_low, max_err_uncalib_low = model_phase2_low.get_error(
    #     q=q_calib, meas=tcp_calib_low, calib_params=err_params_zero
    # ) 
    # mean_err_uncalib = (mean_err_uncalib_high+mean_err_uncalib_low)/2
    # max_err_uncalib = max(max_err_uncalib_high,max_err_uncalib_low)
    
    # #TODO: In order to incorporate the NN approach without modifying too much on the previously used structure, we will calculate the deviation due to compliance beforehand (just by passing the torque through the trained NN)
    # #The dq need to be added to the joint targets before using them in the second stage and later on in calibration/validation error computation
    
    # #This might not be the neatest way to code it so you might wan to incorporate it properly in the structure or use a new class for it
    # dq_calib_high= model.get_dq(tau_calib_high,tau_tr)
    # dq_calib_low= model.get_dq(tau_calib_low,tau_tr)

    # solver = CalibrationSolverIpopt()
    # solved_params = solver.solve_calibration_dual_2(
    #     model_high=model_phase2_high,
    #     model_low=model_phase2_low,
    #     q_high=q_calib+dq_calib_high,
    #     q_low=q_calib+dq_calib_low,
    #     meas_high=tcp_calib_high,
    #     meas_low=tcp_calib_low,
    #     initial_guess=err_params_zero,
    #     use_high_load= calib_with_high
    # )
    
    # mean_err_low, max_err_low = model_phase2_low.get_error(q=q_calib+dq_calib_low, meas=tcp_calib_low, calib_params=solved_params)
    
    # if calib_with_high:
    #     mean_err_high, max_err_high = model_phase2_high.get_error(q=q_calib+dq_calib_high, meas=tcp_calib_high, calib_params=solved_params)
    #     mean_err=(mean_err_low+mean_err_high)/2
    #     max_err=max(max_err_high,max_err_low)
    # else:
    #     mean_err=mean_err_low
    #     max_err=max_err_low
        
    # print(solved_params)
    
    # print(f"mean err (uncalib) = {mean_err_uncalib*1000:.3f} mm")
    # print(f"max err (uncalib) = {max_err_uncalib*1000:.3f} mm")
    # print(f"mean err (calib) = {mean_err*1000:.3f} mm")
    # print(f"max err (calib) = {max_err*1000:.3f} mm")
    
    # dq_valid_high= model.get_dq(tau_valid_high,tau_tr)
    # dq_valid_low= model.get_dq(tau_valid_low,tau_tr)
    
    # mean_err_low, max_err_low = model_phase2_low.get_error(q=q_valid+dq_valid_low, meas=tcp_valid_low, calib_params=solved_params)

    #  #If the high load was used in the second step, it should be used in evaluating the error and in validation
    # if calib_with_high:
    #     mean_err_high, max_err_high = model_phase2_high.get_error(q=q_valid+dq_valid_high, meas=tcp_valid_high, calib_params=solved_params)
    #     mean_err=(mean_err_low+mean_err_high)/2
    #     max_err=max(max_err_high,max_err_low)
    # else:
    #     mean_err=mean_err_low
    #     max_err=max_err_low
    
    # print(f"mean err (valid) = {mean_err*1000:.3f} mm")
    # print(f"max err (valid) = {max_err*1000:.3f} mm")
    
    
    
    
    
    
    
    
    
    #plotting the output model from the NN trained earlier
    fig, axis = plt.subplots(3, 2, figsize=(10, 8))
    model.plot_compliance_NN(axis, tau_tr)
    plt.tight_layout()
    plt.show()
    
    print("done")


if __name__ == "__main__":
    main()
