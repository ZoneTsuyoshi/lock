#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import json

import numpy as np
if False:
    import cupy
    xp = cupy
else:
    xp = np
import matplotlib.pyplot as plt

sys.path.append("../src")
from util_functions import mean_squared_error, mean_absolute_error
from kalman import KalmanFilter
from llock import LocalLOCK


def adjacency_matrix_cyclic_2d(N=10, d=2):
    ceild = math.ceil(d)
    floord = math.floor(d)
    A = xp.eye(N*N, dtype=int)
    
    for i in range(N**2):
        for j in range(-floord, floord+1):
            for k in range(-floord, floord+1):
                A[i, ((i+j*N)%(N**2)//N)*N + (i+k)%N] = 1
        for j in [-ceild, ceild]:
            A[i, (i//N)*N + (i+j)%N] = 1
            A[i, ((i+j*N)%(N**2)//N)*N + i%N] = 1

    return A



def main():
    ## set root directory
    data_root = "../data/global_flow"
    save_root = "results/global_flow"
    

    # make directory
    if not os.path.exists(save_root):
        os.mkdir(save_root)


    ## set seed
    seed = 121
    xp.random.seed(seed)
    print("Set seed number {}".format(seed))


    ## set data
    Tf = 1000  # length of time-series
    N = 30  # widht, height of images
    dtype = "float32"
    obs = xp.asarray(np.load(os.path.join(data_root, "obs.npy")), dtype=dtype)
    true_xp = xp.asarray(np.load(os.path.join(data_root, "true.npy")), dtype=dtype)


    ## set data for kalman filter
    skip = 1  # downsampling
    bg_value = 20  # background value
    Nf = int(N/skip)  # Number of lines after skip
    obs = obs[:Tf, ::skip, ::skip].reshape(Tf, -1)
    true_xp = true_xp[:Tf, ::skip, ::skip].reshape(Tf, -1) + bg_value
    boundary = xp.zeros((Nf,Nf), dtype=bool)
    boundary[0] = True; boundary[-1] = True; boundary[:,0] = True; boundary[:,-1] = True
    boundary = boundary.reshape(-1)


    ## set parameters
    d = 1 # number of adjacency element
    advance = True
    sigma_initial = 0 # standard deviation of normal distribution for random making
    update_interval = 50 # update interval
    eta = 0.8 # learning rate
    cutoff = 1.0 # cutoff distance for update of transition matrix
    sigma = 0.2  # standard deviation of gaussian noise
    Q = sigma**2 * xp.eye(Nf*Nf)
    R = sigma**2 * xp.eye(Nf*Nf) # Nf x nlines


    ## record list
    mse_record = xp.zeros((2, 3, Tf))
    mae_record = xp.zeros((2, 3, Tf))
    time_record = xp.zeros(3)

    all_start_time = time.time()

    ### Execute
    F_initial = xp.eye(Nf*Nf) # identity
    A = xp.asarray(adjacency_matrix_cyclic_2d(Nf, d), dtype="int32")

    ## Kalman Filter
    filtered_value = xp.zeros((Tf, Nf*Nf))
    kf = KalmanFilter(transition_matrix = F_initial,
                         transition_covariance = Q, observation_covariance = R,
                         initial_mean = obs[0], dtype = dtype)
    for t in range(Tf):
        filtered_value[t] = kf.forward_update(t, obs[t], return_on=True)
    xp.save(os.path.join(save_root, "kf_states.npy"), filtered_value)
                            
    ## LLOCK
    save_dir = os.path.join(save_root, "llock")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print("LLOCK : d={}, update_interval={}, eta={}, cutoff={}".format(
        d, update_interval, eta, cutoff))
    llock = LocalLOCK(observation = obs, 
                     transition_matrix = F_initial,
                     transition_covariance = Q, 
                     observation_covariance = R,
                     initial_mean = obs[0], 
                     adjacency_matrix = A,
                     dtype = dtype,
                     update_interval = update_interval,
                     eta = eta, 
                     cutoff = cutoff,
                     save_dir = save_dir,
                     advance_mode = advance,
                     use_gpu = False)
    start_time = time.time()
    llock.forward()
    time_record[0] = time.time() - start_time
    time_record[1] = llock.times[3]
    time_record[2] = llock.times[3] / llock.times[4]
    print("LLOCK times : {}".format(time.time() - start_time))

    # record error infromation
    area_list = [xp.ones((Nf*Nf), dtype=bool), ~boundary]
    for r, area in enumerate(area_list):
        for t in range(Tf):
            mse_record[r,0,t] = mean_squared_error(
                                    llock.get_filtered_value()[t][area],
                                    true_xp[t][area])
            mae_record[r,0,t] = mean_absolute_error(
                                    llock.get_filtered_value()[t][area],
                                    true_xp[t][area])
            mse_record[r,1,t] = mean_squared_error(
                                    filtered_value[t][area],
                                    true_xp[t][area])
            mae_record[r,1,t] = mean_absolute_error(
                                    filtered_value[t][area],
                                    true_xp[t][area])
            mse_record[r,2,t] = mean_squared_error(
                                    obs[t][area],
                                    true_xp[t][area])
            mae_record[r,2,t] = mean_absolute_error(
                                    obs[t][area],
                                    true_xp[t][area])

    ## save error-record
    if True:
        xp.save(os.path.join(save_root, "time_record.npy"), time_record)
        xp.save(os.path.join(save_root, "mse_record.npy"), mse_record)
        xp.save(os.path.join(save_root, "mae_record.npy"), mae_record)

    # mse_record = np.load(os.path.join(save_root, "mse_record.npy"))

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for i, label in enumerate(["LLOCK", "KF", "observation"]):
        ax.plot(mse_record[0,i], label=label, lw=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.legend(fontsize=15)
    fig.savefig(os.path.join(save_root, "mse.png"), bbox_to_inches="tight")


    ## short-term prediction
    color_list = ["r", "g", "b"]
    threshold = 200
    pred_state = xp.zeros((Tf, Nf*Nf))
    pred_mse = mse_record[0].copy()

    s = threshold//update_interval
    F = np.load(os.path.join(save_dir, "transition_matrix_{:02}.npy".format(s)))
    state = np.load(os.path.join(save_dir, "states.npy"))[threshold]
    pred_state[threshold] = state.reshape(-1)
    for t in range(threshold, Tf-1):
        pred_state[t+1] = F @ pred_state[t]

    kf_state = np.load(os.path.join(save_root, "kf_states.npy"))[threshold]
    for t in range(threshold, Tf):
        pred_mse[0,t] = mean_squared_error(pred_state[t], true_xp[t])
        pred_mse[1,t] = mean_squared_error(kf_state.reshape(-1), true_xp[t])
        pred_mse[2,t] = mean_squared_error(obs[threshold], true_xp[t])

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    low = threshold-4; up=threshold+6; lw=2
    for i, label in enumerate(["LLOCK", "KF", "observation"]):
        ax.plot(range(low,up), pred_mse[i,low:up], lw=lw, ls="--", c=color_list[i])
        ax.plot(range(low,threshold+1), mse_record[0,i,low:threshold+1], label=label, lw=lw, c=color_list[i])
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=15)
    fig.savefig(os.path.join(save_root, "prediction.png"), bbox_inches="tight")


    ## substantial mse
    def translation_matrix4(W=10, H=10, direction=0, cyclic=False):
        xp = np
        F = xp.zeros((W*H, W*H))
        
        if direction==0: #right
            F_block = xp.diag(xp.ones(W-1), -1)
            if cyclic:
                F_block[0, -1] = 1
            for i in range(H):
                F[i*W:(i+1)*W, i*W:(i+1)*W] = F_block
            
        elif direction==1: #left
            F_block = xp.diag(xp.ones(W-1), 1)
            if cyclic:
                F_block[-1, 0] = 1
            for i in range(H):
                F[i*W:(i+1)*W, i*W:(i+1)*W] = F_block
        
        elif direction==2: #up
            F_block = xp.eye(W)
            for i in range(H-1):
                F[i*W:(i+1)*W, (i+1)*W:(i+2)*W] = F_block
            if cyclic:
                F[(H-1)*W:H*W, 0:W] = F_block
        
        elif direction==3: # down
            F_block = xp.eye(W)
            for i in range(H-1):
                F[(i+1)*W:(i+2)*W, i*W:(i+1)*W] = F_block
            if cyclic:
                F[0:W, (H-1)*W:H*W] = F_block
        return F

    change_interval = 250 // update_interval
    directions = [0,2,1,3]

    mean_error = np.zeros((2, Tf//update_interval-1))
    for t in range(Tf//update_interval-1):
        fvalue = np.load(os.path.join(save_dir, "transition_matrix_{:02}.npy".format(t)))
        Ftrue = translation_matrix4(Nf, Nf, directions[t//change_interval], True)
        mean_error[0,t] = np.power(np.absolute(fvalue - Ftrue)[A.astype(bool) & ~Ftrue.astype(bool)], 2).mean()
        mean_error[1,t] = np.power(np.absolute(fvalue - Ftrue)[A.astype(bool) & Ftrue.astype(bool)], 2).mean()

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for i in range(2):
        ax.plot(update_interval * np.array(range(Tf//update_interval-1)), mean_error[i], 
                   label="true={}".format(i), lw=lw)
    ax.set_xlabel("Timestep", fontsize=15)
    ax.set_ylabel("SMSE", fontsize=15)
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(save_root, "smse.png"), bbox_inches="tight")


    all_execute_time = int(time.time() - all_start_time)
    print("all time (sec): {} sec".format(all_execute_time))
    print("all time (min): {} min".format(int(all_execute_time//60)))
    print("all time (hour): {} hour + {} min".format(int(all_execute_time//3600), int((all_execute_time//60)%60)))

if __name__ == "__main__":
    main()
