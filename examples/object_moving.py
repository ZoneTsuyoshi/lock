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
from slock import SpatiallyUniformLOCK


def parametric_matrix_2d(w=10, h=10, d=2):
    ceild = math.ceil(d)
    floord = math.floor(d)
    A = xp.zeros((w*h, w*h), dtype=int)
    
    for i in range(w*h):
        count = 1
        for j in range(-floord, floord+1): # vertical grid
            for k in range(-floord, floord+1): # holizontal grid
                if i+j*w >=0 and i+j*w<w*h and i%w+k>=0 and i%w+k<w:
                    A[i, ((i+j*w)//w)*w + i%w+k] = count
                count += 1
        
        if type(d)!=int:
            for j in [-ceild, ceild]:
                if i%w+j>=0 and i%w+j<w:
                    A[i, (i//w)*w + i%w+j] = count
                if i+j*w>=0 and i+j*w<w*h:
                    A[i, ((i+j*w)//w)*w + i%w] = count + 1
                count += 2

    return A



def main():
    ## set root directory
    data_root = "../data/object_moving"
    save_root = "results/object_moving"
    

    # make directory
    if not os.path.exists(save_root):
        os.mkdir(save_root)


    ## set seed
    seed = 121
    xp.random.seed(seed)
    print("Set seed number {}".format(seed))


    ## set data
    Tf = 100  # length of time-series
    N = 25  # widht, height of images
    dtype = "float32"
    obs = xp.asarray(np.load(os.path.join(data_root, "obs.npy")), dtype=dtype)
    true_xp = xp.asarray(np.load(os.path.join(data_root, "true.npy")), dtype=dtype)


    ## set data for kalman filter
    skip = 1  # downsampling
    bg_value = 20  # background value
    Nf = int(N/skip)  # Number of lines after skip
    obs = obs[:Tf, ::skip, ::skip].reshape(Tf, -1)
    true_xp = true_xp[:Tf, ::skip, ::skip].reshape(Tf, -1) + bg_value

    ## set parameters
    d = 1 # number of adjacency element
    advance = True
    sigma_initial = 0 # standard deviation of normal distribution for random making
    update_interval = 1 # update interval
    eta = 1.0 # learning rate
    cutoff = 1.0 # cutoff distance for update of transition matrix
    sigma = 0.2  # standard deviation of gaussian noise
    Q = sigma**2 * xp.eye(Nf*Nf)
    R = sigma**2 * xp.eye(Nf*Nf) # Nf x nlines


    ## record list
    mse_record = xp.zeros((3, Tf))
    mae_record = xp.zeros((3, Tf))
    time_record = xp.zeros(3)

    all_start_time = time.time()

    ### Execute
    F_initial = xp.eye(Nf*Nf) # identity
    A = xp.asarray(parametric_matrix_2d(Nf, Nf, d), dtype="int32")

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

    print("SLOCK : d={}, update_interval={}, eta={}, cutoff={}".format(
        d, update_interval, eta, cutoff))
    slock = SpatiallyUniformLOCK(observation = obs, 
                     transition_matrix = F_initial,
                     transition_covariance = Q, 
                     observation_covariance = R,
                     initial_mean = obs[0], 
                     localization_matrix = A,
                     dtype = dtype,
                     update_interval = update_interval,
                     eta = eta, 
                     cutoff = cutoff,
                     save_dir = save_dir,
                     advance_mode = advance,
                     use_gpu = False)
    start_time = time.time()
    slock.forward()
    time_record[0] = time.time() - start_time
    time_record[1] = slock.times[3]
    time_record[2] = slock.times[3] / slock.times[4]
    print("SLOCK times : {}".format(time.time() - start_time))

    # record error infromation
    for t in range(Tf):
        mse_record[0,t] = mean_squared_error(
                                slock.get_filtered_value()[t],
                                true_xp[t])
        mae_record[0,t] = mean_absolute_error(
                                slock.get_filtered_value()[t],
                                true_xp[t])
        mse_record[1,t] = mean_squared_error(
                                filtered_value[t],
                                true_xp[t])
        mae_record[1,t] = mean_absolute_error(
                                filtered_value[t],
                                true_xp[t])
        mse_record[2,t] = mean_squared_error(
                                obs[t],
                                true_xp[t])
        mae_record[2,t] = mean_absolute_error(
                                obs[t],
                                true_xp[t])

    ## save error-record
    if True:
        xp.save(os.path.join(save_root, "time_record.npy"), time_record)
        xp.save(os.path.join(save_root, "mse_record.npy"), mse_record)
        xp.save(os.path.join(save_root, "mae_record.npy"), mae_record)

    # mse_record = np.load(os.path.join(save_root, "mse_record.npy"))

    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for i, label in enumerate(["SLOCK", "KF", "observation"]):
        ax.plot(np.sqrt(mse_record[i]), label=label, lw=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(fontsize=15)
    fig.savefig(os.path.join(save_root, "rmse.png"), bbox_to_inches="tight")


    ## substantial mse
    def translation_matrix4(W=10, H=10, direction="right", cyclic=False):
        F = xp.zeros((W*H, W*H))
        if direction=="right":
            F_block = xp.diag(xp.ones(W-1), -1)
            if cyclic:
                F_block[0, -1] = 1
            for i in range(H):
                F[i*W:(i+1)*W, i*W:(i+1)*W] = F_block
        elif direction=="left":
            F_block = xp.diag(xp.ones(W-1), 1)
            if cyclic:
                F_block[-1, 0] = 1
            for i in range(H):
                F[i*W:(i+1)*W, i*W:(i+1)*W] = F_block
        elif direction=="up":
            F_block = xp.eye(W)
            for i in range(H-1):
                F[i*W:(i+1)*W, (i+1)*W:(i+2)*W] = F_block
            if cyclic:
                F[(H-1)*W:H*W, 0:W] = F_block
        elif direction=="down":
            F_block = xp.eye(W)
            for i in range(H-1):
                F[(i+1)*W:(i+2)*W, i*W:(i+1)*W] = F_block
            if cyclic:
                F[0:W, (H-1)*W:H*W] = F_block
        return F

    def translation_matrix8(W=10, H=10, direction="right", cyclic=False):
        if direction in ["right", "left", "up", "down"]:
            F = translation_matrix4(W, H, direction, cyclic)
        elif direction in ["up-right", "up-left", "down-right", "down-left"]:
            direction1, direction2 = direction.split("-")
            F = translation_matrix4(W, H, direction1, cyclic) @ translation_matrix4(W, H, direction2, cyclic)
        return F

    direction_count = 0
    directions = ["right", "up", "left", "down", "right", "up", "up-right","left",
           "down-right","up","down-left","up","down-right"]
    direction_changes = [5,10,20,30,35,40,45,55,65,75,85,95,1000]
    Ftrue = translation_matrix8(Nf, Nf, directions[0])


    mean_error = np.zeros((2, Tf//update_interval-1))
    for t in range(Tf//update_interval-1):
        fvalue = np.load(os.path.join(save_dir, "transition_matrix_{:03}.npy".format(t)))

        if t+1>=direction_changes[direction_count]:
            direction_count += 1
            Ftrue = translation_matrix8(Nf, Nf, directions[direction_count], True)

        mean_error[0,t] = np.sqrt(np.power(np.absolute(fvalue - Ftrue)[A.astype(bool) & ~Ftrue.astype(bool)], 2).mean())
        mean_error[1,t] = np.sqrt(np.power(np.absolute(fvalue - Ftrue)[A.astype(bool) & Ftrue.astype(bool)], 2).mean())

    fig, ax = plt.subplots(1,1,figsize=(12,5))
    lw = 2
    for i in range(2):
        ax.plot(update_interval * np.array(range(Tf//update_interval-1)), mean_error[i], 
                   label="true={}".format(i), lw=lw)
    for mc in direction_changes[:-1]:
        ax.axvline(mc, ls="--", color="navy", lw=1)
    ax.set_xlabel("Timestep", fontsize=15)
    ax.set_ylabel("SRMSE", fontsize=15)
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    directions_for_print = ["right", " up", "    left", "    down", "right", "  up", "up-right","      left",
           " down-right","      up","  down-left","       up"," down-right"]
    fig.text(0.09, 0, "Direction: ", fontsize=10)
    for kk, direction, mc in zip(range(len(directions)), directions_for_print, [0] + direction_changes[:-1]):
        fig.text(0.16 + mc*0.0071, 0, direction, fontsize=10)
    fig.savefig(os.path.join(save_root, "srmse.png"), bbox_inches="tight")


    all_execute_time = int(time.time() - all_start_time)
    print("all time (sec): {} sec".format(all_execute_time))
    print("all time (min): {} min".format(int(all_execute_time//60)))
    print("all time (hour): {} hour + {} min".format(int(all_execute_time//3600), int((all_execute_time//60)%60)))

if __name__ == "__main__":
    main()
