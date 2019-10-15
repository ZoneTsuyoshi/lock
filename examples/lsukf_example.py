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

sys.path.append("../src")
from util_functions import mean_squared_error, mean_absolute_error
from lsukf import LocalSequentialUpdateKalmanFilter


def adjacency_matrix_2d(w=10, h=10, d=2):
    ceild = math.ceil(d)
    floord = math.floor(d)
    A = xp.zeros((w*h, w*h), dtype=int)
    
    for i in range(w*h):
        for j in range(-floord, floord+1): # vertical grid
            for k in range(-floord, floord+1): # holizontal grid
                if i+j*w >=0 and i+j*w<w*h and i%w+k>=0 and i%w+k<w:
                    A[i, ((i+j*w)//w)*w + i%w+k] = 1
        
        if type(d)!=int:
            for j in [-ceild, ceild]:
                if i%w+j>=0 and i%w+j<w:
                    A[i, (i//w)*w + i%w+j] = 1
                if i+j*w>=0 and i+j*w<w*h:
                    A[i, ((i+j*w)//w)*w + i%w] = 1
    return A



def main():
    ## set root directory
    data_root = "../data/local_stationary_flow"
    save_root = "lsukf_result"
    

    # make directory
    if not os.path.exists(save_root):
        os.mkdir(save_root)


    ## set seed
    seed = 121
    xp.random.seed(seed)
    print("Set seed number {}".format(seed))


    ## set data
    Tafm = 1000  # length of time-series
    N = 30  # widht, height of images
    dtype = "float32"
    obs_xp = xp.asarray(np.load(os.path.join(data_root, "obs_no_object_noise.npy")))
    obsn_xp = xp.asarray(np.load(os.path.join(data_root, "obs_object_noise.npy")))
    true_xp = xp.asarray(np.load(os.path.join(data_root, "true.npy")), dtype=dtype)


    ## set data for kalman filter
    Tf = 1000  # Number of images for simulation
    skip = 1  # downsampling
    Nf = int(N/skip)  # Number of lines after skip
    obs_xp = obs_xp[:Tf, ::skip, ::skip].reshape(Tf, -1)
    obsn_xp = obsn_xp[:Tf, ::skip, ::skip].reshape(Tf, -1)
    true_xp = true_xp[:Tf, ::skip, ::skip].reshape(Tf, -1)

    ## set parameters
    obs_list = [obs_xp, obsn_xp]
    d = 1 # number of adjacency element
    advance = True
    sigma_initial = 0 # standard deviation of normal distribution for random making
    update_interval = 20 # update interval
    eta = 0.3 # learning rate
    cutoff = 1.0 # cutoff distance for update of transition matrix
    sigma = 0.2  # standard deviation of gaussian noise
    Q = sigma**2 * xp.eye(Nf*Nf)
    R = sigma**2 * xp.eye(Nf*Nf) # Nf x nlines


    ## record list
    # time_record = xp.zeros(4)
    mse_record = xp.zeros((2, len(obs_list), Tf))
    mae_record = xp.zeros((2, len(obs_list), Tf))
    time_record = xp.zeros(len(obs_list))

    all_start_time = time.time()

    ### Execute
    F_initial = xp.eye(Nf*Nf) # identity
    A = xp.asarray(adjacency_matrix_2d(Nf, Nf, d), dtype="int32")
                            
    for o, obs in enumerate(obs_list):
        ## sequential update kalman filter
        save_dir = os.path.join(save_root, "obs{}".format(o))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        print("LSUKF : obs_id={}, advance={}, d={}, update_interval={}, eta={}, cutoff={}".format(
            o, advance, d, update_interval, eta, cutoff))
        lsukf = LocalSequentialUpdateKalmanFilter(
                             observation = obs, 
                             transition_matrices = F_initial,
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
        lsukf.forward()
        time_record[o] = time.time() - start_time
        print("LSUKF times : {}".format(time.time() - start_time))

        # record error infromation
        for t in range(Tf):
            mse_record[0,o,t] = mean_squared_error(
                                    lsukf.get_filtered_value()[t],
                                    true_xp[t])
            mae_record[0,o,t] = mean_absolute_error(
                                    lsukf.get_filtered_value()[t],
                                    true_xp[t])
            mse_record[1,o,t] = mean_squared_error(
                                    lsukf.get_filtered_value()[t],
                                    obs[t])
            mae_record[1,o,t] = mean_absolute_error(
                                    lsukf.get_filtered_value()[t],
                                    obs[t])


    ## save error-record
    if True:
        xp.save(os.path.join(save_root, "time_record.npy"), time_record)
        xp.save(os.path.join(save_root, "mse_record.npy"), mse_record)
        xp.save(os.path.join(save_root, "mae_record.npy"), mae_record)

    all_execute_time = int(time.time() - all_start_time)
    print("all time (sec): {} sec".format(all_execute_time))
    print("all time (min): {} min".format(int(all_execute_time//60)))
    print("all time (hour): {} hour + {} min".format(int(all_execute_time//3600), int((all_execute_time//60)%60)))

if __name__ == "__main__":
    main()
