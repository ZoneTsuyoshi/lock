#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import sys

import numpy as np
if False:
    import cupy
    xp = cupy
    xp_type = "cupy"
else:
    xp = np
    xp_type = "numpy"

import matplotlib.pyplot as plt

sys.path.append("../src")
from sukf import SequentialUpdateKalmanFilter
from model import DampedOscillationModel
from scheme import EulerScheme


def main():
    save_dir = "sukf_result"

    # make directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set parameters
    def w(t):
        return 0
    perf_initial = xp.array([5.0, 0.0])
    sim_initial = xp.array([[6.0, 0.0]])
    m = 1.0
    k = 0.5 #0.5
    r = 0.52 #0.75
    dt = 1
    obs_sd = 0.5
    sys_sd = 0.5
    T = 100
    update_interval = 2
    eta = 0.2
    cutoff = 0.4
    seed = 121
    xp.random.seed(seed)

    # set matrix
    F = np.array([[1, dt], [- k * dt / m, 1 - r * dt / m]])
    G = np.array([[0], [dt / m]])
    H = np.array([[1, 0], [0, 1]])
    V0 = np.array([[1, 0], [0, 1]])
    Q = np.array([[sys_sd**2]])
    R = np.array([[obs_sd**2, 0], [0, obs_sd**2]])

    # number of simulation
    n_simulation = 100

    # memory transition matrices
    initial_transition_matrices = xp.zeros((n_simulation, 2, 2))
    for i in range(2):
        for j in range(2):
            initial_transition_matrices[:,i,j] = xp.random.normal(F[i,j], 1., size=n_simulation)

    ## execute simulation
    fig, axis = plt.subplots(2, 2, figsize=(15,10))
    for i in range(n_simulation):
        # make observation data
        dom = DampedOscillationModel(m, k, r, w, xp_type)
        es = EulerScheme(dt, T, dom, seed, xp_type)
        true, obs = es.noise_added_simulation(perf_initial, 0, obs_sd)

        # set sukf
        sukf = SequentialUpdateKalmanFilter(
                            observation = obs,
                            transition_matrices = initial_transition_matrices[i],
                            observation_matrices = H,
                            transition_covariance = G @ Q @ G.T,
                            observation_covariance = R,
                            initial_mean = sim_initial,
                            initial_covariance = V0,
                            update_interval = update_interval,
                            eta = eta,
                            cutoff = cutoff,
                            accumulate_transition_matrix_on = True,
                            use_gpu = False)
        sukf.forward()
        transition_matrices = xp.asarray(sukf.get_transition_matrices())
        for j in range(2):
            for n in range(2):
                axis[j,n].plot(range(0, T-update_interval, update_interval),
                               transition_matrices[:,j,n], 
                                c="b", ls="-")

    for i in range(2):
        for j in range(2):
            axis[i,j].axhline(F[i,j], c="r")
            axis[i,j].set_xlabel("Timestep")
            axis[i,j].set_ylabel("Value of transition matrix ({},{})".format(i, j))

    fig.savefig(os.path.join(save_dir, "time_change_of_transition_matrices.png"), 
                bbox_to_inches="tight")

    # plot filtered result regarding final execution
    fig, axis = plt.subplots(2, 1, figsize=(10,10))
    for i in range(2):
        axis[i].scatter(range(T), obs[:,i], c="k", marker="o", label="observation")
        axis[i].plot(true[:,i], c="c", ls="--", label="true")
        axis[i].plot(sukf.get_filtered_value(i), c="r", label="sukf")
        axis[i].legend(loc="best")
        axis[i].set_xlabel("Timestep")
        axis[i].set_ylabel("Value")

    fig.savefig(os.path.join(save_dir, "time_change_of_states.png"), 
                bbox_to_inches="tight")


if __name__ == "__main__":
    main()