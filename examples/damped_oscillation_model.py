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
from lock import LOCK
from emkf import ExpectationMaximizationKalmanFilter
from model import DampedOscillationModel, CoefficientChangedDampedOscillationModel
from scheme import EulerScheme


def main():
    save_dir = "results/damped_oscillation_model"

    # make directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set parameters
    def w(t):
        return 0
    perf_initial = xp.array([5.0, 0.0])
    sim_initial = xp.array([[6.0, 0.0]])
    m = 1.0
    k = 0.5
    r = 0.52
    dt = 1
    obs_sd = 0.2
    sys_sd = 0.01
    T = 100
    update_interval = 4
    eta = 0.6
    cutoff = 0.5
    iteration = 5
    seed = 121
    xp.random.seed(seed)

    # set matrix
    F = np.array([[1, dt], [- k * dt / m, 1 - r * dt / m]])
    G = np.array([[0], [dt / m]])
    H = np.array([[1, 0], [0, 1]])
    V0 = np.array([[1, 0], [0, 1]])
    Q = np.array([[sys_sd**2]])
    R = np.array([[obs_sd**2, 0], [0, obs_sd**2]])

    ## execute experiment 1
    print("Experiment 1")
    dom = DampedOscillationModel(m, k, r, w)
    es = EulerScheme(dt, T, dom, seed, xp_type)
    true, obs = es.noise_added_simulation(perf_initial, sys_sd, obs_sd)

    lock = LOCK(observation = obs,
                transition_matrix = F,
                observation_matrix = H,
                transition_covariance = G @ Q @ G.T,
                observation_covariance = R,
                initial_mean = sim_initial,
                initial_covariance = V0,
                update_interval = update_interval,
                eta = eta,
                cutoff = cutoff,
                store_transition_matrices_on = True,
                use_gpu = False)
    lock.forward()

    emkf = ExpectationMaximizationKalmanFilter(observation = obs,
                transition_matrices = F,
                observation_matrices = H,
                transition_covariance = G @ Q @ G.T,
                observation_covariance = R,
                initial_mean = sim_initial,
                initial_covariance = V0,
                update_interval = update_interval,
                eta = eta,
                cutoff = cutoff,
                iteration = iteration,
                store_transition_matrices_on = True,
                use_gpu = False)
    emkf.forward()

    # plot filtered result regarding final execution
    fig, axis = plt.subplots(1, 2, figsize=(15,5))
    lw = 2
    for i, value in enumerate(["x", "v"]):
        axis[i].scatter(range(T), obs[:,i], color = "k", marker = 'o', label = "obs")
        axis[i].plot(true[:,i], linestyle = '--', color = 'c', label = 'true', lw=lw)
        axis[i].plot(lock.get_filtered_value(i), color = 'r', label =  'LOCK', lw=lw)
        axis[i].plot(emkf.get_filtered_value(i), color = 'g', label =  'EMKF', lw=lw)
        axis[i].set_xlabel('Timestep', fontsize=12)
        axis[i].set_ylabel(value, fontsize=12)
    axis[0].legend(loc = 'upper right', bbox_to_anchor=(1.5, -0.15), ncol=4, fontsize=12)
    fig.savefig(os.path.join(save_dir, "ex1_states.png"), bbox_to_inches="tight")


    # number of simulation
    n_simulation = 100

    # memory transition matrices
    initial_transition_matrices = xp.zeros((n_simulation, 2, 2))
    for i in range(2):
        for j in range(2):
            initial_transition_matrices[:,i,j] = xp.random.normal(F[i,j], 1., size=n_simulation)

    ## execute experiment 2 of LOCK
    print("Experiment 2")
    fig, axis = plt.subplots(2, 2, figsize=(15,10))
    for i in range(n_simulation):
        # make observation data
        dom = DampedOscillationModel(m, k, r, w, xp_type)
        es = EulerScheme(dt, T, dom, seed, xp_type)
        true, obs = es.noise_added_simulation(perf_initial, 0, obs_sd)

        # set lock class
        lock = LOCK(observation = obs,
                    transition_matrix = initial_transition_matrices[i],
                    observation_matrix = H,
                    transition_covariance = G @ Q @ G.T,
                    observation_covariance = R,
                    initial_mean = sim_initial,
                    initial_covariance = V0,
                    update_interval = update_interval,
                    eta = eta,
                    cutoff = cutoff,
                    store_transition_matrices_on = True,
                    use_gpu = False)
        lock.forward()

        transition_matrices = xp.asarray(lock.get_transition_matrices())
        for j in range(2):
            for n in range(2):
                axis[j,n].plot(range(0, T, update_interval),
                               transition_matrices[:,j,n], 
                                c="b", ls="-")

    for i in range(2):
        for j in range(2):
            axis[i,j].axhline(F[i,j], c="r")
            axis[i,j].set_xlabel("Timestep")
            axis[i,j].set_ylabel("Value of transition matrix ({},{})".format(i, j))

    fig.savefig(os.path.join(save_dir, "ex2_lock_transition_matrices.png"), 
                bbox_to_inches="tight")


    ## execute experiment 2 of EMKF
    fig, axis = plt.subplots(2, 2, figsize=(15,10))
    for i in range(n_simulation):
        # make observation data
        dom = DampedOscillationModel(m, k, r, w, xp_type)
        es = EulerScheme(dt, T, dom, seed, xp_type)
        true, obs = es.noise_added_simulation(perf_initial, 0, obs_sd)

        # set emkf class
        emkf = ExpectationMaximizationKalmanFilter(observation = obs,
                    transition_matrices = initial_transition_matrices[i],
                    observation_matrices = H,
                    transition_covariance = G @ Q @ G.T,
                    observation_covariance = R,
                    initial_mean = sim_initial,
                    initial_covariance = V0,
                    update_interval = update_interval,
                    eta = eta,
                    cutoff = cutoff,
                    iteration = iteration,
                    store_transition_matrices_on = True,
                    use_gpu = False)
        emkf.forward()

        transition_matrices = xp.asarray(emkf.get_transition_matrices())
        for j in range(2):
            for n in range(2):
                axis[j,n].plot(range(0, T+update_interval, update_interval),
                               transition_matrices[:,j,n], 
                                c="b", ls="-")

    for i in range(2):
        for j in range(2):
            axis[i,j].axhline(F[i,j], c="r")
            axis[i,j].set_xlabel("Timestep")
            axis[i,j].set_ylabel("Value of transition matrix ({},{})".format(i, j))

    fig.savefig(os.path.join(save_dir, "ex2_emkf_transition_matrices.png"), 
                bbox_to_inches="tight")




    ### from now, we reset for experiment 3 and 4
    def make_linear_function(st, ed, T):
        def linear(t):
            return st*(1-t/T) + ed*t/T
        return linear

    k_init = 0.65
    r_init = 0.37
    k = make_linear_function(0.65, 0.35, T)
    r = make_linear_function(0.37, 0.67, T)
    F = np.array([[1, dt], [- k_init * dt / m, 1 - r_init * dt / m]])
    eta = 0.8

    ## execute experiment 3
    print("Experiment 3")
    dom = CoefficientChangedDampedOscillationModel(m, k, r, w)
    es = EulerScheme(dt, T, dom, seed, xp_type)
    true, obs = es.noise_added_simulation(perf_initial, sys_sd, obs_sd)

    lock = LOCK(observation = obs,
                transition_matrix = F,
                observation_matrix = H,
                transition_covariance = G @ Q @ G.T,
                observation_covariance = R,
                initial_mean = sim_initial,
                initial_covariance = V0,
                update_interval = update_interval,
                eta = eta,
                cutoff = cutoff,
                store_transition_matrices_on = True,
                use_gpu = False)
    lock.forward()

    emkf = ExpectationMaximizationKalmanFilter(observation = obs,
                transition_matrices = F,
                observation_matrices = H,
                transition_covariance = G @ Q @ G.T,
                observation_covariance = R,
                initial_mean = sim_initial,
                initial_covariance = V0,
                update_interval = update_interval,
                eta = eta,
                cutoff = cutoff,
                iteration = iteration,
                store_transition_matrices_on = True,
                use_gpu = False)
    emkf.forward()

    # plot filtered result regarding final execution
    fig, axis = plt.subplots(1, 2, figsize=(15,5))
    for i, value in enumerate(["x", "v"]):
        axis[i].scatter(range(T), obs[:,i], color = "k", marker = 'o', label = "obs")
        axis[i].plot(true[:,i], linestyle = '--', color = 'c', label = 'true', lw=lw)
        axis[i].plot(lock.get_filtered_value(i), color = 'r', label =  'LOCK', lw=lw)
        axis[i].plot(emkf.get_filtered_value(i), color = 'g', label =  'EMKF', lw=lw)
        axis[i].set_xlabel('Timestep', fontsize=12)
        axis[i].set_ylabel(value, fontsize=12)
    axis[0].legend(loc = 'upper right', bbox_to_anchor=(1.5, -0.15), ncol=4, fontsize=12)
    fig.savefig(os.path.join(save_dir, "ex3_states.png"), bbox_to_inches="tight")


    # memory transition matrices
    initial_transition_matrices = xp.zeros((n_simulation, 2, 2))
    for i in range(2):
        for j in range(2):
            initial_transition_matrices[:,i,j] = xp.random.normal(F[i,j], 1., size=n_simulation)


    ## execute experiment 4 of LOCK
    fig, axis = plt.subplots(2, 2, figsize=(15,10))
    true_F = np.zeros((T,2,2))
    for t in range(T):
        true_F[t] =  np.array([[1, dt], [- k(t*dt) * dt / m, 1 - r(t*dt) * dt / m]])

    for i in range(n_simulation):
        # make observation data
        dom = CoefficientChangedDampedOscillationModel(m, k, r, w, xp_type)
        es = EulerScheme(dt, T, dom, seed, xp_type)
        true, obs = es.noise_added_simulation(perf_initial, 0, obs_sd)

        # set lock class
        lock = LOCK(observation = obs,
                    transition_matrix = initial_transition_matrices[i],
                    observation_matrix = H,
                    transition_covariance = G @ Q @ G.T,
                    observation_covariance = R,
                    initial_mean = sim_initial,
                    initial_covariance = V0,
                    update_interval = update_interval,
                    eta = eta,
                    cutoff = cutoff,
                    store_transition_matrices_on = True,
                    use_gpu = False)
        lock.forward()

        transition_matrices = xp.asarray(lock.get_transition_matrices())
        for j in range(2):
            for n in range(2):
                axis[j,n].plot(range(0, T, update_interval),
                               transition_matrices[:,j,n], 
                                c="b", ls="-")

    for i in range(2):
        for j in range(2):
            axis[i,j].plot(true_F[:,i,j], c="r")
            axis[i,j].set_xlabel("Timestep")
            axis[i,j].set_ylabel("Value of transition matrix ({},{})".format(i, j))

    fig.savefig(os.path.join(save_dir, "ex4_lock_transition_matrices.png"), 
                bbox_to_inches="tight")


    ## execute experiment 4 of EMKF
    fig, axis = plt.subplots(2, 2, figsize=(15,10))
    for i in range(n_simulation):
        # make observation data
        dom = CoefficientChangedDampedOscillationModel(m, k, r, w, xp_type)
        es = EulerScheme(dt, T, dom, seed, xp_type)
        true, obs = es.noise_added_simulation(perf_initial, 0, obs_sd)

        # set emkf class
        emkf = ExpectationMaximizationKalmanFilter(observation = obs,
                    transition_matrices = initial_transition_matrices[i],
                    observation_matrices = H,
                    transition_covariance = G @ Q @ G.T,
                    observation_covariance = R,
                    initial_mean = sim_initial,
                    initial_covariance = V0,
                    update_interval = update_interval,
                    eta = eta,
                    cutoff = cutoff,
                    store_transition_matrices_on = True,
                    use_gpu = False,
                    iteration = iteration)
        emkf.forward()

        transition_matrices = xp.asarray(emkf.get_transition_matrices())
        for j in range(2):
            for n in range(2):
                axis[j,n].plot(range(0, T+update_interval, update_interval),
                               transition_matrices[:,j,n], 
                                c="b", ls="-")

    for i in range(2):
        for j in range(2):
            axis[i,j].plot(true_F[:,i,j], c="r")
            axis[i,j].set_xlabel("Timestep")
            axis[i,j].set_ylabel("Value of transition matrix ({},{})".format(i, j))

    fig.savefig(os.path.join(save_dir, "ex4_emkf_transition_matrices.png"), 
                bbox_to_inches="tight")


if __name__ == "__main__":
    main()