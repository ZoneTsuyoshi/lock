"""
=============================
Inference with Kalman Filter
=============================
This module implements the Kalman Filter
for Linear-Gaussian state space models

This code is inference for real-time calculation of Kalman Filter.
"""

import time
import math
import os
import itertools

from multiprocessing import Pool
import multiprocessing as multi

import numpy as np

from utils import array1d, array2d
from util_functions import _determine_dimensionality




class KalmanFilter(object) :
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm.
    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + G_{t} v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt[t]`.
    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | y_{0:T-1})`.
    The EM algorithm aims to find for
    :math:`\theta = (F, b, H, d, Q, R, \mu_0, \Sigma_0)`
    .. math::
        \max_{\theta} P(y_{0:T-1}; \theta)
    If we define :math:`L(x_{0:T-1},\theta) = \log P(y_{0:T-1}, x_{0:T-1};
    \theta)`, then the EM algorithm works by iteratively finding,
    .. math::
        P(x_{0:T-1} | y_{0:T-1}, \theta_i)
    then by maximizing,
    .. math::
        \theta_{i+1} = \arg\max_{\theta}
            \mathbb{E}_{x_{0:T-1}} [
                L(x_{0:T-1}, \theta)| y_{0:T-1}, \theta_i
            ]

    Args:
        initial_mean [n_dim_sys] {float} 
            : also known as :math:`\mu_0`. initial state mean
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            : also known as :math:`\Sigma_0`. initial state covariance
        transition_matrix [n_dim_sys, n_dim_sys]{numpy-array, float}
            : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
        observation_matrix [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            : also known as :math:`H`. observation matrix from x_{t} to y_{t}
        transition_covariance [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            : also known as :math:`Q`. system transition covariance for times
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            : also known as :math:`R`. observation covariance for times.
        transition_offset [n_dim_sys], {numpy-array, float} 
            : also known as :math:`b`. system offset for times.
         observation_offset [n_dim_obs] {numpy-array, float}
            : also known as :math:`d`. observation offset for times.
        n_dim_sys {int}
            : dimension of system transition variable
        n_dim_obs {int}
            : dimension of observation variable
        dtype {type}
            : data type of numpy-array
        save_path {str, path-like}
            : directory for save state and covariance.

    Attributes:
        F : `transition_matrix`
        Q : `transition_covariance`
        b : `transition_offset`
        H : `observation_matrix`
        R : `observation_covariance`
        d : `observation_offset`
    """

    def __init__(self, initial_mean = None, initial_covariance = None,
                transition_matrix = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                transition_offset = None, observation_offset = None,
                n_dim_sys = None, n_dim_obs = None,
                dtype = "float32",
                save_path = None,
                use_gpu = False):
        """Setup initial parameters.
        """
        self.use_gpu = use_gpu
        if use_gpu:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

        # determine dimensionality
        self.n_dim_sys = _determine_dimensionality(
            [(transition_matrix, array2d, -2),
             (transition_offset, array1d, -1),
             (transition_covariance, array2d, -2),
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_offset, array1d, -1),
             (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        if initial_mean is None:
            self.x = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.x = self.xp.asarray(initial_mean, dtype = dtype)
        
        if initial_covariance is None:
            self.V = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.V = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrix is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = self.xp.asarray(transition_matrix, dtype = dtype)

        if transition_covariance is None:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)

        if transition_offset is None :
            self.b = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else :
            self.b = self.xp.asarray(transition_offset, dtype = dtype)

        if observation_matrix is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = self.xp.asarray(observation_matrix, dtype = dtype)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = self.xp.asarray(observation_covariance, dtype = dtype)

        if observation_offset is None:
            self.d = self.xp.zeros(self.n_dim_obs, dtype = dtype)
        else :
            self.d = self.xp.asarray(observation_offset, dtype = dtype)

        self.dtype = dtype
        self.times = self.xp.zeros(3)

        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)



    def forward_update(self, t, y, F=None, H=None, Q=None, R=None, b=None, d=None, on_save=False,
                    save_path=None, fillnum = 3, return_on=False):
        """Calculate prediction and filter regarding arguments.

        Args:
            t {int}
                : time for calculating.
            y [n_dim_obs] {numpy-array, float}
                : also known as :math:`y`. observation value
            F [n_dim_sys, n_dim_sys]{numpy-array, float}
                : also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
            H [n_dim_sys, n_dim_obs] {numpy-array, float}
                : also known as :math:`H`. observation matrix from x_{t} to y_{t}
            Q [n_dim_sys, n_dim_noise] {numpy-array, float}
                : also known as :math:`Q`. system transition covariance for times
            R [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
                : also known as :math:`R`. observation covariance for times.
            b [n_dim_sys], {numpy-array, float} 
                : also known as :math:`b`. system offset for times.
            d [n_dim_obs] {numpy-array, float}
                : also known as :math:`d`. observation offset for times.
            on_save {boolean}
                : if true, save state x and covariance V.
            fillnum {int}
                : number of filling for zfill.

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t 
        """

        if F is None:
            F = self.F
        if H is None:
            H = self.H
        if Q is None:
            Q = self.Q
        if R is None:
            R = self.R
        if b is None:
            b = self.b
        if d is None:
            d = self.d
        if save_path is None and on_save:
            save_path = self.save_path

        # calculate predicted distribution for time t
        if t != 0:
            self.x = F @ self.x + b
            self.V = F @ self.V @ F.T + Q

            if on_save:
                self.xp.save(os.path.join(save_path, "predictive_mean_"
                                                    + str(t).zfill(fillnum) + ".npy"),
                        self.x)
                self.xp.save(os.path.join(save_path, "predictive_covariance_"
                                                    + str(t).zfill(fillnum) + ".npy"),
                        self.V)

        K = self.V @ ( H.T @ self.xp.linalg.inv(H @ (self.V @ H.T) + R) )
        self.x = self.x + K @ ( y - (H @ self.x + d) )
        self.V = self.V - K @ (H @ self.V)

        if on_save:
            self.xp.save(os.path.join(save_path, "filtered_mean_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.x)
            self.xp.save(os.path.join(save_path, "filtered_covariance_"
                                            + str(t).zfill(fillnum) + ".npy"),
                    self.V)

        if return_on:
            return self.x
