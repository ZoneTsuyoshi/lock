"""
==========================================================
Inference with Parametric Sequential Update Kalman Filter
==========================================================
This module implements the Parametric Sequential Update Kalman Filter
and Kalman Smoother for Linear-Gaussian state space models
"""
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger("plsukf")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import os
import math
import time
import multiprocessing as mp
import itertools

import numpy as np

from utils import array1d, array2d
from util_functions import _parse_observations, _last_dims, \
    _determine_dimensionality


def _xi_construction(y, B):
    return y[B].sum()


class ParametricSequentialUpdateKalmanFilter(object) :
    """Implements the Parametric Sequential Update Kalman Filter.
    This class implements the PSUKF,
    for a Linear Gaussian model specified by,
    .. math::
        x_{t+1}   &= F_{t} x_{t} + b_{t} + v_{t} \\
        y_{t}     &= H_{t} x_{t} + d_{t} + w_{t} \\
        [v_{t}, w_{t}]^T &\sim N(0, [[Q_{t}, O], [O, R_{t}]])
    The PSUKF is an algorithm designed to estimate
    :math:`P(x_t | y_{0:t})` and :math:`F` in real-time. 
    As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `x_filt[t]` and covariances `V_filt`.

    Args:
        observation [n_time, n_dim_obs] {numpy-array, float}
            also known as :math:`y`. observation value
            観測値[時間軸,観測変数軸]
        initial_mean [n_dim_sys] {float} 
            also known as :math:`\mu_0`. initial state mean
            初期状態分布の期待値[状態変数軸]
        initial_covariance [n_dim_sys, n_dim_sys] {numpy-array, float} 
            also known as :math:`\Sigma_0`. initial state covariance
            初期状態分布の共分散行列[状態変数軸，状態変数軸]
        transition_matrix [n_dim_sys, n_dim_sys] 
            or [n_dim_sys, n_dim_sys]{numpy-array, float}
            also known as :math:`F`. transition matrix from x_{t-1} to x_{t}
            システムモデルの変換行列[状態変数軸，状態変数軸]
        observation_matrix [n_time, n_dim_sys, n_dim_obs] or [n_dim_sys, n_dim_obs]
             {numpy-array, float}
            also known as :math:`H`. observation matrix from x_{t} to y_{t}
            観測行列[時間軸，状態変数軸，観測変数軸] or [状態変数軸，観測変数軸]
        transition_covariance [n_time - 1, n_dim_noise, n_dim_noise]
             or [n_dim_sys, n_dim_noise]
            {numpy-array, float}
            also known as :math:`Q`. system transition covariance
            システムノイズの共分散行列[時間軸，ノイズ変数軸，ノイズ変数軸]
        observation_covariance [n_time, n_dim_obs, n_dim_obs] {numpy-array, float} 
            also known as :math:`R`. observation covariance
            観測ノイズの共分散行列[時間軸，観測変数軸，観測変数軸]
        transition_offsets [n_time, n_dim_sys] or [n_dim_sys] {numpy-array, float}
            also known as :math:`b`. transition offsets from x_{t-1} to x_{t}
        parametric_matrix [n_dim_obs, n_dim_obs] {numpy-array, float}
            also known as :math:`A`. matrix which indicates parameter number,
            same number mean same parameter. each element of the matrix corresponds to
            observation transition matrix, which controls transition from y_{t-1} to y_{t}.
        parametric_mode {str}
            mode of parametric matrix
        update_interval {int}
            interval of update transition matrix F
        eta (in (0.1))
            update rate for update transition matrix F
        cutoff
            cutoff distance for update transition matrix F
        save_dir {str, directory-like}
            directory for saving transition matrices and filtered states.
            if this variable is `None`, cannot save them.
        advance_mode {bool}
            if True, calculate transition matrix before filtering.
            if False, calculate the matrix after filtering.
        n_dim_sys {int}
            dimension of system transition variable
            システム変数の次元
        n_dim_obs {int}
            dimension of observation variable
            観測変数の次元
        dtype {type}
            data type of numpy-array
            numpy のデータ形式
        use_gpu {bool}
            wheather use gpu and cupy.
            if True, you need install package `cupy`.
            if False, set `numpy` for calculation.
        num_cpu {int} or `all`
            number of cpus duaring calculating transition matrix.
            you can set `all` or positive integer.

    Attributes:
        y : `observation`
        F : `transition_matrix`
        Q : `transition_covariance`
        H : `observation_matrix`
        R : `observation_covariance`
        x_pred [n_time+1, n_dim_sys] {numpy-array, float} 
            mean of predicted distribution
            予測分布の平均 [時間軸，状態変数軸]
        x_filt [n_time+1, n_dim_sys] {numpy-array, float}
            mean of filtered distribution
            フィルタ分布の平均 [時間軸，状態変数軸]
    """

    def __init__(self, observation = None,
                initial_mean = None, initial_covariance = None,
                transition_matrix = None, observation_matrix = None,
                transition_covariance = None, observation_covariance = None,
                transition_offsets = None,
                parametric_matrix = None,
                parametric_mode = "all",
                update_interval = 1, eta = 0.1, cutoff = 0.1, 
                save_dir = None,
                advance_mode = True,
                n_dim_sys = None, n_dim_obs = None, dtype = "float32",
                use_gpu = True, num_cpu = "all"):
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
             (initial_mean, array1d, -1),
             (initial_covariance, array2d, -2),
             (observation_matrix, array2d, -1),
             (transition_offsets, array1d, -1)],
            n_dim_sys
        )

        self.n_dim_obs = _determine_dimensionality(
            [(observation_matrix, array2d, -2),
             (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        # self.y = _parse_observations(observation)
        self.y = self.xp.asarray(observation).copy()

        if initial_mean is None:
            self.initial_mean = self.xp.zeros(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_mean = self.xp.asarray(initial_mean, dtype = dtype)
        
        if initial_covariance is None:
            self.initial_covariance = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.initial_covariance = self.xp.asarray(initial_covariance, dtype = dtype)

        if transition_matrix is None:
            self.F = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.F = self.xp.asarray(transition_matrix, dtype = dtype)

        if transition_covariance is None:
            self.Q = self.xp.eye(self.n_dim_sys, dtype = dtype)
        else:
            self.Q = self.xp.asarray(transition_covariance, dtype = dtype)

        if transition_offsets is None:
            self.b = self.xp.zeros(self.n_dim_sys, dtype=dtype)
        else:
            self.b = self.xp.asarray(transition_offsets, dtype=dtype)

        if observation_matrix is None:
            self.H = self.xp.eye(self.n_dim_obs, self.n_dim_sys, dtype = dtype)
        else:
            self.H = self.xp.asarray(observation_matrix, dtype = dtype)
        self.HI = self.xp.linalg.pinv(self.H)
        
        if observation_covariance is None:
            self.R = self.xp.eye(self.n_dim_obs, dtype = dtype)
        else:
            self.R = self.xp.asarray(observation_covariance, dtype = dtype)

        self.update_interval = int(update_interval)

        if parametric_matrix is None:
            self.A = np.eye(self.n_dim_obs, dtype=dtype)
        else:
            if self.use_gpu:
                self.A = parametric_matrix.get()
            else:
                self.A = np.asarray(parametric_matrix, dtype = int)

        self.Amax = int(self.A.max())
        self.B = np.zeros((self.n_dim_obs*self.Amax, self.n_dim_obs), dtype=bool)
        for i in range(self.n_dim_obs):
            for a in range(self.Amax):
                self.B[i*self.Amax+a] = self.A[i]==a+1
        self.B = np.tile(self.B, (self.update_interval, 1))
        self.A = self.xp.asarray(self.A, dtype=int)

        if parametric_mode in ["all", "identical"]:
            self.parametric_mode = parametric_mode

        if save_dir is None:
            self.save_change = False
        else:
            self.save_change = True
            self.save_dir = save_dir
            self.tm_count = 1
            self.fillnum = len(str(int(self.y.shape[0] / self.update_interval)))
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(0).zfill(self.fillnum) + ".npy"), self.F)

        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
        else:
            self.num_cpu = num_cpu

        self.advance_mode = advance_mode
        self.eta = eta
        self.cutoff = cutoff
        self.dtype = dtype
        # self.times = self.xp.zeros(6)


    def forward(self):
        """Calculate prediction and filter for observation times.

        Attributes:
            T {int}
                : length of data y （時系列の長さ）
            x_pred [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations
                 from times [0...t-1]
                時刻 t における状態変数の予測期待値 [時間軸，状態変数軸]
            V_pred [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t-1]
                時刻 t における状態変数の予測共分散 [時間軸，状態変数軸，状態変数軸]
            x_filt [n_time, n_dim_sys] {numpy-array, float}
                : mean of hidden state at time t given observations from times [0...t]
                時刻 t における状態変数のフィルタ期待値 [時間軸，状態変数軸]
            V_filt [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
                : covariance of hidden state at time t given observations
                 from times [0...t]
                時刻 t における状態変数のフィルタ共分散 [時間軸，状態変数軸，状態変数軸]
        """

        T = self.y.shape[0]
        self.x_pred = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        # self.V_pred = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
        #      dtype = self.dtype)
        self.x_filt = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
        # self.V_filt = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
        #      dtype = self.dtype)

        # calculate prediction and filter for every time
        for t in range(T) :
            # visualize calculating time
            print("\r filter calculating... t={}".format(t) + "/" + str(T), end="")

            if t == 0:
                # initial setting
                self.x_pred[0] = self.initial_mean
                self.V_pred = self.initial_covariance.copy()
            else:
                if self.advance_mode and t<T-self.update_interval and t%self.update_interval==0:
                    self._update_transition_matrix(t+self.update_interval-1)
                # start_time = time.time()
                self._predict_update(t)
                # self.times[0] += time.time() - start_time
            
            if self.xp.any(self.xp.isnan(self.y[t])) :
                self.x_filt[t] = self.x_pred[t]
                self.V_filt = self.V_pred.copy()
            else :
                # start_time = time.time()
                self._filter_update(t)
                # self.times[1] += time.time() - start_time
                if (not self.advance_mode) and t>0 and t%self.update_interval==0:
                    self._update_transition_matrix(t)

        if self.save_change:
            self.xp.save(os.path.join(self.save_dir, "filtered_state.npy"), self.x_filt)


    def _predict_update(self, t):
        """Calculate fileter update

        Args:
            t {int} : observation time
        """
        # extract parameters for time t-1
        F = _last_dims(self.F, t - 1, 2)
        b = _last_dims(self.b, t - 1, 1)
        Q = _last_dims(self.Q, t - 1, 2)

        # calculate predicted distribution for time t
        self.x_pred[t] = F @ self.x_filt[t-1] + b
        self.V_pred = F @ self.V_filt @ F.T + Q


    def _filter_update(self, t):
        """Calculate fileter update without noise

        Args:
            t {int} : observation time

        Attributes:
            K [n_dim_sys, n_dim_obs] {numpy-array, float}
                : Kalman gain matrix for time t [状態変数軸，観測変数軸]
                カルマンゲイン
        """
        # extract parameters for time t
        R = _last_dims(self.R, t, 2)

        # calculate filter step
        K = self.V_pred @ (
            self.H.T @ self.xp.linalg.inv(self.H @ (self.V_pred @ self.H.T) + R)
            )
        # target = self.xp.isnan(self.y[t])
        # self.y[t][target] = (H @ self.x_pred[t])[target]
        self.x_filt[t] = self.x_pred[t] + K @ (
            self.y[t] - (self.H @ self.x_pred[t])
            )
        # self.y[t][target] = (H @ self.x_filt[t])[target]
        self.V_filt = self.V_pred - K @ (self.H @ self.V_pred)


    def _update_transition_matrix(self, t):
        """Update transition matrix

        Args:
            t {int} : observation time
        """
        if self.parametric_mode=="all":
            G = self.xp.zeros((self.n_dim_obs, self.n_dim_obs), dtype=self.dtype)
        elif self.parametric_mode=="identical":
            G = self.xp.eye(self.n_dim_obs, dtype=self.dtype)

        # start_time = time.time()

        p = mp.Pool(self.num_cpu)
        if self.use_gpu:
            y = self.xp.repeat(self.y[t-self.update_interval:t], self.n_dim_obs*self.Amax, axis=0).get()
        else:
            y = self.xp.repeat(self.y[t-self.update_interval:t], self.n_dim_obs*self.Amax, axis=0)
        Xi = p.starmap(_xi_construction, zip(y,
                                            self.B))
        Xi = self.xp.asarray(Xi, dtype=self.dtype).reshape(self.n_dim_obs*self.update_interval, self.Amax)
        p.close()
        # self.times[2] += time.time() - start_time

        # start_time2 = time.time()
        if self.parametric_mode=="all":
            theta = self.xp.linalg.pinv(Xi) @ self.y[t-self.update_interval+1:t+1].reshape(-1)
        elif self.parametric_mode=="identical":
            theta = self.xp.linalg.pinv(Xi) \
                    @ (self.y[t-self.update_interval+1:t+1] - self.y[t-self.update_interval:t]).reshape(-1)

        for a in range(self.Amax):
            G[self.A==a+1] = theta[a]
        # self.times[3] += time.time() - start_time2

        # start_time2 = time.time()
        Fh = self.HI @ G @ self.H
        # self.times[4] += time.time() - start_time2

        self.F = self.F - self.eta * self.xp.minimum(self.xp.maximum(-self.cutoff, self.F - Fh), self.cutoff)

        # self.times[5] += time.time() - start_time

        if self.save_change:
            self.xp.save(os.path.join(self.save_dir, "transition_matrix_" + str(self.tm_count).zfill(self.fillnum)
                + ".npy"), self.F)
            self.tm_count += 1


    def get_predicted_value(self, dim = None):
        """Get predicted value

        Args:
            dim {int} : dimensionality for extract from predicted result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t-1]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_pred[0]
        except :
            self.filter()

        if dim is None:
            return self.x_pred
        elif dim <= self.x_pred.shape[1]:
            return self.x_pred[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_pred.shape[1] + '.')


    def get_filtered_value(self, dim = None):
        """Get filtered value

        Args:
            dim {int} : dimensionality for extract from filtered result

        Returns (numpy-array, float)
            : mean of hidden state at time t given observations
            from times [0...t]
        """
        # if not implement `filter`, implement `filter`
        try :
            self.x_filt[0]
        except :
            self.filter()

        if dim is None:
            return self.x_filt
        elif dim <= self.x_filt.shape[1]:
            return self.x_filt[:, int(dim)]
        else:
            raise ValueError('The dim must be less than '
                 + self.x_filt.shape[1] + '.')


    # def smooth(self):
    #     """Calculate RTS smooth for times.

    #     Args:
    #         T : length of data y (時系列の長さ)
    #         x_smooth [n_time, n_dim_sys] {numpy-array, float}
    #             : mean of hidden state distributions for times
    #              [0...n_times-1] given all observations
    #             時刻 t における状態変数の平滑化期待値 [時間軸，状態変数軸]
    #         V_smooth [n_time, n_dim_sys, n_dim_sys] {numpy-array, float}
    #             : covariances of hidden state distributions for times
    #              [0...n_times-1] given all observations
    #             時刻 t における状態変数の平滑化共分散 [時間軸，状態変数軸，状態変数軸]
    #         A [n_dim_sys, n_dim_sys] {numpy-array, float}
    #             : fixed interval smoothed gain
    #             固定区間平滑化ゲイン [時間軸，状態変数軸，状態変数軸]
    #     """

    #     # if not implement `filter`, implement `filter`
    #     try :
    #         self.x_pred[0]
    #     except :
    #         self.filter()

    #     T = self.y.shape[0]
    #     self.x_smooth = self.xp.zeros((T, self.n_dim_sys), dtype = self.dtype)
    #     self.V_smooth = self.xp.zeros((T, self.n_dim_sys, self.n_dim_sys),
    #          dtype = self.dtype)
    #     A = self.xp.zeros((self.n_dim_sys, self.n_dim_sys), dtype = self.dtype)

    #     self.x_smooth[-1] = self.x_filt[-1]
    #     self.V_smooth[-1] = self.V_filt[-1]

    #     # t in [0, T-2] (notice t range is reversed from 1~T)
    #     for t in reversed(range(T - 1)) :
    #         # visualize calculating times
    #         print("\r smooth calculating... t={}".format(T - t)
    #              + "/" + str(T), end="")

    #         # extract parameters for time t
    #         F = _last_dims(self.F, t, 2)

    #         # calculate fixed interval smoothing gain
    #         A = self.xp.dot(self.V_filt[t], self.xp.dot(F.T, self.xp.linalg.pinv(self.V_pred[t + 1])))
            
    #         # fixed interval smoothing
    #         self.x_smooth[t] = self.x_filt[t] \
    #             + self.xp.dot(A, self.x_smooth[t + 1] - self.x_pred[t + 1])
    #         self.V_smooth[t] = self.V_filt[t] \
    #             + self.xp.dot(A, self.xp.dot(self.V_smooth[t + 1] - self.V_pred[t + 1], A.T))

            
    # def get_smoothed_value(self, dim = None):
    #     """Get RTS smoothed value

    #     Args:
    #         dim {int} : dimensionality for extract from RTS smoothed result

    #     Returns (numpy-array, float)
    #         : mean of hidden state at time t given observations
    #         from times [0...T]
    #     """
    #     # if not implement `smooth`, implement `smooth`
    #     try :
    #         self.x_smooth[0]
    #     except :
    #         self.smooth()

    #     if dim is None:
    #         return self.x_smooth
    #     elif dim <= self.x_smooth.shape[1]:
    #         return self.x_smooth[:, int(dim)]
    #     else:
    #         raise ValueError('The dim must be less than '
    #              + self.x_smooth.shape[1] + '.')
