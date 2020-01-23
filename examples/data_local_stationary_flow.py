import os, time
import numpy as np


def local_flow_generation(S, B,
    noise_type="abs-normal", noise_param=[0.0, 1.0], 
    xp_type="numpy"):
    """Generate translation data with lag by each point.

    Args:
        S [T+L-1, L] {xp-float}
            : Source matrix to generate data
        B [m, n] {xp-int, xp-str}
            : Block matrix to decide direction of flow
            0 : "right"
            1 : "left"
            2 : "up"
            3 : "down"
        xp_type {string}
            : "numpy" or "cupy"
    
    Attributes:
        m, n {int}
            : Number of blocks
        W, H {int}
            : Width, Height for data, W=m*L, H=n*L
        L {int}
            : Blowk length
        T {int}
            : Final timestep for data

    Returns:
        obs [T, W, W] {numpy-float}
            : Generated observation data
        obsn [T,W,W] {numpy-float}
            : Generated observation data added molecule noise
        true [T,W,W] {numpy-float}
            : Generated true data by translation
    """
    L = S.shape[1]
    T = S.shape[0] - L + 1
    m, n = B.shape
    W = m*L; H=n*L
    
    xp = judge_xp_type(xp_type)
    true = xp.zeros((T, W, H))
    
    for i, bs in enumerate(B):
        for j, b in enumerate(bs):
            # initial field
            if b=="down": # right
                true[0, i*L:(i+1)*L, j*L:(j+1)*L] = xp.flip(S[:L], 0)
                for t in range(T-1):
                    true[t+1, i*L+1:(i+1)*L, j*L:(j+1)*L] = true[t, i*L:(i+1)*L-1, j*L:(j+1)*L]
                    true[t+1, i*L, j*L:(j+1)*L] = S[L+t]
                    
            elif b=="up": # left
                true[0, i*L:(i+1)*L, j*L:(j+1)*L] = S[:L]
                for t in range(T-1):
                    true[t+1, i*L:(i+1)*L-1, j*L:(j+1)*L] = true[t, i*L+1:(i+1)*L, j*L:(j+1)*L]
                    true[t+1, (i+1)*L-1, j*L:(j+1)*L] = S[L+t]
                    
            elif b=="left": # up
                true[0, i*L:(i+1)*L, j*L:(j+1)*L] = S[:L].T
                for t in range(T-1):
                    true[t+1, i*L:(i+1)*L, j*L:(j+1)*L-1] = true[t, i*L:(i+1)*L, j*L+1:(j+1)*L]
                    true[t+1, i*L:(i+1)*L, (j+1)*L-1] = S[L+t]
                
            elif b=="right": # down
                true[0, i*L:(i+1)*L, j*L:(j+1)*L] = xp.flip(S[:L], 0).T
                for t in range(T-1):
                    true[t+1, i*L:(i+1)*L, j*L+1:(j+1)*L] = true[t, i*L:(i+1)*L, j*L:(j+1)*L-1]
                    true[t+1, i*L:(i+1)*L, j*L] = S[L+t]


    if noise_type=="normal":
        noise =  xp.random.normal(noise_param[0], noise_param[1], size=(T,W,H))
    elif noise_type=="abs-normal":
        noise = xp.absolute(xp.random.normal(noise_param[0], noise_param[1], size=(T,W,H)))
    
    obs = true + noise
    return obs, true


def judge_xp_type(xp_type="numpy"):
    """Judge type of xp-array (numpy or cupy)

    Args:
        xp_type {string}
            : "numpy" or "cupy"

    Returns:
        numpy or cupy
    """
    if xp_type=="cupy":
        try:
            import cupy
            return cupy
        except:
            raise TypeError("CuPy cannnot be installed in this environment.")
    elif xp_type=="numpy":
        return np


def source_generation(T=100, L=15, minsize=2, maxsize=4, value_method="normal", value_param=(150, 20),
                      num_object=30, xp_type="numpy"):
    xp = judge_xp_type(xp_type)
    S = xp.zeros((T+L-1, L))
    
    for i in range(num_object):
        x = xp.random.randint(T+L-minsize)
        y = xp.random.randint(L)
        M = int(min(xp.random.randint(x+minsize, x+maxsize+1), T+L-1))
        N = int(min(xp.random.randint(y+minsize, y+maxsize+1), L))
        
        if value_method=="normal":
            S[x:M, y:N] = xp.maximum(xp.random.normal(value_param[0], value_param[1], size=(M-x, N-y)), 0)
        
    return S


def main():
    data_root = "../data/local_stationary_flow"
    np.random.seed(121)

    T = 1000
    L = 15
    minsize = 2; maxsize=4
    num_object = 1200
    mean = 20; sd = 20

    S = source_generation(T, L, minsize, maxsize, num_object=num_object)
    B = np.array([["up", "right"], ["left", "down"]])

    obs, true = local_flow_generation(S, B, noise_param=[mean, sd])

    np.save(os.path.join(data_root, "obs.npy"), obs)
    np.save(os.path.join(data_root, "true.npy"), true)


if __name__ == "__main__":
    main()