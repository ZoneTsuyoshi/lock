import os, time
import numpy as np


def global_flow_generation(S, T, move, move_change,
    noise_type="abs-normal", noise_param=[0.0, 1.0], v=1,
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
    L = S.shape[0]
    move_count = 0
    
    xp = judge_xp_type(xp_type)
    true = xp.zeros((T, L*L))
    true[0] = S.reshape(-1)
    F = translation_matrix4(L, L, move[0])
    
    for t in range(T-1):
        if t==move_change[move_count]:
            move_count += 1
            F = translation_matrix4(L, L, move[move_count])
        
        true[t+1] = F @ true[t]
    
    true = true.reshape(T, L, L)


    if noise_type=="normal":
        noise =  xp.random.normal(noise_param[0], noise_param[1], size=(T,L,L))
    elif noise_type=="abs-normal":
        noise = xp.absolute(xp.random.normal(noise_param[0], noise_param[1], size=(T,L,L)))
    
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


def translation_matrix4(W=10, H=10, direction="right", xp=np):
    F = xp.zeros((W*H, W*H))
    
    if direction=="right":
        F_block = xp.diag(xp.ones(W-1), -1)
        F_block[0, -1] = 1
        for i in range(H):
            F[i*W:(i+1)*W, i*W:(i+1)*W] = F_block
        
    elif direction=="left":
        F_block = xp.diag(xp.ones(W-1), 1)
        F_block[-1, 0] = 1
        for i in range(H):
            F[i*W:(i+1)*W, i*W:(i+1)*W] = F_block
    
    elif direction=="up":
        F_block = xp.eye(W)
        for i in range(H-1):
            F[i*W:(i+1)*W, (i+1)*W:(i+2)*W] = F_block
        F[(H-1)*W:H*W, 0:W] = F_block
    
    elif direction=="down":
        F_block = xp.eye(W)
        for i in range(H-1):
            F[(i+1)*W:(i+2)*W, i*W:(i+1)*W] = F_block
        F[0:W, (H-1)*W:H*W] = F_block
    
    return F


def source_generation(L=15, minsize=2, maxsize=4, value_method="normal", value_param=(150, 20),
                      num_object=30, xp_type="numpy"):
    xp = judge_xp_type(xp_type)
    S = xp.zeros((L, L))
    
    for i in range(num_object):
        x = xp.random.randint(L)
        y = xp.random.randint(L)
        M = int(min(xp.random.randint(x+minsize, x+maxsize+1), L))
        N = int(min(xp.random.randint(y+minsize, y+maxsize+1), L))
        
        if value_method=="normal":
            S[x:M, y:N] = xp.maximum(xp.random.normal(value_param[0], value_param[1], size=(M-x, N-y)), 0)
        
    return S


def main():
    data_root = "../data/global_flow"
    np.random.seed(121)

    T = 1000
    L = 30
    minsize = 2; maxsize=4
    num_object = 60
    mean = 20; sd = 20
    S = source_generation(L, minsize, maxsize, num_object=num_object)
    move = ["right", "up", "left", "down"]
    move_change = [250, 500, 750, 10000]

    obs, true = global_flow_generation(S, T, move, move_change, noise_param=[mean, sd], v=1)

    np.save(os.path.join(data_root, "obs.npy"), obs)
    np.save(os.path.join(data_root, "true.npy"), true)


if __name__ == "__main__":
    main()