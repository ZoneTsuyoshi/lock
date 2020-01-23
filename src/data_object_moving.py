import os, math
import numpy as np

def trans_generation(data, methods=["random-l"], method_changes=[], 
    T=100, v=1, noise_type="abs-normal", noise_param=[0.0, 1.0], 
    add_noise_object_on=True, xp_type="numpy"):
    """Generate translation data with lag by each point.

    Args:
        data [W^3, W^3] {xp-float}
            : Data for generation of translation data
        methods {string-list}
            : Methods for translation
            "random4": random translation for each timestep
            "random8": random translation for each timestep
            "right": right translation for each timestep
            "left": left translation for each timestep
            "up": up translation for each timestep
            "down": down translation for each timestep
            "up-right", "up-left", "down-right", "donw-left": diagonally translation for each timestep
        method_changes {int-list or numpy-int}
            : Timestep list for method change. This value should be monotolly increasing.
        W {int}
            : Width for data
        T {int}
            : Final timestep for data
        v {int}
            : Velocity of translation
        sd {float}
            : Standard deviation of Gaussian noise for each time
        xp_type {string}
            : "numpy" or "cupy"

    Returns:
        obs [T, W, W] {numpy-float}
            : Generated observation data from pseudo data
        pseudo [T,W,W] {numpy-float}
            : Generated time-lag data from true data
        true [T,W,W] {numpy-float}
            : Generated true data by translation

    Attributes:
        nums [4] {xp-float}
            : Number of translation for each direction
            0; right, 1; left, 2; up, 3; down
    """
    ## Check input arguments
    shp = data.shape
    if data.ndim != 2:
        raise ValueError("Number of axes of data must be 2. ndim={} isn't appropriate.".format(data.ndim))
    elif len(methods) > len(method_changes)+1:
        raise ValueError("Number of methods {} is larger than number of method change timesteps {}+1.".format(
            len(methods), len(method_changes)))

    xp = judge_xp_type(xp_type)
    W, H = data.shape
    

    ## Setup for attributes
    direction_list = ["right", "left", "up", "down", "up-right", "up-left", "down-right", "down-left"]
    obs = xp.zeros((T, W*H), dtype=data.dtype)
    true = xp.zeros((T, W*H), dtype=data.dtype)
    true[0] = data.reshape(-1)
    if len(method_changes)==0:
        method_chagens = [T+1]
    method_changes_diff = xp.append(xp.insert(xp.diff(method_changes), 0, method_changes[0]), T+1)
    trans_count = -1
    method_number = 0
    method = methods[0]


    ## Calculate output data
    for t in range(T-1):
        # Set method
        trans_count += 1
        if trans_count >= method_changes_diff[method_number]:
            method_number += 1
            method = methods[method_number]
            trans_count = 0

        # Set direction
        if method in direction_list:
            F = translation_matrix8(W, H, method, xp)
        elif method=="random4":
            direction = xp.random.randint(4)
            F = translation_matrix4(W, H, direction, xp)
        elif method=="random8":
            direction = xp.random.randint(8)
            F = translation_matrix8(W, H, direction, xp)

        # Calculate output data
        true[t+1] = F @ true[t]
        for vv in range(v-1):
            true[t+1] = F @ true[t+1]

    if noise_type=="normal":
        noise =  xp.random.normal(noise_param[0], noise_param[1], size=(T,W*H))
    elif noise_type=="abs-normal":
        noise = xp.absolute(xp.random.normal(noise_param[0], noise_param[1], size=(T,W*H)))
    
    obs = true + noise
    return obs.reshape(T, W, H), true.reshape(T, W, H)


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


def translation_matrix4(W=10, H=10, direction="right", cyclic=False, xp_type="numpy"):
    xp = judge_xp_type(xp_type)
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


def translation_matrix8(W=10, H=10, direction="right", cyclic=False, xp_type="numpy"):
    if direction in ["right", "left", "up", "down"]:
        F = translation_matrix4(W, H, direction, cyclic, xp_type)
    elif direction in ["up-right", "up-left", "down-right", "down-left"]:
        direction1, direction2 = direction.split("-")
        F = translation_matrix4(W, H, direction1, cyclic, xp_type) @ translation_matrix4(W, H, direction2, cyclic, xp_type)
    return F


def random_connect_image_generation(W=100, N=20, M=30, Nmin=10, Nmax=90, vmin=100, vmax=150,
    w=1, d=1, mean=5, sigma=10, mmean=0, msd=0, reverse=False, seed=123, xp_type="numpy"):
    """Generate image with space correlation

    Args:
        W {int}
            : Width of generating image data
        N {int}
            : Number of core points
        M {int}
            : Number of combination for connected line
        Nmin, Nmax {int}
            : Minimum(Maximum) value of N
        vmin, vmax {float}
            : Minimum(Maximum) value of initial image
        w {int}
            : Width of connected line
        d {int}
            : Number of space moving
        mean {float}
            : Mean of noise for all
        sd {float}
            : Standard deviation of noise for all
        mmean {float}
            : Mean of noise for molecule
        msd {float}
            : Standar deviation of noise for molecule
        seed {int}
            : Random seed
    """
    xp = judge_xp_type(xp_type)
    xp.random.seed(seed)
    result = xp.zeros((W,W))
    core_points = xp.random.randint(Nmin, Nmax, size=(N,2))
    if reverse:
        mrange = reversed(range(M))
    else:
        mrange = range(M)

    for m in mrange:
        cp1 = xp.random.randint(N)
        cp2 = xp.random.randint(N)

        while cp1==cp2:
            cp2 = xp.random.randint(N)

        x1 = min(core_points[cp1,0], core_points[cp2,0])
        x2 = max(core_points[cp1,0], core_points[cp2,0])
        y1 = min(core_points[cp1,1], core_points[cp2,1])
        y2 = max(core_points[cp1,1], core_points[cp2,1])

        if core_points[cp1,0] == core_points[cp2,0]:
            y1 = min(core_points[cp1,1], core_points[cp2,1])
            y2 = max(core_points[cp1,1], core_points[cp2,1])
            result[core_points[cp1,0], y1:y2+1] = vmin + (m+1)*(vmax-vmin)/M
        else:
            if core_points[cp1,0] < core_points[cp2,0]:
                findex = cp1
                sindex = cp2
            else:
                findex = cp2
                sindex = cp1
            x1 = core_points[findex, 0]
            x2 = core_points[sindex, 0]
            y1 = core_points[findex, 1]
            y2 = core_points[sindex, 1]
            for xx in range(x2-x1+1):
                slope = xx*(y2-y1)/(x2-x1)
                noise = xp.random.normal(mmean, msd,
                                        result[max(x1+xx+1-w,0):x1+xx+w, 
                                               max(y1+math.floor(slope)+1-w,0):y1+math.ceil(slope)+1+w].shape)
                result[max(x1+xx+1-w,0):x1+xx+w, max(y1+math.floor(slope)+1-w,0):y1+math.ceil(slope)+1+w] =\
                    vmin + (m+1)*(vmax-vmin)/M + noise

    return space_average(random_image_generation(result, mean=mean, sigma=sigma, seed=seed, 
                                                 xp_type=xp_type), d, xp_type)


def random_image_generation(points, mean=5, sigma=10, seed=123, xp_type="numpy"):
    """Generate random image with given points

    Args:
        points [WxW] {xp-array, float}
            : Point data for add random noise
        seed {int}
            : Random seed
    """
    xp = judge_xp_type(xp_type)
    xp.random.seed(seed)
    return xp.minimum(xp.absolute(points + xp.random.normal(mean, sigma, size=points.shape)), 255)


def space_average(data, d=1, xp_type="numpy"):
    """Apply space average to original data

    Args:
        data {xp-array, float}
            : original data
        d {int}
            : Number of space moving
    """
    xp = judge_xp_type(xp_type)
    result = xp.zeros(data.shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i,j] = xp.mean(data[max(i-d,0):i+d+1, max(j-d,0):j+d+1])
    
    return result



def main():
    data_root = "../data/object_moving"
    W=25
    N=15
    M=10
    Nmin=7; Nmax=17 # coordinate min, max
    vmin=100; vmax=150 # z value min, max
    w=2 # linewidth
    mean=0; sigma=0
    mmean=10; msigma=10
    d=0
    dat = random_connect_image_generation(W,N,M,Nmin,Nmax,vmin,vmax,w,d,mean,sigma,mmean,msigma)

    methods = ["right", "up", "left", "down", "right", "up", "up-right","left","down-right","up","down-left","up","down-right"]
    method_changes = [5,10,20,30,35,40, 45,55,65,75,85,95]
    T=100
    v=1
    mean=20; sd=20
    obs, true = trans_generation(dat, methods, method_changes, T=T, v=v, noise_type="abs-normal",
                                  noise_param=(mean, sd))

    np.save(os.path.join(data_root, "obs1.npy"), obs)
    np.save(os.path.join(data_root, "true1.npy"), true)

if __name__ == "__main__":
    main()

