# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time
from scipy.spatial import KDTree
from scipy.spatial import distance_matrix, cKDTree
from scipy.optimize import linprog  # Added for LP‐based radii optimization

# Two new packing paradigms are provided:
# 1. Discrete hill-climbing local search (circle_packing_discrete)
# 2. Simulated annealing optimization (circle_packing_sa)

def circle_packing_discrete() -> np.ndarray:
    """
    Discrete local hill-climbing: iteratively adjust circle positions to maximize sum of radii.
    """
    n = 32
    # Initialize random positions
    circs = np.zeros((n,3))
    circs[:,0:2] = np.random.rand(n,2)
    # Compute initial radii function
    def compute_radii(arr):
        # boundary distances
        rad = np.minimum.reduce([arr[:,0], 1-arr[:,0], arr[:,1], 1-arr[:,1]])
        # use cKDTree for faster, vectorized neighbor queries
        tree = cKDTree(arr[:,0:2])
        # query all points at once for their two nearest neighbors
        # remove n_jobs parameter for compatibility with current SciPy API
        dists, _ = tree.query(arr[:,0:2], k=2)
        # update each radius to be at most half its nearest-neighbor distance
        rad = np.minimum(rad, dists[:,1] * 0.5)
        return rad
    circs[:,2] = compute_radii(circs)
    best = circs.copy()
    best_sum = best[:,2].sum()
    # Hill-climbing loop with adaptive step size and more iterations
    max_iter = 15000
    for t in range(max_iter):
        # gradually shrink step from 0.1 down to ~0.01
        step = 0.1 * (1 - t / max_iter) + 0.01  
        i = random.randrange(n)
        cand = best.copy()
        # random move with current step size
        cand[i,0:2] += (np.random.rand(2) - 0.5) * step
        # recompute radii to know how to project
        cand[:,2] = compute_radii(cand)
        # keep circle inside the unit square
        cand[i,0] = np.clip(cand[i,0], cand[i,2], 1 - cand[i,2])
        cand[i,1] = np.clip(cand[i,1], cand[i,2], 1 - cand[i,2])
        # finalize radii after projection
        cand[:,2] = compute_radii(cand)
        s = cand[:,2].sum()
        # accept strictly better or with a small random chance
        if s > best_sum or random.random() < 0.01:
            best = cand
            best_sum = s
    return best

def circle_packing_sa() -> np.ndarray:
    """
    Simulated annealing: optimize circle positions and radii with a penalty-based objective.
    """
    n = 32
    # state vector: [x0,y0,r0,...]
    state = np.zeros(n*3)
    # initialize positions and small radii
    for i in range(n):
        state[3*i:3*i+2] = np.random.rand(2)
        state[3*i+2] = 0.01
    def objective(s):
        xs = s[0::3]; ys = s[1::3]; rs = s[2::3]
        # boundary penalty
        pen = np.sum(np.maximum(rs - xs, 0)**2 + np.maximum(xs + rs - 1, 0)**2
                     + np.maximum(rs - ys, 0)**2 + np.maximum(ys + rs - 1, 0)**2)
        # overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                d = math.hypot(xs[i]-xs[j], ys[i]-ys[j])
                if d < rs[i] + rs[j]:
                    pen += (rs[i] + rs[j] - d)**2
        return -np.sum(rs) + 1e3 * pen
    best = state.copy()
    best_obj = objective(state)
    T0 = 1.0
    # annealing loop
    for k in range(10000):
        T = T0 * (1 - k/10000)
        idx = random.randrange(n*3)
        cand = state.copy()
        cand[idx] += (random.random() - 0.5) * 0.1
        cand[idx] = np.clip(cand[idx], 0, 1)
        val = objective(cand)
        if val < best_obj or random.random() < math.exp((best_obj - val) / max(T,1e-8)):
            state = cand
            best_obj = val
            best = cand.copy()
    xs = best[0::3]; ys = best[1::3]; rs = best[2::3]
    # enforce boundary after annealing
    rs = np.minimum.reduce([rs, xs, 1-xs, ys, 1-ys])
    return np.vstack([xs, ys, rs]).T

def circle_packing_hex() -> np.ndarray:
    """
    Hexagonal lattice + iterative expansion + hill-climbing refinement.
    """
    rows, cols = 6, 6
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) / cols
            y = (i + 0.5) / (rows + 1)
            pts.append([x, y])
    pts = np.array(pts)

    # boundary‐limited distances
    d_boundary = np.minimum.reduce([pts[:,0], 1-pts[:,0], pts[:,1], 1-pts[:,1]])
    # neighbor constraints
    D = distance_matrix(pts, pts)
    np.fill_diagonal(D, np.inf)
    d_nearest = np.min(D, axis=1) * 0.5
    radii = np.minimum(d_boundary, d_nearest)

    # pick top-32
    idx_sel = np.argsort(-radii)[:32]
    centers = pts[idx_sel]
    r_sel = radii[idx_sel].copy()

    # Optimize radii via linear programming (inspired by Inspiration 1)
    n_sel = centers.shape[0]
    # compute boundary distances for selected centers
    d_boundary_sel = np.minimum.reduce(
        [centers[:,0], 1-centers[:,0], centers[:,1], 1-centers[:,1]]
    )
    # pairwise distances among selected centers
    D_sel = distance_matrix(centers, centers)
    # build LP: maximize sum(r) -> minimize -sum(r)
    c_lp = -np.ones(n_sel)
    A_lp = []
    b_lp = []
    # boundary constraints: r_i <= d_boundary_sel[i]
    for i in range(n_sel):
        row = np.zeros(n_sel)
        row[i] = 1
        A_lp.append(row)
        b_lp.append(d_boundary_sel[i])
    # non-overlap constraints: r_i + r_j <= D_sel[i,j]
    for i in range(n_sel):
        for j in range(i+1, n_sel):
            row = np.zeros(n_sel)
            row[i] = 1
            row[j] = 1
            A_lp.append(row)
            b_lp.append(D_sel[i,j])
    A_lp = np.array(A_lp)
    b_lp = np.array(b_lp)
    res_lp = linprog(
        c_lp, A_ub=A_lp, b_ub=b_lp,
        bounds=[(0, None)] * n_sel, method='highs'
    )
    if res_lp.success:
        radi = res_lp.x
    else:
        # fallback to half the nearest‐neighbor distance
        radi = np.minimum(
            d_boundary_sel,
            0.5 * np.min(np.where(np.eye(n_sel), np.inf, D_sel), axis=1)
        )
    circles = np.hstack([centers, radi.reshape(-1,1)])
    return circles


def circle_packing32() -> np.ndarray:
    """
    Entry point: runs discrete local search, simulated annealing, and hex lattice method,
    returning the configuration with the largest sum of radii and final LP radii refinement.
    """
    cand1 = circle_packing_discrete()
    cand2 = circle_packing_sa()
    cand3 = circle_packing_hex()
    # select best initial candidate
    sums = [cand1[:,2].sum(), cand2[:,2].sum(), cand3[:,2].sum()]
    best = [cand1, cand2, cand3][int(np.argmax(sums))]
    # LP-based radii optimization on best centers
    xs = best[:, 0]; ys = best[:, 1]
    n = len(xs)
    c = -np.ones(n)
    A = []
    b = []
    # boundary constraints: r_i <= distance to nearest square edge
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(min(xs[i], 1-xs[i], ys[i], 1-ys[i]))
    # non-overlap constraints: r_i + r_j <= dist(center_i, center_j)
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A.append(row)
            b.append(math.hypot(xs[i]-xs[j], ys[i]-ys[j]))
    A = np.array(A)
    b = np.array(b)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)]*n, method='highs')
    if res.success:
        best[:,2] = res.x
    else:
        # fallback to simple boundary-based radii
        rad = np.minimum.reduce([xs, 1-xs, ys, 1-ys])
        best[:,2] = rad
    return best


# EVOLVE-BLOCK-END
