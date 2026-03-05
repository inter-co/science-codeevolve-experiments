# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize  # for local SLSQP refinement

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Exploration method: Greedy largest-circle insertion by sampling random candidate centers.
    Deterministic with fixed seed for reproducibility.
    """
    np.random.seed(0)
    n = 32
    circles = np.zeros((n, 3))
    # Initialize first circle randomly
    x0, y0 = np.random.rand(), np.random.rand()
    r0 = min(x0, 1 - x0, y0, 1 - y0)
    circles[0] = [x0, y0, r0]
    M = 10000  # candidates per insertion (further increased for richer initial sampling)
    for i in range(1, n):
        # sample random candidate centers
        cand = np.random.rand(M, 2)
        # distance to square boundary
        db = np.minimum.reduce([cand[:,0], 1 - cand[:,0], cand[:,1], 1 - cand[:,1]])
        # distance from candidates to existing circles (center-to-center minus existing radius)
        ctrs = circles[:i, :2]
        rs = circles[:i, 2]
        diffs = cand[:, None, :] - ctrs[None, :, :]
        dist_ctr = np.sqrt((diffs**2).sum(axis=2)) - rs[None, :]
        dm = np.min(dist_ctr, axis=1)
        # feasible radius for each candidate
        rad_cand = np.minimum(db, dm)
        # choose the candidate with the largest feasible radius
        best = int(np.argmax(rad_cand))
        circles[i] = [cand[best, 0], cand[best, 1], rad_cand[best]]
    # Local SLSQP refinement using initial greedy solution (inspired by continuous optimization)
    x0 = circles.flatten()
    n = circles.shape[0]
    # Define bounds for variables: x_i in [0,1], y_i in [0,1], r_i in [0,0.5]
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)]
    # Build constraints: containment and non-overlap
    cons = []
    for i in range(n):
        cons += [
            {'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]},
            {'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]},
            {'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]},
            {'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]}
        ]
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (
                    np.hypot(v[3*i] - v[3*j], v[3*i+1] - v[3*j+1])
                    - (v[3*i+2] + v[3*j+2])
                )
            })
    # Objective: maximize sum of radii => minimize negative sum
    def objective(v):
        return -np.sum(v[2::3])
    # Perform local optimization
    # First SLSQP pass: more iterations and tighter tolerance
    res = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=cons,
        options={'maxiter':200, 'ftol':1e-5}
    )
    circles_refined = res.x.reshape((n, 3))

    # Multi‐start SLSQP refinements (3 random seeds) to escape local optima
    best_config = circles_refined
    best_sum = np.sum(best_config[:, 2])
    for seed in (42, 43, 44):
        x0_cont = np.zeros(n * 3)
        np.random.seed(seed)
        x0_cont[0::3] = np.random.rand(n)    # random x
        x0_cont[1::3] = np.random.rand(n)    # random y
        x0_cont[2::3] = 0.01                 # small initial radii
        res_cont = minimize(
            objective, x0_cont, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter':150, 'ftol':1e-5}
        )
        config = res_cont.x.reshape((n, 3))
        s = np.sum(config[:, 2])
        if s > best_sum:
            best_sum = s
            best_config = config
    return best_config


# EVOLVE-BLOCK-END
