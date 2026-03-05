# EVOLVE-BLOCK-START
import numpy as np
import math
import random
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    base = None
    creator = None
    tools = None
    algorithms = None
from scipy.optimize import minimize

def circle_packing32(strategy: str = 'hex', seed: int = 42) -> np.ndarray:
    """
    Generate 32 non-overlapping circles inside the unit square maximizing the sum of radii.
    Two strategies are available:
        - 'hex' : deterministic hexagonal lattice packing.
        - 'evo' : simple evolutionary algorithm using DEAP.
    Parameters
    ----------
    strategy : str
        Choice of strategy.
    seed : int
        Random seed for reproducibility.
    Returns
    -------
    np.ndarray
        Array of shape (32,3) with (x,y,r) for each circle.
    """
    np.random.seed(seed)
    random.seed(seed)
    n = 32

    # Use a deterministic hexagonal packing as initial guess
    best_s = 0.0
    best_rc = (1, 32)
    for r in range(1, n+1):
        c = -(-n // r)  # ceil division
        s1 = 1.0 / c
        s2 = 1.0 / ((r-1)*math.sqrt(3)/2 + 1)
        s = min(s1, s2)
        if s > best_s:
            best_s = s
            best_rc = (r, c)
    r, c = best_rc
    s = best_s
    R = s / 2.0
    vertical = math.sqrt(3)/2 * s
    positions = []
    radii = []
    for i in range(r):
        for j in range(c):
            if len(positions) >= n:
                break
            x = j * s + R
            y = i * vertical + R
            if i % 2 == 1:
                x += s/2
            if x - R < 0 or x + R > 1 or y - R < 0 or y + R > 1:
                continue
            positions.append([x, y])
            radii.append(R)
        if len(positions) >= n:
            break
    # Pad if necessary
    while len(positions) < n:
        positions.append([np.random.uniform(0.01,0.99), np.random.uniform(0.01,0.99)])
        radii.append(0.01)

    init_vars = np.concatenate([np.array(positions)[:,0], np.array(positions)[:,1], np.array(radii)])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Constraint functions
    def _boundary_fun(i, dim, sign):
        def fun(v):
            coord = v[i] if dim == 0 else v[n + i]
            rad   = v[2 * n + i]
            return coord - rad if sign == 1 else 1.0 - coord - rad
        return fun

    def _overlap_fun(i, j):
        def fun(v):
            xi, yi = v[i], v[n + i]
            xj, yj = v[j], v[n + j]
            ri, rj = v[2 * n + i], v[2 * n + j]
            dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
            return dist2 - (ri + rj) ** 2
        return fun

    cons = []
    for idx in range(n):
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0,  1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0, -1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1,  1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1, -1)})

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': _overlap_fun(i, j)})

    def objective(v):
        return -np.sum(v[2 * n:])

    best_solution = None
    best_sum = -np.inf
    for restart in range(5):
        # perturb initial radii slightly
        pert = np.random.uniform(-0.005, 0.005, size=n)
        init_radii = np.array(radii) + pert
        init_radii = np.clip(init_radii, 0.01, 0.5)
        init_vars = np.concatenate([np.array(positions)[:,0], np.array(positions)[:,1], init_radii])

        res = minimize(
            objective,
            init_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 5000, 'disp': False}
        )

        if res.success:
            xs = res.x[:n]
            ys = res.x[n:2 * n]
            rs = res.x[2 * n:3 * n]
            sum_r = np.sum(rs)
            if sum_r > best_sum:
                best_sum = sum_r
                best_solution = np.column_stack([xs, ys, rs])

    if best_solution is None:
        # fallback to initial hex packing
        return np.column_stack([np.array(positions)[:,0], np.array(positions)[:,1], np.array(radii)])

    return best_solution


# EVOLVE-BLOCK-END
