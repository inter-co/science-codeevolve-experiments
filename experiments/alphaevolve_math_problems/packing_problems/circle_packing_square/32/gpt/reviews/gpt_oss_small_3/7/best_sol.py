# EVOLVE-BLOCK-START
# NEW IMPLEMENTATION: Hybrid deterministic + stochastic optimization
import numpy as np
from scipy.optimize import minimize
import sys

# Fixed random seed for reproducibility
_rng = np.random.default_rng(42)

def _hex_initialization(n: int):
    """Generate positions and a common radius using a hexagonal lattice."""
    import math
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        rows = int(math.floor((1 - 2 * mid) / (math.sqrt(3) * mid))) + 1
        cols = int(math.floor((1 - 2 * mid) / (2 * mid))) + 1
        if rows * cols >= n:
            lo = mid
        else:
            hi = mid
    r = lo
    circles = np.zeros((n, 3))
    count = 0
    y = r
    row = 0
    while count < n and y <= 1 - r + 1e-12:
        offset = r if row % 2 == 0 else 2 * r
        x = offset
        while count < n and x <= 1 - r + 1e-12:
            circles[count] = [x, y, r]
            count += 1
            x += 2 * r
        y += math.sqrt(3) * r
        row += 1
    while count < n:
        circles[count] = [r, r, r]
        count += 1
    return circles[:, :2], circles[:, 2]

def _grid_initialization(n: int):
    """6×6 grid truncated to n points with a small random perturbation."""
    grid_cols, grid_rows = 6, 6
    xs = np.linspace(0.5 / grid_cols, 1 - 0.5 / grid_cols, grid_cols)
    ys = np.linspace(0.5 / grid_rows, 1 - 0.5 / grid_rows, grid_rows)
    grid_positions = [(x, y) for y in ys for x in xs]
    init_pos = np.array(grid_positions[:n])
    init_pos += _rng.uniform(-0.01, 0.01, size=init_pos.shape)
    init_pos = np.clip(init_pos, 0.0, 1.0)
    init_r = np.full(n, 0.05)
    return init_pos, init_r

def _random_initialization(n: int):
    """Random positions and radii within bounds."""
    init_pos = _rng.uniform(0.05, 0.95, size=(n, 2))
    init_r = _rng.uniform(0.01, 0.15, size=n)
    return init_pos, init_r

def circle_packing32() -> np.ndarray:
    """
    Places 32 non‑overlapping circles in the unit square to maximize the sum of radii.
    The algorithm explores several deterministic and stochastic initializations,
    then refines each with SLSQP. The best solution across all restarts is returned.
    """
    n = 32
    init_methods = [_hex_initialization, _grid_initialization, _random_initialization]
    best_sum = -np.inf
    best_circles = None

    # Helper to extract indices
    def idx_x(i): return i
    def idx_y(i): return n + i
    def idx_r(i): return 2 * n + i

    # Objective: minimize negative sum of radii
    def objective(v):
        r = v[2 * n:3 * n]
        return -np.sum(r)

    for init_fn in init_methods:
        init_pos, init_r = init_fn(n)
        x0 = np.concatenate([init_pos[:, 0], init_pos[:, 1], init_r])

        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

        cons = []

        # Vectorized pairwise non‑overlap constraints
        def pairwise_constraints(v):
            """Return array of (dist² - (ri+rj)²) for all i<j."""
            x = v[:n]
            y = v[n:2*n]
            r = v[2*n:3*n]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dr = r[:, None] + r[None, :]
            dist_sq = dx**2 + dy**2
            iu = np.triu_indices(n, k=1)
            return dist_sq[iu] - dr[iu]**2
        cons.append({'type': 'ineq', 'fun': pairwise_constraints})

        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_x(i)] - v[idx_r(i)]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_x(i)] - v[idx_r(i)]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_y(i)] - v[idx_r(i)]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_y(i)] - v[idx_r(i)]})

        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 5000, 'disp': False}
        )

        if not res.success:
            # Skip infeasible runs
            continue

        sol = res.x
        circles = np.empty((n, 3))
        for i in range(n):
            circles[i, 0] = sol[idx_x(i)]
            circles[i, 1] = sol[idx_y(i)]
            circles[i, 2] = sol[idx_r(i)]

        radius_sum = np.sum(circles[:, 2])
        if radius_sum > best_sum:
            best_sum = radius_sum
            best_circles = circles

    # Fallback: if all runs failed, use hexagonal lattice
    if best_circles is None:
        pos, r = _hex_initialization(n)
        best_circles = np.column_stack((pos, r))

    return best_circles


# EVOLVE-BLOCK-END
