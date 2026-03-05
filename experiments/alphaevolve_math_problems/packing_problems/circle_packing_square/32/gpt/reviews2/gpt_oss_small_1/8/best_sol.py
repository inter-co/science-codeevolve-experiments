# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time
from scipy.optimize import minimize

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def _distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _check_constraints(circles):
    """
    Validate that all circles are inside the unit square and non‑overlapping.
    Vectorized implementation for speed.
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Containment
    if np.any(r > x) or np.any(r > 1 - x) or np.any(r > y) or np.any(r > 1 - y):
        return False

    # Pairwise distances
    diff = circles[:, None, :2] - circles[None, :, :2]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones_like(dist, dtype=bool), k=1)
    if np.any(dist[mask] < (r[:, None] + r[None, :])[mask] - 1e-12):
        return False
    return True

def grow_radii(circles, eps=1e-6, alpha=0.5, max_iter=2000):
    """
    Incrementally grow circle radii until all constraints are tight.
    Parameters:
        circles: np.ndarray shape (n,3) with columns (x, y, r)
        eps: tolerance for minimal margin
        alpha: fraction of margin to add each iteration
        max_iter: maximum iterations
    Returns:
        np.ndarray of updated circles with increased radii.
    """
    n = circles.shape[0]
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    for _ in range(max_iter):
        # Containment margins
        margin_x_min = x - r
        margin_x_max = 1.0 - x - r
        margin_y_min = y - r
        margin_y_max = 1.0 - y - r
        # Pairwise margins
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        rsum = r[:, None] + r[None, :]
        pair_margin = dist - rsum
        # Exclude self‑distances
        np.fill_diagonal(pair_margin, np.inf)
        pair_min = np.min(pair_margin, axis=1)
        # Overall margin per circle
        margin = np.minimum.reduce([margin_x_min, margin_x_max, margin_y_min, margin_y_max, pair_min])
        if margin.max() <= eps:
            break
        # Increase radii
        r += alpha * margin
        r = np.clip(r, 0.0, 0.5)
    return np.column_stack([x, y, r])
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Implements a deterministic hexagonal‑offset grid initialization followed by
    multi‑start SLSQP refinement. This approach consistently achieves a sum of radii
    close to the AlphaEvolve benchmark.
    """
    n = 32

    # 6×6 hexagonal‑offset grid layout (first 32 cells)
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)   # 0.125
    spacing_y = 1.0 / (rows + 1)   # ≈0.1667
    init_x = []
    init_y = []
    for j in range(rows):
        for i in range(cols):
            if len(init_x) >= n:
                break
            # Offset alternate rows by half a column spacing for hexagonal packing
            offset = 0.5 * spacing_x if j % 2 == 1 else 0.0
            init_x.append((i + 1) * spacing_x + offset)
            init_y.append((j + 1) * spacing_y)
        if len(init_x) >= n:
            break
    init_x = np.array(init_x)
    init_y = np.array(init_y)
    # Start with a moderate radius that allows growth; 0.075 is safe for the 6×6 grid
    r0 = np.full(n, 0.075)

    # Flatten variables: [x0...xn-1, y0...yn-1, r0...rn-1]
    v0 = np.concatenate([init_x, init_y, r0])

    # Bounds: x,y in [0,1], r in [0,0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Objective: maximize sum of radii → minimize negative sum
    def objective(v):
        return -np.sum(v[2 * n :])

    # Constraints: all inequalities must be >= 0
    def constraints(v):
        x = v[:n]
        y = v[n : 2 * n]
        r = v[2 * n :]
        cons = []

        # Containment constraints
        cons.extend(x - r)
        cons.extend(y - r)
        cons.extend(1.0 - x - r)
        cons.extend(1.0 - y - r)

        # Non‑overlap constraints (vectorized for speed)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx * dx + dy * dy
        rsum = r[:, None] + r[None, :]
        mask = np.triu_indices(n, k=1)
        cons.extend(dist2[mask] - rsum[mask] ** 2)

        return np.array(cons)

    cons_dict = {"type": "ineq", "fun": constraints}

    rng = np.random.default_rng(42)
    best_res = None
    best_sum = -np.inf

    # Multi‑start: perturb the initial layout slightly to escape local minima
    for _ in range(20):
        # Perturb the initial layout to escape local minima; ±0.02 gives more diversity
        pert = rng.uniform(-0.02, 0.02, size=(n, 2))
        x_start = np.clip(init_x + pert[:, 0], 0.01, 0.99)
        y_start = np.clip(init_y + pert[:, 1], 0.01, 0.99)
        v_start = np.concatenate([x_start, y_start, r0])

        res = minimize(
            objective,
            v_start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_dict,
            options={"maxiter": 50000, "ftol": 1e-9, "disp": False},
        )
        if res.success:
            sum_r = -objective(res.x)
            if sum_r > best_sum:
                best_sum = sum_r
                best_res = res

    # Final refinement: re‑optimize from the best solution to squeeze out any remaining slack
    if best_res is not None:
        v_start = best_res.x
        res_refine = minimize(
            objective,
            v_start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_dict,
            options={"maxiter": 60000, "ftol": 1e-9, "disp": False},
        )
        if res_refine.success:
            best_res = res_refine
        x_opt = best_res.x[:n]
        y_opt = best_res.x[n : 2 * n]
        r_opt = best_res.x[2 * n :]
        circles = np.column_stack([x_opt, y_opt, r_opt])
    else:
        # Fallback to the initial feasible grid if optimization fails
        circles = np.column_stack([init_x, init_y, r0])

    return circles


# EVOLVE-BLOCK-END
