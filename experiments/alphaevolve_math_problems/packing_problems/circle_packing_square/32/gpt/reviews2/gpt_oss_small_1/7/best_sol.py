# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time
import scipy.optimize as opt

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def _distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _check_constraints(circles):
    """
    Validate that all circles are inside the unit square and non‑overlapping.
    circles: np.ndarray shape (n,3) with columns (x, y, r)
    Returns: True if constraints satisfied, False otherwise.
    """
    n = circles.shape[0]
    # Containment
    if np.any(circles[:, 2] > circles[:, 0]) or np.any(circles[:, 2] > 1 - circles[:, 0]):
        return False
    if np.any(circles[:, 2] > circles[:, 1]) or np.any(circles[:, 2] > 1 - circles[:, 1]):
        return False
    # Non‑overlap
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            if _distance((xi, yi), (xj, yj)) < ri + rj - 1e-12:
                return False
    return True

# ------------------------------------------------------------------
# Hexagonal grid initialization + multi‑start SLSQP (Strategy A)
# ------------------------------------------------------------------
def _hexagonal_grid_slsqp(n_circles=32, seed=42):
    rng = np.random.default_rng(seed)
    n = n_circles

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
            offset = 0.5 * spacing_x if j % 2 == 1 else 0.0
            init_x.append((i + 1) * spacing_x + offset)
            init_y.append((j + 1) * spacing_y)
        if len(init_x) >= n:
            break
    init_x = np.array(init_x)
    init_y = np.array(init_y)
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

    best_res = None
    best_sum = -np.inf

    # Multi‑start: perturb the initial layout slightly to escape local minima
    for _ in range(20):
        pert = rng.uniform(-0.02, 0.02, size=(n, 2))
        x_start = np.clip(init_x + pert[:, 0], 0.01, 0.99)
        y_start = np.clip(init_y + pert[:, 1], 0.01, 0.99)
        v_start = np.concatenate([x_start, y_start, r0])

        res = opt.minimize(
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
        res_refine = opt.minimize(
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
        circles = np.column_stack([init_x, init_y, r0])

    return circles

# ------------------------------------------------------------------
# Simulated Annealing (global search) – Strategy B
# ------------------------------------------------------------------
def _simulated_annealing(n_circles=32, max_iter=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    r0 = 0.02
    circles = np.zeros((n_circles, 3))
    circles[:, 2] = r0
    circles[:, 0] = np.random.uniform(r0, 1 - r0, n_circles)
    circles[:, 1] = np.random.uniform(r0, 1 - r0, n_circles)

    best = circles.copy()
    best_sum = best[:, 2].sum()

    T0 = 1.0
    T_end = 1e-4
    alpha = (T_end / T0) ** (1.0 / max_iter)

    T = T0
    for it in range(max_iter):
        idx = random.randrange(n_circles)
        new_circles = best.copy()

        delta_r = random.uniform(-0.005, 0.005)
        delta_x = random.uniform(-0.01, 0.01)
        delta_y = random.uniform(-0.01, 0.01)

        new_r = new_circles[idx, 2] + delta_r
        new_x = new_circles[idx, 0] + delta_x
        new_y = new_circles[idx, 1] + delta_y

        new_r = max(0.001, min(new_r, 0.5))
        new_x = max(new_r, min(new_x, 1 - new_r))
        new_y = max(new_r, min(new_y, 1 - new_r))

        new_circles[idx] = [new_x, new_y, new_r]

        if not _check_constraints(new_circles):
            T *= alpha
            continue

        new_sum = new_circles[:, 2].sum()
        delta = new_sum - best_sum

        if delta > 0 or random.random() < math.exp(delta / T):
            best = new_circles
            best_sum = new_sum

        T *= alpha

    return best

# ------------------------------------------------------------------
# Main routine – choose best strategy
# ------------------------------------------------------------------
def circle_packing32() -> np.ndarray:
    """
    Places 32 non‑overlapping circles in the unit square in order to maximize the sum of radii.
    Implements two strategies: a deterministic hexagonal‑grid SLSQP (Strategy A) and
    a simulated‑annealing baseline (Strategy B). The best feasible configuration
    found by either strategy is returned.
    """
    n = 32

    # Strategy A: hexagonal grid + SLSQP
    hex_sol = _hexagonal_grid_slsqp(n_circles=n, seed=42)
    hex_sum = hex_sol[:, 2].sum()

    # Strategy B: simulated annealing (fallback)
    sa_sol = _simulated_annealing(n_circles=n, max_iter=5000, seed=42)
    sa_sum = sa_sol[:, 2].sum()

    # Choose the best
    if hex_sum >= sa_sum:
        best = hex_sol
    else:
        best = sa_sol

    # Final sanity check
    if not _check_constraints(best):
        best = sa_sol if hex_sum < sa_sum else hex_sol

    return best


# EVOLVE-BLOCK-END
