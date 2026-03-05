# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Strategy: Continuous optimization using SLSQP to maximize sum of radii
# with containment and non-overlap constraints expressed as inequalities.
# Deterministic random seed ensures reproducibility.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non‑overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a 6×6 hexagonal‑offset grid initialization with radius 0.07 and SLSQP refinement.
    Returns:
        circles: np.array of shape (32,3), where the i‑th row (x,y,r) stores the (x,y) coordinates of the i‑th circle of radius r.
    """
    n = 32

    # 6×6 hexagonal‑offset grid layout (first 32 cells)
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    init_x, init_y = [], []
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
    # Start with a slightly larger radius to give the optimizer more room to grow
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
            options={"maxiter": 40000, "ftol": 1e-9, "disp": False},
        )
        if res.success:
            sum_r = -objective(res.x)
            if sum_r > best_sum:
                best_sum = sum_r
                best_res = res

    # Final refinement from the best solution
    if best_res is not None:
        v_start = best_res.x
        res_refine = minimize(
            objective,
            v_start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_dict,
            options={"maxiter": 50000, "ftol": 1e-9, "disp": False},
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
