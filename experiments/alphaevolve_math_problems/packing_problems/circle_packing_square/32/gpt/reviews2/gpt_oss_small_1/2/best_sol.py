# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import minimize

# Simulated annealing based circle packing for 32 circles
# This approach explores a distinct algorithmic pathway by treating the packing problem as a stochastic optimization
# with a deterministic random seed to ensure reproducibility.
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a deterministic SLSQP optimization with a hexagonal‑offset grid initialization,
    followed by a simulated‑annealing refinement phase.
    Returns:
        circles: np.array of shape (32,3), where the i‑th row (x,y,r) stores the (x,y) coordinates
        of the i‑th circle of radius r.
    """
    n = 32

    # ------------------------------
    # 1. Hexagonal‑offset grid initialization
    # ------------------------------
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
    r0 = np.full(n, 0.075)  # initial radius that allows growth

    # Flatten variables: [x0...xn-1, y0...yn-1, r0...rn-1]
    v0 = np.concatenate([init_x, init_y, r0])

    # Bounds: x,y in [0,1], r in [0,0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Objective: minimize negative sum of radii
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

        # Non‑overlap constraints (vectorized)
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
            options={"maxiter": 50000, "ftol": 1e-9, "disp": False},
        )
        if res.success:
            sum_r = -objective(res.x)
            if sum_r > best_sum:
                best_sum = sum_r
                best_res = res

    # If optimization failed, fall back to the initial grid
    if best_res is None:
        x_opt = init_x
        y_opt = init_y
        r_opt = r0
    else:
        x_opt = best_res.x[:n]
        y_opt = best_res.x[n : 2 * n]
        r_opt = best_res.x[2 * n :]

    slsqp_circles = np.column_stack([x_opt, y_opt, r_opt])

    # ------------------------------
    # 2. Simulated‑annealing refinement
    # ------------------------------
    def simulated_annealing_refinement(circles_init: np.ndarray,
                                       max_iter: int = 3000,
                                       T0: float = 1.0,
                                       alpha: float = 0.995) -> np.ndarray:
        rng_sa = np.random.default_rng(42)
        circles = circles_init.copy()
        best = circles.copy()
        best_sum = np.sum(circles[:, 2])
        T = T0
        n_c = circles.shape[0]

        for _ in range(max_iter):
            i = rng_sa.integers(0, n_c)
            # Propose new center uniformly
            x_new = rng_sa.random()
            y_new = rng_sa.random()
            # Propose new radius by scaling current radius
            scale = np.exp(rng_sa.standard_normal() * 0.1)
            r_new = circles[i, 2] * scale
            # Clip to boundaries
            r_new = min(r_new, x_new, 1 - x_new, y_new, 1 - y_new)
            if r_new <= 0:
                continue
            # Check overlap
            others = np.delete(circles, i, axis=0)
            dx = x_new - others[:, 0]
            dy = y_new - others[:, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if np.any(dist < r_new + others[:, 2] - 1e-9):
                continue
            old_r = circles[i, 2]
            new_sum = np.sum(circles[:, 2]) - old_r + r_new
            if new_sum > best_sum or rng_sa.random() < np.exp((new_sum - best_sum) / T):
                circles[i] = [x_new, y_new, r_new]
                if new_sum > best_sum:
                    best_sum = new_sum
                    best = circles.copy()
            T *= alpha
        return best

    sa_circles = simulated_annealing_refinement(slsqp_circles)

    # Choose better of SLSQP and SA
    if np.sum(sa_circles[:, 2]) > np.sum(slsqp_circles[:, 2]):
        circles = sa_circles
    else:
        circles = slsqp_circles

    return circles


# EVOLVE-BLOCK-END
