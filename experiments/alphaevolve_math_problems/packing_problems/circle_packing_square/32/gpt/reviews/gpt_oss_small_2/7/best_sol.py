# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Deterministic seed for reproducibility
    np.random.seed(42)
    n = 32

    # Initialise with random positions and a small radius
    # Initialise with a deterministic 8×4 grid layout to cover the unit square more evenly.
    cols, rows = 8, 4
    xs = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols)
    ys = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows)
    grid = [(x, y) for y in ys for x in xs]
    circles = np.zeros((n, 3))
    for i, (x, y) in enumerate(grid):
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = 0.05  # modest initial radius

    # --- Penalty helper ----------------------------------------------------
    def penalty(c: np.ndarray) -> float:
        """
        Computes a penalty for containment violations and overlaps.
        The penalty is the sum of all positive violations; larger penalties
        discourage infeasible configurations.
        """
        xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]

        # Containment: circle must lie entirely inside the unit square
        pen = np.sum(np.maximum(0, rs - xs))
        pen += np.sum(np.maximum(0, xs - (1 - rs)))
        pen += np.sum(np.maximum(0, rs - ys))
        pen += np.sum(np.maximum(0, ys - (1 - rs)))

        # Overlap: pairwise distance must be at least the sum of radii
        coords = np.stack([xs, ys], axis=1)
        dists = distance.cdist(coords, coords)
        i, j = np.triu_indices(n, k=1)
        overlap = rs[i] + rs[j] - dists[i, j]
        pen += np.sum(np.maximum(0, overlap))

        return pen

    # --- SLSQP multi‑start optimization ------------------------------------------------
    best_solution = circles.copy()
    best_sum = np.sum(best_solution[:, 2])

    # Multi‑start SLSQP
    for seed_offset in range(5):
        np.random.seed(42 + seed_offset)
        # Slight random perturbation to grid positions
        perturbed_positions = [(x + np.random.randn() * 0.01, y + np.random.randn() * 0.01)
                               for (x, y) in grid]
        x0 = np.array([pos[0] for pos in perturbed_positions])
        y0 = np.array([pos[1] for pos in perturbed_positions])
        r0 = np.full(n, 0.02)
        z0 = np.empty(3 * n)
        z0[0::3] = x0
        z0[1::3] = y0
        z0[2::3] = r0

        def objective(z: np.ndarray) -> float:
            return -np.sum(z[2::3])

        def constraints_fun(z: np.ndarray) -> np.ndarray:
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            cons_list = []
            cons_list.append(x - r)          # x >= r
            cons_list.append(1 - r - x)      # x <= 1 - r
            cons_list.append(y - r)          # y >= r
            cons_list.append(1 - r - y)      # y <= 1 - r
            idx_i, idx_j = np.triu_indices(n, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx ** 2 + dy ** 2
            sum_r = r[idx_i] + r[idx_j]
            cons_list.append(dist_sq - sum_r ** 2)
            return np.concatenate(cons_list)

        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.0, 0.5))  # r

        result = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints_fun},
            options={'ftol': 1e-9, 'maxiter': 2000, 'disp': False}
        )

        if result.success:
            z_opt = result.x
            current_sum = np.sum(z_opt[2::3])
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = z_opt.copy()
        else:
            # Fallback to equal radius grid
            r_max = 1.0 / (2 * cols)
            z_opt = np.empty(3 * n)
            z_opt[0::3] = np.array([pos[0] for pos in grid])
            z_opt[1::3] = np.array([pos[1] for pos in grid])
            z_opt[2::3] = np.full(n, r_max)
            current_sum = np.sum(z_opt[2::3])
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = z_opt.copy()

    # Convert best_solution to (n,3) array
    best = np.empty((n, 3))
    best[:, 0] = best_solution[0::3]
    best[:, 1] = best_solution[1::3]
    best[:, 2] = best_solution[2::3]

    # ------------------------------------------------------------------
    #  Post‑processing: iteratively tighten radii to the maximum allowed
    #  given the current positions and the radii of all other circles.
    #  This step guarantees a feasible configuration with maximised radii.
    # ------------------------------------------------------------------
    for _ in range(10):  # iterate a few times to converge
        xs, ys, rs = best[:, 0], best[:, 1], best[:, 2]
        # Compute pairwise distances once
        coords = np.stack([xs, ys], axis=1)
        dists = distance.cdist(coords, coords)
        # Boundary limits for each circle
        r_max = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
        for i in range(n):
            # Allowed radius from other circles
            allowed = dists[i] - rs
            allowed[i] = np.inf  # ignore self
            r_max[i] = min(r_max[i], np.min(allowed))
        best[:, 2] = np.clip(r_max, 0, 0.5)

    return best


# EVOLVE-BLOCK-END
