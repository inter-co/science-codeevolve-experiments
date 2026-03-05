# EVOLVE-BLOCK-START
import numpy as np
import math

# --------------------------------------------------------------------
# Helper: compute maximal radii for a given set of circle centres
# --------------------------------------------------------------------
def _compute_radii(centres: np.ndarray, max_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """
    Given fixed centres, compute the largest possible radii that satisfy
    containment and non-overlap constraints by fixed-point iteration.
    """
    n = centres.shape[0]
    # Pre‑compute pairwise Euclidean distances
    diff = centres[:, None, :] - centres[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Ensure self‑distance is large so it is ignored in the min operation
    np.fill_diagonal(dist, np.inf)

    # Distances to the four square boundaries
    dist_to_boundary = np.minimum(
        np.minimum(centres[:, 0], 1.0 - centres[:, 0]),
        np.minimum(centres[:, 1], 1.0 - centres[:, 1]),
    )

    radii = np.zeros(n)
    for _ in range(max_iter):
        # For each circle, the limiting radius is the minimum of:
        #   * distance to boundary
        #   * half the distance to every other circle minus that other circle's radius
        # Since we are iterating, we use the current radii estimate.
        new_radii = np.minimum(
            dist_to_boundary,
            np.min(dist - radii[None, :], axis=1),
        )
        # Radii cannot be negative
        new_radii = np.maximum(new_radii, 0.0)
        if np.max(np.abs(new_radii - radii)) < tol:
            break
        radii = new_radii
    return radii


# --------------------------------------------------------------------
# Main packing routine
# --------------------------------------------------------------------
import scipy.optimize as opt

def _slsqp_optimize(z0: np.ndarray, n: int, maxiter: int = 2000) -> np.ndarray:
    """
    Run a single SLSQP optimisation starting from z0.
    Returns the optimized vector if successful, otherwise returns z0 unchanged.
    """
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

    result = opt.minimize(
        objective,
        z0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints_fun},
        options={'ftol': 1e-9, 'maxiter': 2000, 'disp': False}
    )

    if result.success:
        return result.x
    return z0  # fallback to initial guess if optimisation fails

def circle_packing32() -> np.ndarray:
    """
    Generates an arrangement of 32 non‑overlapping circles inside the unit square.
    This implementation follows a two‑stage optimisation:
    1. Multi‑start SLSQP optimisation (inspired by Inpiration 2) to obtain a high‑quality
       initial solution for both positions and radii.
    2. Hill‑climbing on the centres to locally refine the packing.
    3. Final SLSQP refinement on the hill‑climbed solution to fine‑tune positions and radii.
    """
    np.random.seed(0)  # Deterministic behaviour

    n = 32
    # --------------------------------------------------------------------
    # Stage 1 – Multi‑start SLSQP optimisation
    # --------------------------------------------------------------------
    # Use a 9×4 rectangular grid (36 points) and remove four corners to keep 32.
    cols, rows = 9, 4
    init_r = 0.07  # Slightly larger initial radius to give SLSQP more room
    xs = np.linspace(init_r, 1 - init_r, cols)
    ys = np.linspace(init_r, 1 - init_r, rows)
    grid_positions = [(x, y) for y in ys for x in xs]
    if len(grid_positions) != cols * rows:
        raise ValueError("Grid size mismatch: expected 36 points")

    # Remove four corners: top‑left, top‑right, bottom‑left, bottom‑right
    corner_indices = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
    keep_mask = np.ones(len(grid_positions), dtype=bool)
    keep_mask[corner_indices] = False
    grid_positions = [grid_positions[i] for i in range(len(grid_positions)) if keep_mask[i]]

    if len(grid_positions) != n:
        raise ValueError("After removing corners, expected 32 circles")

    best_solution = None
    best_sum = -np.inf

    # Three starts to give SLSQP more exploration while staying within time budget
    for seed_offset in range(3):
        np.random.seed(42 + seed_offset)
        # Slight random perturbation to grid positions
        perturbed_positions = [
            (x + np.random.randn() * 0.01, y + np.random.randn() * 0.01)
            for (x, y) in grid_positions
        ]
        x0 = np.array([pos[0] for pos in perturbed_positions])
        y0 = np.array([pos[1] for pos in perturbed_positions])
        r0 = np.full(n, init_r)
        z0 = np.empty(3 * n)
        z0[0::3] = x0
        z0[1::3] = y0
        z0[2::3] = r0

        z_opt = _slsqp_optimize(z0, n)

        current_sum = np.sum(z_opt[2::3])
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = z_opt.copy()

    # Extract centres and radii from the best SLSQP solution
    centres = np.column_stack((best_solution[0::3], best_solution[1::3]))
    radii = best_solution[2::3]

    # --------------------------------------------------------------------
    # Stage 2 – Hill‑climbing refinement on the centres
    # --------------------------------------------------------------------
    best_centres = centres.copy()
    best_radii = radii.copy()
    best_sum = radii.sum()

    # Increase hill‑climb budget for finer search
    n_iter = 15000
    step_size = 0.05
    for _ in range(n_iter):
        idx = np.random.randint(0, n)
        old_pos = best_centres[idx].copy()

        perturb = (np.random.rand(2) - 0.5) * 2 * step_size
        new_pos = old_pos + perturb
        new_pos = np.clip(new_pos, 0.0, 1.0)

        best_centres[idx] = new_pos
        new_radii = _compute_radii(best_centres)

        new_sum = new_radii.sum()
        if new_sum > best_sum:
            best_radii = new_radii
            best_sum = new_sum
        else:
            best_centres[idx] = old_pos

        step_size *= 0.9995

    # --------------------------------------------------------------------
    # Stage 3 – Final SLSQP refinement on the hill‑climbed solution
    # --------------------------------------------------------------------
    z_final = np.empty(3 * n)
    z_final[0::3] = best_centres[:, 0]
    z_final[1::3] = best_centres[:, 1]
    z_final[2::3] = best_radii
    # Use a tighter tolerance for the final refinement
    z_final = _slsqp_optimize(z_final, n, maxiter=4000)

    # Assemble the final (x, y, r) array
    circles = np.column_stack((z_final[0::3], z_final[1::3], z_final[2::3]))
    return circles


# EVOLVE-BLOCK-END
