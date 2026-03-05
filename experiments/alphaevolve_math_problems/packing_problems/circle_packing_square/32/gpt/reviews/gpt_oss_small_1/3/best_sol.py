# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import minimize

# Hybrid SLSQP + local refinement based circle packing for 32 circles
# This implementation uses a deterministic seed, a hexagonal initial grid,
# SLSQP optimization with full constraints, greedy radius maximization,
# extensive local refinement, and a final radius perturbation phase.
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # deterministic seed

    # Initial hexagonal grid: 6 columns × 7 rows (42 cells, take first 32)
    cols, rows = 6, 7
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    init_x, init_y = [], []
    for j in range(rows):
        for i in range(cols):
            if len(init_x) >= n:
                break
            init_x.append((i + 1) * spacing_x)
            init_y.append((j + 1) * spacing_y)
        if len(init_x) >= n:
            break

    # Start with a safe radius that fits the grid
    init_r = [0.07] * n

    # Flatten initial state vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.empty(3 * n)
    for i in range(n):
        x0[3 * i] = init_x[i]
        x0[3 * i + 1] = init_y[i]
        x0[3 * i + 2] = init_r[i]

    # Bounds for each variable
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    # Objective: maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])

    # Constraints list
    cons = []

    # Boundary constraints: each circle must stay within the unit square
    for i in range(n):
        idx = 3 * i
        cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx] - x[idx + 2]})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: 1 - x[idx] - x[idx + 2]})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx + 1] - x[idx + 2]})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: 1 - x[idx + 1] - x[idx + 2]})

    # Pairwise non-overlap constraints
    def pair_constraint_factory(i, j):
        idx_i = 3 * i
        idx_j = 3 * j
        return lambda x: (x[idx_i] - x[idx_j]) ** 2 + (x[idx_i + 1] - x[idx_j + 1]) ** 2 - (x[idx_i + 2] + x[idx_j + 2]) ** 2

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': pair_constraint_factory(i, j)})

    # Helper to compute the maximum feasible radius for a circle given current positions and radii
    def max_radius(idx, pos, radii):
        x, y = pos
        max_r = min(x, 1 - x, y, 1 - y)
        for j in range(n):
            if j == idx:
                continue
            dx = x - positions[j, 0]
            dy = y - positions[j, 1]
            dist = math.hypot(dx, dy) - radii[j]
            if dist < max_r:
                max_r = dist
        return max(0.0, max_r)

    # Run SLSQP optimization
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 50000, 'ftol': 1e-12, 'disp': False}
    )

    # If optimization fails, fall back to initial guess
    if not res.success:
        sol = x0
    else:
        sol = res.x

    # Extract positions and radii from solution
    positions = sol[0::3].reshape(-1, 1)
    positions = np.hstack((positions, sol[1::3].reshape(-1, 1)))
    radii = sol[2::3]

    # Greedy radius maximization: iteratively increase each circle's radius to the maximum feasible value.
    for _ in range(20):
        improved = False
        for i in range(n):
            new_r = max_radius(i, positions[i], radii)
            if new_r > radii[i] + 1e-6:
                radii[i] = new_r
                improved = True
        if not improved:
            break

    # Local refinement: small random perturbations followed by recomputation of maximal radii.
    rng = np.random.default_rng(42)
    def local_refine(pos, rad, rng, steps=5000, step_size=0.01):
        best_pos = pos.copy()
        best_rad = rad.copy()
        best_sum = rad.sum()
        for _ in range(steps):
            i = rng.integers(0, n)
            new_pos = best_pos[i] + rng.normal(scale=step_size, size=2)
            new_pos = np.clip(new_pos, 0.0, 1.0)
            new_r = max_radius(i, new_pos, best_rad)
            if new_r <= 0:
                continue
            # Ensure no overlap with other circles
            overlap = False
            for j in range(n):
                if j == i:
                    continue
                dist = math.hypot(new_pos[0] - best_pos[j, 0], new_pos[1] - best_pos[j, 1])
                if dist < new_r + best_rad[j]:
                    overlap = True
                    break
            if overlap:
                continue
            new_sum = best_sum - best_rad[i] + new_r
            if new_sum > best_sum:
                best_pos[i] = new_pos
                best_rad[i] = new_r
                best_sum = new_sum
        return best_pos, best_rad

    positions, radii = local_refine(positions, radii, rng)

    # Final radius perturbation phase to escape shallow local optima
    for _ in range(5000):
        i = rng.integers(0, n)
        new_rad = radii[i] + rng.normal(scale=0.005)
        if new_rad <= 0:
            continue
        # Check containment
        if new_rad > positions[i, 0] or new_rad > 1 - positions[i, 0] or new_rad > positions[i, 1] or new_rad > 1 - positions[i, 1]:
            continue
        # Check overlap
        overlap = False
        for j in range(n):
            if j == i:
                continue
            dist = math.hypot(positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1])
            if dist < new_rad + radii[j]:
                overlap = True
                break
        if overlap:
            continue
        # Accept if sum increases
        if new_rad + radii.sum() - radii[i] > radii.sum():
            radii[i] = new_rad

    circles = np.column_stack((positions[:, 0], positions[:, 1], radii))
    return circles


# EVOLVE-BLOCK-END
