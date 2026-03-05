# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    import numpy as np
    import math
    from scipy.optimize import minimize

    n = 32

    # ------------------------------------------------------------------
    # 1. Hexagonal packing initial guess (inspired by Inspiration 1)
    # ------------------------------------------------------------------
    best_s = 0.0
    best_rc = (1, 32)
    for r in range(1, n + 1):
        c = -(-n // r)  # ceil division
        s1 = 1.0 / c
        s2 = 1.0 / ((r - 1) * math.sqrt(3) / 2 + 1)
        s = min(s1, s2)
        if s > best_s:
            best_s = s
            best_rc = (r, c)
    r_rows, c_cols = best_rc
    s = best_s
    R = s / 2.0
    vertical = math.sqrt(3) / 2 * s

    positions = []
    radii = []
    for i in range(r_rows):
        for j in range(c_cols):
            if len(positions) >= n:
                break
            x = j * s + R
            y = i * vertical + R
            if i % 2 == 1:
                x += s / 2
            if x - R < 0 or x + R > 1 or y - R < 0 or y + R > 1:
                continue
            positions.append([x, y])
            radii.append(R)
        if len(positions) >= n:
            break
    # Pad if necessary (unlikely)
    while len(positions) < n:
        positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
        radii.append(0.01)

    positions = np.array(positions)
    radii = np.array(radii)

    # ------------------------------------------------------------------
    # 2. SLSQP refinement with multiple restarts
    # ------------------------------------------------------------------
    base_radius = 0.07  # slightly larger base radius for more room
    best_solution = None
    best_sum = -np.inf

    for restart in range(8):  # increased restarts for better exploration
        # Deterministic seed for reproducibility
        np.random.seed(42 + restart)

        # Random perturbation of radii and positions
        pert_r = 0.01 * np.random.randn(n)
        init_radii = np.clip(radii + pert_r, 0.01, 0.5)

        pert_pos = 0.01 * np.random.randn(n, 2)  # larger position perturbation
        init_positions = np.clip(positions + pert_pos, 0.01, 0.99)

        init_vars = np.concatenate([init_positions[:, 0], init_positions[:, 1], init_radii])

        # Bounds: x, y ∈ [0,1], r ∈ [0,0.5]
        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

        # Constraints
        cons = []

        # Boundary constraints for each circle
        for idx in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: v[idx] - v[2 * n + idx]})
            cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: 1.0 - v[idx] - v[2 * n + idx]})
            cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: v[n + idx] - v[2 * n + idx]})
            cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: 1.0 - v[n + idx] - v[2 * n + idx]})

        # Non‑overlap constraints for every pair of circles
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i] - v[j]) ** 2 + (v[n + i] - v[n + j]) ** 2
                    - (v[2 * n + i] + v[2 * n + j]) ** 2
                })

        # Objective: maximize sum of radii → minimize negative sum
        def objective(v):
            return -np.sum(v[2 * n:])

        # Run the optimizer
        res = minimize(
            objective,
            init_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-8, 'maxiter': 6000, 'disp': False}
        )

        if res.success:
            xs = res.x[:n]
            ys = res.x[n:2 * n]
            rs = res.x[2 * n:3 * n]
            sum_r = np.sum(rs)
            if sum_r > best_sum:
                best_sum = sum_r
                best_solution = np.column_stack([xs, ys, rs])

    # Fallback: return the hex packing if optimization failed
    if best_solution is None:
        return np.column_stack([positions, radii])

    return best_solution


# EVOLVE-BLOCK-END
