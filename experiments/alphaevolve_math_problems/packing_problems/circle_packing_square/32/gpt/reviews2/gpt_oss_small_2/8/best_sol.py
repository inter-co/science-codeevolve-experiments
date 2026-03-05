# EVOLVE-BLOCK-START
import numpy as np
import math
import random
from scipy.optimize import minimize

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # --- Deterministic seed for reproducibility ------------------------------------
    np.random.seed(42)
    random.seed(42)

    n = 32

    # --- Hexagonal lattice initial guess -------------------------------------------------
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
    r, c = best_rc
    s = best_s
    R = s / 2.0
    vertical = math.sqrt(3) / 2 * s
    positions = []
    radii = []
    for i in range(r):
        for j in range(c):
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
    # Pad with tiny circles if needed
    while len(positions) < n:
        r_small = 0.01
        x = np.random.uniform(r_small, 1 - r_small)
        y = np.random.uniform(r_small, 1 - r_small)
        positions.append([x, y])
        radii.append(r_small)

    init_vars = np.concatenate([np.array(positions)[:, 0], np.array(positions)[:, 1], np.array(radii)])

    # --- Bounds for variables: x, y ∈ [0,1], r ∈ [0,0.5] ---------------------------
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # --- Helper to build boundary constraints --------------------------------------
    def _boundary_fun(i, dim, sign):
        """
        dim: 0 for x, 1 for y
        sign:  1  → lower bound (x - r ≥ 0 or y - r ≥ 0)
               -1 → upper bound (1 - x - r ≥ 0 or 1 - y - r ≥ 0)
        """
        def fun(v):
            coord = v[i] if dim == 0 else v[n + i]
            rad = v[2 * n + i]
            return coord - rad if sign == 1 else 1.0 - coord - rad
        return fun

    # --- Helper to build overlap constraints ---------------------------------------
    def _overlap_fun(i, j):
        def fun(v):
            xi, yi = v[i], v[n + i]
            xj, yj = v[j], v[n + j]
            ri, rj = v[2 * n + i], v[2 * n + j]
            dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
            return dist2 - (ri + rj) ** 2
        return fun

    # --- Assemble constraints -------------------------------------------------------
    cons = []

    # --- Inflation helper: iteratively enlarge radii while respecting constraints ----
    def _inflate(positions: np.ndarray, radii: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
        """
        Gradually inflate each circle until it touches a neighbor or a boundary.
        Parameters
        ----------
        positions : np.ndarray, shape (n, 2)
            (x, y) coordinates of circles.
        radii : np.ndarray, shape (n,)
            Current radii of circles.
        max_iter : int
            Maximum number of inflation iterations.
        tol : float
            Minimum radius increase to continue iterating.
        Returns
        -------
        np.ndarray
            Inflated radii array.
        """
        n = len(positions)
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                xi, yi = positions[i]
                # Boundary limits
                max_r = min(xi, 1.0 - xi, yi, 1.0 - yi)
                # Neighbor limits
                for j in range(n):
                    if j == i:
                        continue
                    xj, yj = positions[j]
                    d = np.hypot(xi - xj, yi - yj) - radii[j]
                    if d < max_r:
                        max_r = d
                # Incrementally increase radius to avoid large jumps
                new_r = min(max_r, radii[i] + 0.005)
                if new_r > radii[i] + tol:
                    radii[i] = new_r
                    changed = True
            if not changed:
                break
        # Clip to valid bounds
        return np.clip(radii, 0.01, 0.5)

    # Boundary constraints
    for idx in range(n):
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0,  1)})  # x - r ≥ 0
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0, -1)})  # 1 - x - r ≥ 0
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1,  1)})  # y - r ≥ 0
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1, -1)})  # 1 - y - r ≥ 0

    # Non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': _overlap_fun(i, j)})

    # --- Objective: maximize sum of radii → minimize negative sum ------------------
    def objective(v):
        return -np.sum(v[2 * n:])

    # --- Multiple restarts of the optimizer -----------------------------------------
    best_sol = None
    best_sum = -np.inf
    for restart in range(5):
        # Deterministic RNG for this restart
        rng = np.random.default_rng(42 + restart)

        # Perturb positions and radii
        pert_pos = 0.005 * rng.standard_normal((n, 2))
        pert_r   = 0.01  * rng.standard_normal(n)

        init_positions = np.clip(np.array(positions) + pert_pos, 0.01, 0.99)
        init_radii = np.clip(np.array(radii) + pert_r, 0.01, 0.5)

        init_vars = np.concatenate([init_positions[:, 0], init_positions[:, 1], init_radii])

        res = minimize(
            objective,
            init_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 8000, 'disp': False}
        )

        if res.success:
            rs = res.x[2 * n:]
            sum_r = np.sum(rs)
            # Inflate radii after successful optimization
            rs = _inflate(np.column_stack([res.x[:n], res.x[n:2 * n]]), rs)
            sum_r = np.sum(rs)
            if sum_r > best_sum:
                best_sum = sum_r
                best_sol = np.concatenate([res.x[:n], res.x[n:2 * n], rs])

    # --- Fallback in case all restarts fail -----------------------------------------
    if best_sol is None:
        # Inflate initial guess before returning
        init_radii = _inflate(np.column_stack([np.array(positions)[:, 0], np.array(positions)[:, 1]]), np.array(radii))
        return np.column_stack([np.array(positions)[:, 0], np.array(positions)[:, 1], init_radii])

    # --- Extract solution ---------------------------------------------------------
    sol = best_sol
    xs = sol[:n]
    ys = sol[n:2 * n]
    rs = sol[2 * n:3 * n]
    return np.column_stack([xs, ys, rs])


# EVOLVE-BLOCK-END
