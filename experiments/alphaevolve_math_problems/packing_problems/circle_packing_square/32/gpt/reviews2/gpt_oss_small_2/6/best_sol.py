# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import distance
from scipy.optimize import minimize

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# Helper: compute maximal radii for a fixed set of centres
def _compute_radii(centres: np.ndarray, max_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """
    Given fixed centres, compute the largest possible radii that satisfy
    containment and non‑overlap constraints by fixed‑point iteration.
    """
    n = centres.shape[0]
    diff = centres[:, None, :] - centres[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    dist_to_boundary = np.minimum(
        np.minimum(centres[:, 0], 1.0 - centres[:, 0]),
        np.minimum(centres[:, 1], 1.0 - centres[:, 1]),
    )

    radii = np.zeros(n)
    for _ in range(max_iter):
        new_radii = np.minimum(
            dist_to_boundary,
            np.min(dist - radii[None, :], axis=1),
        )
        new_radii = np.maximum(new_radii, 0.0)
        if np.max(np.abs(new_radii - radii)) < tol:
            break
        radii = new_radii
    return radii

# Helper: single SLSQP optimisation
def _slsqp_optimize(z0: np.ndarray, n: int) -> np.ndarray:
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

    result = minimize(
        objective,
        z0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints_fun},
        options={'ftol': 1e-9, 'maxiter': 5000, 'disp': False}
    )
    if result.success:
        return result.x
    return z0
def _physics_relaxation(centres: np.ndarray, rng: np.random.Generator, steps: int = 2000, dt: float = 0.01) -> np.ndarray:
    """
    Simple physics‑based relaxation: treat circles as particles with repulsive forces.
    At each step, compute forces from other circles and boundaries, update positions,
    then recompute radii via fixed‑point iteration.
    """
    n = centres.shape[0]
    pos = centres.copy()
    for _ in range(steps):
        # pairwise distance matrix
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        # avoid division by zero
        inv_dist_sq = 1.0 / (dist**2 + 1e-12)
        # repulsive force vectors
        force = diff * inv_dist_sq[:, :, None]
        # sum forces from all other circles
        net_force = np.sum(force, axis=1)
        # boundary forces (push away from edges)
        boundary_force = np.zeros_like(pos)
        boundary_force[:, 0] += 1.0 / (pos[:, 0] + 1e-6) - 1.0 / (1 - pos[:, 0] + 1e-6)
        boundary_force[:, 1] += 1.0 / (pos[:, 1] + 1e-6) - 1.0 / (1 - pos[:, 1] + 1e-6)
        net_force += boundary_force
        # update positions
        pos += dt * net_force
        pos = np.clip(pos, 0, 1)
        # recompute radii to keep them maximal
        _compute_radii(pos)
    return pos
def _slsqp_optimize(z0: np.ndarray, n: int) -> np.ndarray:
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

    result = minimize(
        objective,
        z0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints_fun},
        options={'ftol': 1e-9, 'maxiter': 500, 'disp': False}
    )
    if result.success:
        return result.x
    return z0

def circle_packing32() -> np.ndarray:
    """
    Generates an arrangement of 32 non‑overlapping circles inside the unit square.
    Uses a deterministic hexagonal lattice initialization, multiple SLSQP restarts,
    and a final tightening step to maximize the sum of radii.
    """
    rng = np.random.default_rng(42)
    n = 32

    # ------------------------------------------------------------------
    #  Hexagonal lattice initialization
    # ------------------------------------------------------------------
    best_s = 0.0
    best_rc = (1, n)
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
    # Pad if necessary
    while len(positions) < n:
        positions.append([rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99)])
        radii.append(0.01)

    init_vars = np.concatenate([np.array(positions)[:, 0],
                                np.array(positions)[:, 1],
                                np.array(radii)])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Constraint functions
    def _boundary_fun(i, dim, sign):
        def fun(v):
            coord = v[i] if dim == 0 else v[n + i]
            rad   = v[2 * n + i]
            return coord - rad if sign == 1 else 1.0 - coord - rad
        return fun

    def _overlap_fun(i, j):
        def fun(v):
            xi, yi = v[i], v[n + i]
            xj, yj = v[j], v[n + j]
            ri, rj = v[2 * n + i], v[2 * n + j]
            dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
            return dist2 - (ri + rj) ** 2
        return fun

    cons = []
    for idx in range(n):
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0,  1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0, -1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1,  1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1, -1)})

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': _overlap_fun(i, j)})

    def objective(v):
        return -np.sum(v[2 * n:])

    best_solution = None
    best_sum = -np.inf

    # ------------------------------------------------------------------
    #  Multiple SLSQP restarts with perturbed radii
    # ------------------------------------------------------------------
    for restart in range(5):
        pert = rng.uniform(-0.005, 0.005, size=n)
        init_radii = np.array(radii) + pert
        init_radii = np.clip(init_radii, 0.01, 0.5)
        init_vars = np.concatenate([np.array(positions)[:, 0],
                                    np.array(positions)[:, 1],
                                    init_radii])

        res = minimize(
            objective,
            init_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 5000, 'disp': False}
        )

        if res.success:
            xs = res.x[:n]
            ys = res.x[n:2 * n]
            rs = res.x[2 * n:3 * n]
            sum_r = np.sum(rs)
            if sum_r > best_sum:
                best_sum = sum_r
                best_solution = np.column_stack([xs, ys, rs])

    if best_solution is None:
        # fallback to initial hex packing
        return np.column_stack([np.array(positions)[:, 0],
                                np.array(positions)[:, 1],
                                np.array(radii)])

    # ------------------------------------------------------------------
    #  Optional tightening to ensure radii are maximal
    # ------------------------------------------------------------------
    circles = best_solution.copy()
    for _ in range(5):
        xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
        coords = np.stack([xs, ys], axis=1)
        dists = distance.cdist(coords, coords)
        r_max = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
        for i in range(n):
            allowed = dists[i] - rs
            allowed[i] = np.inf
            r_max[i] = min(r_max[i], np.min(allowed))
        circles[:, 2] = np.clip(r_max, 0, 0.5)

    return circles


# EVOLVE-BLOCK-END
