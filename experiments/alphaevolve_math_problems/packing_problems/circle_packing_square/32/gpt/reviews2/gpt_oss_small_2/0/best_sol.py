# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import math
from scipy.optimize import linprog, minimize

# ------------------------------------------------------------------
# Helper: compute optimal radii for fixed centers via linear programming
# ------------------------------------------------------------------
def _optimal_radii(centers: np.ndarray) -> np.ndarray:
    """
    Solve the linear program that maximizes the sum of radii given fixed centers.
    Constraints:
        - r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
        - r_i + r_j <= dist_ij for all i<j
    """
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    A_ub = []
    b_ub = []

    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row); b_ub.append(centers[i, 0])  # r_i <= x_i
        A_ub.append(row); b_ub.append(centers[i, 1])  # r_i <= y_i
        A_ub.append(row); b_ub.append(1.0 - centers[i, 0])  # r_i <= 1-x_i
        A_ub.append(row); b_ub.append(1.0 - centers[i, 1])  # r_i <= 1-y_i

    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0; row[j] = 1.0
            A_ub.append(row); b_ub.append(dist[i, j])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    c = -np.ones(n)  # maximize sum(r) -> minimize -sum(r)
    bounds = [(0.0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        return np.zeros(n)
    return res.x

# ------------------------------------------------------------------
# Helper: deterministic radius tightening
# ------------------------------------------------------------------
def _tighten_radii(pos: np.ndarray, rad: np.ndarray, iterations: int = 12) -> np.ndarray:
    """
    Deterministically tighten each radius to the maximum allowed by the current positions
    and other circles.  Vectorised implementation inspired by the best‑performing
    version in the inspiration programs.
    """
    n = pos.shape[0]
    for _ in range(iterations):
        # Boundary limits for all circles
        max_r = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                           np.minimum(pos[:, 1], 1 - pos[:, 1]))
        # Pairwise distances minus current radii
        diff = pos[:, None, :] - pos[None, :, :]          # shape (n, n, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=2)) - rad[None, :]
        np.fill_diagonal(dists, np.inf)                    # ignore self
        min_dists = np.min(dists, axis=1)                  # nearest neighbour clearance
        max_r = np.minimum(max_r, min_dists)               # tightest possible radius
        rad = np.maximum(0.0, max_r)                       # enforce non‑negative
    return rad

# ------------------------------------------------------------------
# Helper: annealing over centers
# ------------------------------------------------------------------
def _anneal_refine_centers(centers: np.ndarray,
                           n_iter: int = 5000,
                           init_step: float = 0.025,
                           seed: int = 0) -> np.ndarray:
    """
    Simulated annealing over center positions only.
    After each move, recompute optimal radii via LP and accept the move
    if the LP‑computed sum of radii improves or with a temperature‑based probability.
    """
    rng = np.random.default_rng(seed)
    cur_centers = centers.copy()
    cur_radii = _optimal_radii(cur_centers)
    cur_sum = cur_radii.sum()

    T = 0.12
    step = init_step

    for _ in range(n_iter):
        idx = rng.integers(0, cur_centers.shape[0])
        delta = rng.uniform(-step, step, size=2)
        new_pos = cur_centers[idx] + delta
        new_pos = np.clip(new_pos, 0.0, 1.0)

        new_centers = cur_centers.copy()
        new_centers[idx] = new_pos

        new_radii = _optimal_radii(new_centers)
        new_sum = new_radii.sum()

        if new_sum > cur_sum or rng.random() < np.exp((new_sum - cur_sum) / T):
            cur_centers = new_centers
            cur_radii = new_radii
            cur_sum = new_sum

        T *= 0.995
        step *= 0.995
        if T < 1e-4:
            break

    return cur_centers

# ------------------------------------------------------------------
# Helper: generate hexagonal lattice initial positions
# ------------------------------------------------------------------
def _hex_initial(n_circles: int, rng: np.random.Generator):
    """
    Construct a hexagonal lattice arrangement and find the largest uniform radius r
    such that at least n_circles circles fit in the unit square without overlap.
    Returns arrays (xs, ys, rs) for the initial positions and radii.
    """
    best_s = 0.0
    best_rc = (1, n_circles)
    for r in range(1, n_circles + 1):
        c = -(-n_circles // r)
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
    for i in range(r):
        for j in range(c):
            if len(positions) >= n_circles:
                break
            x = j * s + R
            y = i * vertical + R
            if i % 2 == 1:
                x += s / 2
            if x - R < 0 or x + R > 1 or y - R < 0 or y + R > 1:
                continue
            positions.append([x, y])
        if len(positions) >= n_circles:
            break
    while len(positions) < n_circles:
        positions.append([rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99)])
    xs, ys = zip(*positions)
    rs = [R] * n_circles
    return np.array(xs), np.array(ys), np.array(rs)

# ------------------------------------------------------------------
# Main packing routine
# ------------------------------------------------------------------
def circle_packing32() -> np.ndarray:
    """
    Places 32 non‑overlapping circles in the unit square in order to maximize the sum of radii.
    Pipeline:
        1. Hexagonal lattice initialization with LP‑based radii.
        2. Annealing over centers (with radius recomputation).
        3. Deterministic tightening.
        4. SLSQP refinement.
        5. Final deterministic tightening + local radius refinement.
    """
    n_circles = 32
    best_solution = None
    best_sum = -np.inf
    rng = np.random.default_rng(42)
    start_time = time.time()

    # ------------------------------------------------------------
    # 1. Multiple restarts
    # ------------------------------------------------------------
    for restart in range(20):  # more restarts for robustness
        # Guard against running out of time
        if time.time() - start_time > 55.0:
            break

        # Hexagonal lattice + LP radii
        xs, ys, rs = _hex_initial(n_circles, rng)
        centers = np.column_stack([xs, ys])

        # Jitter to escape symmetry
        centers += rng.uniform(-0.015, 0.015, size=centers.shape)
        centers = np.clip(centers, 0.0, 1.0)

        # Initial LP radii (already computed in _hex_initial)
        radii = _optimal_radii(centers)

        # --------------------------------------------------------
        # 2. Annealing over centers
        # --------------------------------------------------------
        refined_centers = _anneal_refine_centers(centers,
                                                 n_iter=12000,
                                                 init_step=0.025,
                                                 seed=restart)

        # --------------------------------------------------------
        # 3. Final LP radii + tightening
        # --------------------------------------------------------
        final_radii = _optimal_radii(refined_centers)
        final_radii = _tighten_radii(refined_centers, final_radii, iterations=12)

        # --------------------------------------------------------
        # 4. SLSQP refinement
        # --------------------------------------------------------
        x0 = np.empty(3 * n_circles)
        x0[:n_circles] = refined_centers[:, 0]
        x0[n_circles:2 * n_circles] = refined_centers[:, 1]
        x0[2 * n_circles:] = final_radii

        # Build constraints once (closure captures n_circles)
        cons = []
        for i in range(n_circles):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n_circles + i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2 * n_circles + i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n_circles + i] - v[2 * n_circles + i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n_circles + i] - v[2 * n_circles + i]})
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                cons.append({'type': 'ineq',
                             'fun': lambda v, i=i, j=j: (v[i] - v[j]) ** 2 + (v[n_circles + i] - v[n_circles + j]) ** 2
                             - (v[2 * n_circles + i] + v[2 * n_circles + j]) ** 2})

        def obj(v):
            return -np.sum(v[2 * n_circles:])

        bounds = [(0.0, 1.0)] * n_circles + [(0.0, 1.0)] * n_circles + [(0.0, 0.5)] * n_circles

        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 6000, 'disp': False})

        if not res.success:
            continue

        xs_opt = res.x[:n_circles]
        ys_opt = res.x[n_circles:2 * n_circles]
        rs_opt = res.x[2 * n_circles:3 * n_circles]
        sum_r = np.sum(rs_opt)

        # --------------------------------------------------------
        # 5. Final tightening + local radius refinement
        # --------------------------------------------------------
        # Re‑compute optimal radii for the SLSQP‑obtained centers
        final_radii = _optimal_radii(np.column_stack([xs_opt, ys_opt]))
        rs_opt = _tighten_radii(np.column_stack([xs_opt, ys_opt]), final_radii, iterations=12)

        # Local radius refinement (small random tweaks)
        rng_local = np.random.default_rng(restart + 100)
        current_sum = sum_r
        for _ in range(20000):
            i = rng_local.integers(n_circles)
            new_r = rs_opt[i] + rng_local.uniform(-0.004, 0.004)
            new_r = np.clip(new_r, 0.0, 0.5)
            # Containment
            if not (new_r <= xs_opt[i] <= 1 - new_r and
                    new_r <= ys_opt[i] <= 1 - new_r):
                continue
            # Overlap
            dx = xs_opt - xs_opt[i]
            dy = ys_opt - ys_opt[i]
            dists2 = dx ** 2 + dy ** 2
            dists2[i] = np.inf
            if np.any(dists2 < (new_r + rs_opt) ** 2):
                continue
            new_sum = current_sum - rs_opt[i] + new_r
            if new_sum > current_sum:
                rs_opt[i] = new_r
                current_sum = new_sum

        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = np.column_stack([xs_opt, ys_opt, rs_opt])

    if best_solution is None:
        xs, ys, rs = _hex_initial(n_circles, rng)
        best_solution = np.column_stack([xs, ys, rs])

    # Sort by decreasing radius for easier inspection
    best_solution = best_solution[best_solution[:, 2].argsort()[::-1]]
    return best_solution


# EVOLVE-BLOCK-END
