# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog, minimize

# ------------------------------------------------------------------
# Utility: compute optimal radii for a fixed set of circle centers
# ------------------------------------------------------------------
def _optimal_radii(centers: np.ndarray) -> np.ndarray:
    """
    Solve the linear program that maximizes the sum of radii given fixed centers.
    Constraints:
        - r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
        - r_i + r_j <= dist_ij for all i<j
    """
    n = centers.shape[0]
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # Build inequality constraints A_ub * r <= b_ub
    # 4 wall constraints per circle
    A_ub = []
    b_ub = []

    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        # r_i <= x_i
        A_ub.append(row)
        b_ub.append(centers[i, 0])
        # r_i <= y_i
        A_ub.append(row)
        b_ub.append(centers[i, 1])
        # r_i <= 1 - x_i
        A_ub.append(row)
        b_ub.append(1.0 - centers[i, 0])
        # r_i <= 1 - y_i
        A_ub.append(row)
        b_ub.append(1.0 - centers[i, 1])

    # Inter-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist[i, j])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)

    # Bounds: radii >= 0
    bounds = [(0.0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        # Fallback: return zeros if LP fails
        return np.zeros(n)

    return res.x

# ------------------------------------------------------------------
# Greedy placement of centers
# ------------------------------------------------------------------
def _greedy_centers(seed: int, n_circles: int = 32, n_candidates: int = 4000) -> np.ndarray:
    """
    Place circle centers greedily: at each step pick the candidate position that allows
    the largest possible radius given the already placed circles.
    """
    rng = np.random.default_rng(seed)
    centers = np.empty((0, 2))
    radii = np.empty((0,))

    for k in range(n_circles):
        best_pos = None
        best_radius = -1.0

        # Generate random candidates
        candidates = rng.uniform(0.0, 1.0, size=(n_candidates, 2))

        for cand in candidates:
            # Compute minimal distance to walls
            dist_to_walls = min(cand[0], cand[1], 1.0 - cand[0], 1.0 - cand[1])

            # Compute minimal distance to existing centers
            if centers.shape[0] > 0:
                dists = np.linalg.norm(centers - cand, axis=1)
                min_dist = np.min(dists)
                radius = min(dist_to_walls, min_dist / 2.0)
            else:
                radius = dist_to_walls

            if radius > best_radius:
                best_radius = radius
                best_pos = cand

        centers = np.vstack([centers, best_pos])
        radii = np.append(radii, best_radius)

    return centers

# ------------------------------------------------------------------
# Local refinement via simulated annealing on centers
# ------------------------------------------------------------------
def _anneal_refine(centers: np.ndarray, n_iter: int = 500, init_step: float = 0.05, seed: int = 0) -> np.ndarray:
    """
    Perform a simple simulated annealing over the center positions.
    After each move, recompute optimal radii via LP.
    """
    rng = np.random.default_rng(seed)
    current_centers = centers.copy()
    current_radii = _optimal_radii(current_centers)
    current_sum = current_radii.sum()

    T = 0.1
    step = init_step

    for it in range(n_iter):
        i = rng.integers(0, current_centers.shape[0])
        # propose new position
        delta = rng.uniform(-step, step, size=2)
        new_pos = current_centers[i] + delta
        new_pos = np.clip(new_pos, 0.0, 1.0)

        new_centers = current_centers.copy()
        new_centers[i] = new_pos

        new_radii = _optimal_radii(new_centers)
        new_sum = new_radii.sum()

        if new_sum > current_sum or rng.random() < np.exp((new_sum - current_sum) / T):
            current_centers = new_centers
            current_radii = new_radii
            current_sum = new_sum

        # cooling schedule
        T *= 0.995
        step *= 0.995

    return current_centers

# ------------------------------------------------------------------
# Additional local hill‑climbing refinement on centers
# ------------------------------------------------------------------
def _hill_climb_centers(centers: np.ndarray, max_steps: int = 2000, init_step: float = 0.02) -> np.ndarray:
    """
    Perform a deterministic hill‑climb on the center positions.
    Moves are accepted only if the LP‑computed sum of radii increases.
    """
    rng = np.random.default_rng()
    current_centers = centers.copy()
    current_radii = _optimal_radii(current_centers)
    current_sum = current_radii.sum()
    step = init_step

    for _ in range(max_steps):
        # Randomly pick a circle to perturb
        i = rng.integers(0, current_centers.shape[0])
        delta = rng.uniform(-step, step, size=2)
        new_pos = current_centers[i] + delta
        new_pos = np.clip(new_pos, 0.0, 1.0)

        new_centers = current_centers.copy()
        new_centers[i] = new_pos
        new_radii = _optimal_radii(new_centers)
        new_sum = new_radii.sum()

        if new_sum > current_sum:
            current_centers = new_centers
            current_radii = new_radii
            current_sum = new_sum
            # reset step size when improvement occurs
            step = init_step
        else:
            # gradually reduce step size to refine search
            step *= 0.99

    return current_centers

# ------------------------------------------------------------------
# Main packing routine
# ------------------------------------------------------------------
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid of hexagonal and random starts, refined by SLSQP, and final LP check.
    """
    n_circles = 32
    best_solution = None
    best_sum = -np.inf

    # Ensure reproducibility
    np.random.seed(42)

    start_time = time.time()

    # Helper: generate hexagonal lattice initial positions
    def _hex_initial():
        best_s = 0.0
        best_rc = (1, 32)
        for r in range(1, n_circles + 1):
            c = -(-n_circles // r)  # ceil division
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
        # Pad if necessary
        while len(positions) < n_circles:
            positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
        xs, ys = zip(*positions)
        rs = [R] * n_circles
        return np.array(xs), np.array(ys), np.array(rs)

    # SLSQP constraint helpers
    def _boundary_fun(i, dim, sign):
        def fun(v):
            coord = v[i] if dim == 0 else v[n_circles + i]
            rad   = v[2 * n_circles + i]
            return coord - rad if sign == 1 else 1.0 - coord - rad
        return fun

    def _overlap_fun(i, j):
        def fun(v):
            xi, yi = v[i], v[n_circles + i]
            xj, yj = v[j], v[n_circles + j]
            ri, rj = v[2 * n_circles + i], v[2 * n_circles + j]
            dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
            return dist2 - (ri + rj) ** 2
        return fun

    # Objective: maximize sum of radii
    def objective(v):
        return -np.sum(v[2 * n_circles:])

    # Bounds for variables
    bounds = [(0.0, 1.0)] * n_circles + [(0.0, 1.0)] * n_circles + [(0.0, 0.5)] * n_circles

    # Constraint list
    cons = []
    for idx in range(n_circles):
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0,  1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0, -1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1,  1)})
        cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1, -1)})

    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            cons.append({'type': 'ineq', 'fun': _overlap_fun(i, j)})

    # Additional helper: random LP‑optimised start
    def _random_start(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        centers = rng.uniform(0.01, 0.99, size=(n_circles, 2))
        radii = _optimal_radii(centers)
        return np.column_stack([centers, radii])

    # Multiple restarts (hex + random)
    for restart in range(15):
        if time.time() - start_time > 20.0:
            break

        # Alternate between hex and random starts
        if restart % 2 == 0:
            xs, ys, rs = _hex_initial()
        else:
            start = _random_start(restart)
            xs, ys, rs = start[:,0], start[:,1], start[:,2]

        # Perturb radii slightly to escape symmetry
        rs = np.clip(rs + np.random.uniform(-0.005, 0.005, size=n_circles), 0.01, 0.5)
        init_vars = np.concatenate([xs, ys, rs])

        xs, ys, rs = _hex_initial()
        # Perturb radii slightly to escape symmetry
        rs = np.clip(rs + np.random.uniform(-0.005, 0.005, size=n_circles), 0.01, 0.5)
        init_vars = np.concatenate([xs, ys, rs])

        try:
            res = minimize(
                objective,
                init_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'ftol': 1e-9, 'maxiter': 15000, 'disp': False}
            )
        except Exception:
            continue

        if res.success:
            xs_opt = res.x[:n_circles]
            ys_opt = res.x[n_circles:2 * n_circles]
            rs_opt = res.x[2 * n_circles:3 * n_circles]
            sum_r = np.sum(rs_opt)

            # Post‑SLSQP hill‑climb on centers
            centers = np.column_stack([xs_opt, ys_opt])
            centers = _hill_climb_centers(centers, max_steps=2000, init_step=0.02)
            # Recompute optimal radii for the improved centers
            rs_opt = _optimal_radii(centers)
            sum_r = np.sum(rs_opt)

            if sum_r > best_sum:
                best_sum = sum_r
                best_solution = np.column_stack([centers[:,0], centers[:,1], rs_opt])

    if best_solution is None:
        # Fallback to hex initialization if optimization fails
        xs, ys, rs = _hex_initial()
        best_solution = np.column_stack([xs, ys, rs])

    # Final LP refinement to guarantee optimal radii for the chosen centers
    final_centers = best_solution[:, :2]
    final_radii = _optimal_radii(final_centers)
    best_solution[:, 2] = final_radii

    return best_solution


# EVOLVE-BLOCK-END
