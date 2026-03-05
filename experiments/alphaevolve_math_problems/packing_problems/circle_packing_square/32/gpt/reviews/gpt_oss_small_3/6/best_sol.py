# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog

# ------------------------------------------------------------------
# Helper: hexagonal lattice packing
# ------------------------------------------------------------------
def _hexagonal_packing(n: int = 32) -> np.ndarray:
    """
    Generate a hexagonal lattice packing for n circles within the unit square.
    Radii are set to half the minimal distance to neighbors or to boundaries.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    s = 1.0 / (cols + 0.5)
    points = []
    for i in range(rows):
        y = (i + 0.5) * s
        offset = 0.0 if i % 2 == 0 else s / 2
        for j in range(cols):
            x = offset + (j + 0.5) * s
            if x <= 1 and y <= 1:
                points.append([x, y])
            if len(points) == n:
                break
        if len(points) == n:
            break
    points = np.array(points)
    radii = np.zeros(n)
    for idx, (x, y) in enumerate(points):
        d_boundary = min(x, 1 - x, y, 1 - y)
        if n > 1:
            d_neighbors = np.linalg.norm(points - np.array([x, y]), axis=1)
            d_neighbors[idx] = np.inf
            d_min = np.min(d_neighbors)
        else:
            d_min = np.inf
        radii[idx] = min(d_boundary, d_min / 2)
    return np.column_stack((points, radii))

# ------------------------------------------------------------------
# Helper: physics‑based incremental packing
# ------------------------------------------------------------------
def _physics_packing(n: int = 32, max_iter: int = 5000, dt: float = 0.01) -> np.ndarray:
    """
    Simple physics-based packing: random initial positions, incremental radius growth,
    and repulsive forces to avoid overlap.
    """
    np.random.seed(42)
    pos = np.random.rand(n, 2)
    radii = np.full(n, 0.01)
    for _ in range(max_iter):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        overlap = dist - (radii[:, None] + radii[None, :]) < 0
        for i in range(n):
            for j in range(i + 1, n):
                if overlap[i, j]:
                    dir_vec = diff[i, j]
                    if np.all(dir_vec == 0):
                        dir_vec = np.random.randn(2)
                    dir_vec /= np.linalg.norm(dir_vec)
                    move = (radii[i] + radii[j] - dist[i, j]) / 2
                    pos[i] += dir_vec * move
                    pos[j] -= dir_vec * move
        for i in range(n):
            for dim in range(2):
                if pos[i, dim] - radii[i] < 0:
                    pos[i, dim] = radii[i]
                if pos[i, dim] + radii[i] > 1:
                    pos[i, dim] = 1 - radii[i]
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        growth_possible = True
        for i in range(n):
            if pos[i, 0] - radii[i] - dt < 0 or pos[i, 0] + radii[i] + dt > 1:
                growth_possible = False
                break
            if pos[i, 1] - radii[i] - dt < 0 or pos[i, 1] + radii[i] + dt > 1:
                growth_possible = False
                break
            for j in range(n):
                if i == j:
                    continue
                if dist[i, j] < radii[i] + radii[j] + dt:
                    growth_possible = False
                    break
            if not growth_possible:
                break
        if growth_possible:
            radii += dt
        else:
            dt *= 0.5
            if dt < 1e-4:
                break
    return np.column_stack((pos, radii))

# ------------------------------------------------------------------
# Helper: SLSQP refinement with LP post‑processing
# ------------------------------------------------------------------
def _slsqp_packing(n: int = 32, restarts: int = 3) -> np.ndarray:
    """
    Deterministic SLSQP packing: optimize positions and radii simultaneously,
    starting from a hexagonal lattice. After SLSQP, perform LP refinement
    to maximize radii for fixed positions.
    """
    hex_circles = _hexagonal_packing(n)
    init_vars = hex_circles.flatten()

    def obj(v):
        return -np.sum(v[2::3])

    def constraint_func(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        cons = np.concatenate([
            xs - rs,
            1.0 - xs - rs,
            ys - rs,
            1.0 - ys - rs
        ])
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.hypot(dx, dy)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        cons = np.concatenate([cons, (dist - (rs[:, None] + rs[None, :]))[mask]])
        return cons

    bounds = [(0.0, 1.0)] * (3 * n)
    bounds[2::3] = [(0.0, 0.5)] * n

    best_res = None
    best_sum = -np.inf
    for restart in range(restarts):
        if restart == 0:
            init_guess = init_vars
        else:
            init_guess = best_res.x if best_res is not None else init_vars
            np.random.seed(42 + restart)
            init_guess = init_guess + np.random.normal(scale=0.01, size=3 * n)
        res = minimize(
            obj,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 4000, 'ftol': 1e-9, 'disp': False}
        )
        if res.success:
            cur_sum = -res.fun
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_res = res

    if best_res is None:
        return hex_circles

    best_pos = best_res.x.reshape((n, 3))[:, :2]
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        xi, yi = best_pos[i]
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row); b_ub.append(xi)
        A_ub.append(row.copy()); b_ub.append(1.0 - xi)
        A_ub.append(row.copy()); b_ub.append(yi)
        A_ub.append(row.copy()); b_ub.append(1.0 - yi)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(best_pos[i, 0] - best_pos[j, 0],
                            best_pos[i, 1] - best_pos[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row); b_ub.append(dist)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(0, None)] * n, method='highs')
    if res_lp.success:
        radii_opt = res_lp.x
    else:
        radii_opt = best_res.x.reshape((n, 3))[:, 2]
    circles = np.column_stack((best_pos[:, 0], best_pos[:, 1], radii_opt))
    return circles

# ------------------------------------------------------------------
# Main routine: try multiple strategies and pick the best
# ------------------------------------------------------------------
def circle_packing32() -> np.ndarray:
    """
    Generate 32 circles using three distinct strategies:
    1. Hexagonal lattice (fast, good baseline)
    2. Physics‑based incremental growth
    3. SLSQP refinement with LP post‑processing
    The best configuration (by sum of radii) is returned.
    """
    # Strategy 1: Hexagonal lattice
    hex_circles = _hexagonal_packing(32)
    hex_sum = np.sum(hex_circles[:, 2])

    # Strategy 2: Physics‑based packing
    phys_circles = _physics_packing(32)
    phys_sum = np.sum(phys_circles[:, 2])

    # Strategy 3: SLSQP refinement
    slsqp_circles = _slsqp_packing(32)
    slsqp_sum = np.sum(slsqp_circles[:, 2])

    # Choose the best
    best_sum = max(hex_sum, phys_sum, slsqp_sum)
    if best_sum == hex_sum:
        return hex_circles
    elif best_sum == phys_sum:
        return phys_circles
    else:
        return slsqp_circles


# EVOLVE-BLOCK-END
