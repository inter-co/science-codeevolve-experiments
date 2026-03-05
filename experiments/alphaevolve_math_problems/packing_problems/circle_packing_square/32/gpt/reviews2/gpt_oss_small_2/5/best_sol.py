# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt

def _tighten_radii(pos: np.ndarray, rad: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Deterministically tighten each radius to the maximum allowed by the current positions
    and other circles.  This is a local refinement step that often improves the total sum.
    """
    n = pos.shape[0]
    for _ in range(iterations):
        for i in range(n):
            max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            d = np.hypot(pos[i, 0] - pos[:, 0], pos[i, 1] - pos[:, 1]) - rad
            max_r = min(max_r, np.min(d[d > 0]))
            rad[i] = max(0.0, max_r)
    return rad
# Helper functions for initialization, physics relaxation, and SLSQP optimization
def _hex_initialization(n: int) -> np.ndarray:
    """
    Generate an initial set of n circles using a hexagonal lattice.
    Searches over lattice spacing to find a configuration that yields a large sum of radii.
    Returns an array of shape (n,3) with columns (x, y, r).
    """
    best_sum = 0.0
    best_circles = None
    for s in np.linspace(0.05, 0.5, 46):
        vert_spacing = s * np.sqrt(3) / 2
        xs = np.arange(0, 1 + 1e-9, s)
        ys = np.arange(0, 1 + 1e-9, vert_spacing)
        points = []
        for i, y in enumerate(ys):
            offset = (s / 2) if (i % 2) else 0.0
            for x in xs:
                px = x + offset
                if px <= 1:
                    points.append((px, y))
        points = np.array(points)
        inside = (points[:, 0] >= 0) & (points[:, 0] <= 1) & (points[:, 1] >= 0) & (points[:, 1] <= 1)
        points = points[inside]
        if len(points) < n:
            continue
        radii = np.full(len(points), s / 2)
        boundary = np.minimum(
            np.minimum(points[:, 0], 1 - points[:, 0]),
            np.minimum(points[:, 1], 1 - points[:, 1])
        )
        radii = np.minimum(radii, boundary)
        idx = np.argsort(-radii)
        selected = idx[:n]
        sum_r = radii[selected].sum()
        if sum_r > best_sum:
            best_sum = sum_r
            best_circles = np.column_stack((points[selected], radii[selected]))
    if best_circles is None:
        best_circles = np.empty((0, 3))
    return best_circles

def _physics_relaxation(positions: np.ndarray, radii: np.ndarray,
                        iterations: int = 400, dt: float = 0.02) -> np.ndarray:
    """
    Physics‑based relaxation to eliminate overlaps while preserving containment.
    """
    n = positions.shape[0]
    pos = positions.copy()
    rad = radii.copy()

    for _ in range(iterations):
        # Pairwise overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.hypot(dx, dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 0:
                        ux, uy = dx / dist, dy / dist
                    else:
                        ux, uy = np.random.uniform(-1, 1, 2)
                        norm = np.hypot(ux, uy)
                        ux, uy = ux / norm, uy / norm
                    pos[i, 0] -= 0.5 * overlap * ux * dt
                    pos[i, 1] -= 0.5 * overlap * uy * dt
                    pos[j, 0] += 0.5 * overlap * ux * dt
                    pos[j, 1] += 0.5 * overlap * uy * dt

        # Tighten radii
        for i in range(n):
            max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            d = np.hypot(pos[i, 0] - pos[:, 0], pos[i, 1] - pos[:, 1]) - rad
            if np.any(d > 0):
                max_r = min(max_r, np.min(d[d > 0]))
            rad[i] = max(0.0, max_r)

        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)

    return np.column_stack((pos, rad))

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

    res = opt.minimize(
        objective,
        z0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints_fun},
        options={'ftol': 1e-9, 'maxiter': maxiter, 'disp': False}
    )

    if res.success:
        return res.x
    return z0

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Multi‑stage solver: hex initialization -> physics relaxation -> multi‑start SLSQP -> final tightening.
    """
    import numpy as np
    import scipy.optimize as opt

    n = 32
    best_overall = None
    best_sum_overall = -np.inf

    for restart in range(3):
        rng = np.random.default_rng(42 + restart)

        # Stage 1: hexagonal lattice initialization
        init = _hex_initialization(n)
        if init.shape[0] < n:
            # Fallback to grid if hex search fails
            cols, rows = 8, 4
            dx, dy = 1.0 / cols, 1.0 / rows
            xs = np.linspace(dx / 2, 1 - dx / 2, cols)
            ys = np.linspace(dy / 2, 1 - dy / 2, rows)
            grid_positions = np.array([(x, y) for y in ys for x in xs])[:n]
            init_radius = min(dx, dy) / 2 * 0.95
            init = np.column_stack((grid_positions, np.full(n, init_radius)))

        positions = init[:, :2]
        radii = init[:, 2]

        # Stage 2: physics relaxation
        relaxed = _physics_relaxation(positions, radii, iterations=400, dt=0.02)
        positions, radii = relaxed[:, :2], relaxed[:, 2]

        # Stage 3: multi‑start SLSQP refinement
        best_local = None
        best_local_sum = -np.inf
        for seed_offset in range(5):
            perturbed_pos = positions + rng.normal(scale=0.01, size=(n, 2))
            perturbed_pos = np.clip(perturbed_pos, 0, 1)
            z0 = np.empty(3 * n)
            z0[0::3] = perturbed_pos[:, 0]
            z0[1::3] = perturbed_pos[:, 1]
            z0[2::3] = radii
            z_opt = _slsqp_optimize(z0, n, maxiter=2500)
            cur_sum = z_opt[2::3].sum()
            if cur_sum > best_local_sum:
                best_local_sum = cur_sum
                best_local = np.column_stack((z_opt[0::3], z_opt[1::3], z_opt[2::3]))

        # Final deterministic tightening
        best_local[:, 2] = _tighten_radii(best_local[:, :2], best_local[:, 2], iterations=12)

        sum_r = best_local[:, 2].sum()
        if sum_r > best_sum_overall:
            best_sum_overall = sum_r
            best_overall = best_local.copy()

    return best_overall


# EVOLVE-BLOCK-END
