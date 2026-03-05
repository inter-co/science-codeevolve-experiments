# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK: Exploratory strategy using joint nonlinear programming with advanced solvers.
# We jointly optimize all circle positions/radii, maximizing sum of radii.
# The search is seeded with a hexagonal lattice for diversity, then refined using SLSQP.

def hexagonal_lattice_positions(n, margin=0.04):
    """Generate n points in a hexagonal grid within [0,1]x[0,1], using a margin."""
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    if cols == 1: dx = 0.5
    else: dx = (1 - 2 * margin) / (cols - 1)
    if rows == 1: dy = 0.5
    else: dy = (1 - 2 * margin) / (rows - 1)
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = margin + j * dx
            # Offset for hex grid
            y = margin + i * dy + (dx / 2 if j % 2 == 1 else 0)
            x = min(max(margin, x), 1 - margin)
            y = min(max(margin, y), 1 - margin)
            positions.append([x, y])
    return np.array(positions)

def pack_objective(params):
    """Negative sum of radii (since optimizer minimizes)."""
    n = len(params) // 3
    radii = params[2::3]
    return -np.sum(radii)

def constraint_containment(x):
    n = len(x) // 3
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    return np.concatenate([
        xs - rs,            # x >= r
        1 - rs - xs,        # x <= 1-r
        ys - rs,            # y >= r
        1 - rs - ys,        # y <= 1-r
        rs                  # r >= 0
    ])

def constraint_nonoverlap(x):
    n = len(x) // 3
    xs = x[:n]
    ys = x[n:2*n]
    rs = x[2*n:3*n]
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    sum_radii = rs[idx_i] + rs[idx_j]
    return dists - sum_radii

import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    start_time = time.time()

    KNOWN_BEST = 2.931414587079416  # Known best sum for early stop

    def objective(x):
        # Maximize sum of radii
        return -np.sum(x[2*n:3*n])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.01, 0.15)] * n
    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]

    best_circles = None
    best_sum = -np.inf
    n_starts = 12  # More multi-starts for robustness

    from scipy.spatial import Voronoi, KDTree

    def voronoi_init_positions_and_radii(n, seed=42, grid_shape=None, radii_scale=None):
        rng = np.random.RandomState(seed)
        if grid_shape is None:
            grid_choices = [(6,6), (5,7), (7,5)]
            num_rows, num_cols = grid_choices[rng.randint(len(grid_choices))]
        else:
            num_rows, num_cols = grid_shape
        dx = 1.0 / (num_cols + 0.5)
        dy = (np.sqrt(3)/2) * dx
        points = []
        for row in range(num_rows):
            for col in range(num_cols):
                if len(points) >= n:
                    break
                x = dx * (col + 0.5 * (row % 2) + 0.5)
                y = dy * (row + 0.5)
                x += 0.014 * rng.randn()
                y += 0.014 * rng.randn()
                x = np.clip(x, 0.07, 0.93)
                y = np.clip(y, 0.07, 0.93)
                points.append([x, y])
            if len(points) >= n:
                break
        points = np.array(points[:n])
        vor = Voronoi(points)
        radii = []
        tree = KDTree(vor.vertices)
        for idx, (x, y) in enumerate(points):
            min_dist = tree.query([x, y])[0]
            min_dist = min(min_dist, x, 1-x, y, 1-y)
            if radii_scale is None:
                scale = 0.68 + 0.12 * rng.rand()
            else:
                scale = radii_scale
            radii.append(min_dist * scale)
        return np.concatenate([points.reshape(-1,2), np.array(radii).reshape(-1,1)], axis=1)

    for start in range(n_starts):
        if time.time() - start_time > 55:  # leave room for fallback
            break
        seed = 42 + 17 * start
        rng = np.random.RandomState(seed)

        use_voronoi = (start % 2 == 0)
        if use_voronoi:
            grid_shape = None
            radii_scale = 0.69 + 0.07 * rng.rand()
            vor_init = voronoi_init_positions_and_radii(n, seed=seed, grid_shape=grid_shape, radii_scale=radii_scale)
            lattice = vor_init[:, :2]
            init_r = np.clip(vor_init[:, 2], 0.037, 0.13)
        else:
            lattice = hexagonal_lattice_positions(n, margin=0.03)
            lattice += rng.uniform(-0.012, 0.012, lattice.shape)
            lattice = np.clip(lattice, 0.04, 0.96)
            init_r = 0.065 + 0.018 * rng.randn(n)
            init_r = np.clip(init_r, 0.037, 0.12)
        x0 = np.concatenate([lattice[:,0], lattice[:,1], init_r])

        result = minimize(objective, x0, method='SLSQP',
                          constraints=cons, bounds=bounds,
                          options={'maxiter': 3400, 'ftol': 1e-8, 'disp': False})

        if result.success:
            xs = result.x[:n]
            ys = result.x[n:2*n]
            rs = result.x[2*n:3*n]
            circles_candidate = np.vstack([xs, ys, rs]).T

            # Robust greedy postprocessing: multiple inflation passes, smallest first
            for _ in range(3):
                idx_order = np.argsort(rs)
                for idx in idx_order:
                    # Maximum allowed by containment
                    rmax = min(xs[idx], ys[idx], 1-xs[idx], 1-ys[idx])
                    # Maximum allowed by other circles
                    for j in range(n):
                        if idx != j:
                            d = np.hypot(xs[idx]-xs[j], ys[idx]-ys[j])
                            if d > rs[j]:
                                rmax = min(rmax, d - rs[j])
                    rs[idx] = min(rmax, 0.15)
                    rs[idx] = max(rs[idx], 0.01)
                circles_candidate[:,2] = rs

            # Polishing: short SLSQP run after inflation
            x_polish = np.concatenate([xs, ys, rs])
            res2 = minimize(objective, x_polish, method='SLSQP',
                            constraints=cons, bounds=bounds,
                            options={'maxiter': 180, 'ftol': 1e-9, 'disp': False})
            if res2.success:
                xs2 = res2.x[:n]
                ys2 = res2.x[n:2*n]
                rs2 = res2.x[2*n:3*n]
                circles_candidate = np.vstack([xs2, ys2, rs2]).T

            # Final containment/overlap check and fix (2 passes, as in best inspiration)
            xs, ys, rs = circles_candidate[:,0], circles_candidate[:,1], circles_candidate[:,2]
            for _ in range(2):
                for i in range(n):
                    rs[i] = min(rs[i], xs[i], ys[i], 1-xs[i], 1-ys[i], 0.15)
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
                        max_rij = max(0.01, (d - 1e-10)/2)
                        if rs[i] + rs[j] > d:
                            if rs[i] > rs[j]:
                                rs[i] = min(rs[i], max_rij)
                            else:
                                rs[j] = min(rs[j], max_rij)
            circles_candidate[:,2] = rs

            radii_sum = np.sum(circles_candidate[:,2])
            if radii_sum > best_sum:
                best_sum = radii_sum
                best_circles = circles_candidate.copy()
            if radii_sum >= KNOWN_BEST:
                return circles_candidate

    if best_circles is not None:
        return best_circles

    # fallback: robust inflation and containment/overlap fix (as in inspiration 1)
    lattice = hexagonal_lattice_positions(n, margin=0.03)
    init_r = 0.065 + 0.022 * np.random.randn(n)
    init_r = np.clip(init_r, 0.035, 0.12)
    xs, ys = lattice[:,0], lattice[:,1]
    rs = init_r.copy()
    for _ in range(3):
        idx_order = np.argsort(rs)
        for idx in idx_order:
            rmax = min(xs[idx], ys[idx], 1-xs[idx], 1-ys[idx])
            for j in range(n):
                if idx != j:
                    d = np.hypot(xs[idx]-xs[j], ys[idx]-ys[j])
                    if d > rs[j]:
                        rmax = min(rmax, d - rs[j])
            rs[idx] = min(rmax, 0.15)
            rs[idx] = max(rs[idx], 0.01)
    for _ in range(2):
        for i in range(n):
            rs[i] = min(rs[i], xs[i], ys[i], 1-xs[i], 1-ys[i], 0.15)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
                max_rij = max(0.01, (d - 1e-10)/2)
                if rs[i] + rs[j] > d:
                    if rs[i] > rs[j]:
                        rs[i] = min(rs[i], max_rij)
                    else:
                        rs[j] = min(rs[j], max_rij)
    circles = np.vstack([xs, ys, rs]).T
    return circles


# EVOLVE-BLOCK-END
