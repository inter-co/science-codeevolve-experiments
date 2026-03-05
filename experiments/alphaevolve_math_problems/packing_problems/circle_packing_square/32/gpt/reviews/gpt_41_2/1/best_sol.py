# EVOLVE-BLOCK-START
import numpy as np

# EVOLVE-BLOCK: New imports for SLSQP only
from scipy.optimize import minimize

def non_overlap_constraint(circles):
    # Returns array of min_dist - (ri+rj) for all pairs (vectorized)
    n = circles.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = circles[idx_i,0] - circles[idx_j,0]
    dy = circles[idx_i,1] - circles[idx_j,1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - (circles[idx_i,2] + circles[idx_j,2])

def containment_constraint(circles):
    # Returns array of min(xi-ri, yi-ri, 1-xi-ri, 1-yi-ri)
    xs, ys, rs = circles[:,0], circles[:,1], circles[:,2]
    return np.concatenate([
        xs - rs,
        ys - rs,
        1.0 - xs - rs,
        1.0 - ys - rs,
        rs
    ])

def hexagonal_lattice_positions(n, margin=0.03):
    """Generate n points in a hexagonal grid within [0,1]x[0,1], using a margin."""
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    if cols == 1:
        dx = 0.5
    else:
        dx = (1 - 2 * margin) / (cols - 1)
    if rows == 1:
        dy = 0.5
    else:
        dy = (1 - 2 * margin) / (rows - 1)
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

import time

def optimize_radii_and_positions(n, time_limit=52):
    # Multi-start SLSQP with robust greedy inflation, post-polish, and robust fallback with both hex and Voronoi initializations (see Inspiration 2).
    import time
    start_time = time.time()
    best_circles = None
    best_sum = -np.inf

    def objective(x):
        return -np.sum(x[2*n:3*n])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.01, 0.15)] * n

    def constraint_containment(x):
        xs = x[:n]
        ys = x[n:2*n]
        rs = x[2*n:3*n]
        return np.concatenate([
            xs - rs,
            ys - rs,
            1.0 - xs - rs,
            1.0 - ys - rs,
            rs
        ])
    def constraint_nonoverlap(x):
        xs = x[:n]
        ys = x[n:2*n]
        rs = x[2*n:3*n]
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = xs[idx_i] - xs[idx_j]
        dy = ys[idx_i] - ys[idx_j]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        sum_radii = rs[idx_i] + rs[idx_j]
        return dists - sum_radii

    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]

    # Voronoi jittered initialization for diversity (as in Inspiration 2)
    def voronoi_init_points(n, seed=42):
        rng = np.random.default_rng(seed)
        grid_size = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(pts) < n:
                    x = (i + rng.uniform(0.2,0.8)) / grid_size
                    y = (j + rng.uniform(0.2,0.8)) / grid_size
                    pts.append([x,y])
        pts = np.array(pts)
        return np.clip(pts, 0.05, 0.95)

    # Global perturbation for initial guess
    def perturb_init(x0, n, rng):
        x0_perturb = np.copy(x0)
        if rng.uniform() < 0.35:
            perm = rng.permutation(n)
            x0_perturb[2*n:] = x0_perturb[2*n:][perm]
        if rng.uniform() < 0.25:
            i, j = rng.choice(n, size=2, replace=False)
            x0_perturb[2*i:2*i+2], x0_perturb[2*j:2*j+2] = x0_perturb[2*j:2*j+2].copy(), x0_perturb[2*i:2*i+2].copy()
        if rng.uniform() < 0.45:
            x0_perturb[:2*n] += rng.normal(0, 0.012, 2*n)
            x0_perturb[:2*n] = np.clip(x0_perturb[:2*n], 0.0, 1.0)
        return x0_perturb

    # Multi-start: some hex lattice, some Voronoi jittered, all perturbed
    rng = np.random.default_rng(2024)
    hex_starts = 10
    vor_starts = 4
    seeds = [42, 84, 321, 123, 777, 888]
    for seed in seeds:
        if time.time() - start_time > time_limit-6:
            break
        # Hex lattice starts
        for _ in range(hex_starts):
            if time.time() - start_time > time_limit-6:
                break
            np.random.seed(seed + _)
            lattice = hexagonal_lattice_positions(n, margin=0.03)
            lattice += np.random.uniform(-0.019, 0.019, lattice.shape)
            lattice = np.clip(lattice, 0.02, 0.98)
            init_r = 0.065 + 0.025 * np.random.randn(n)
            init_r = np.clip(init_r, 0.032, 0.13)
            x0 = np.concatenate([lattice[:,0], lattice[:,1], init_r])
            # With 40% probability, apply global perturbation
            if rng.uniform() < 0.4:
                x0 = perturb_init(x0, n, rng)
            result = minimize(objective, x0, method='SLSQP',
                              constraints=cons, bounds=bounds,
                              options={'maxiter': 2600, 'ftol': 1e-8, 'disp': False})
            if result.success:
                xs = result.x[:n]
                ys = result.x[n:2*n]
                rs = result.x[2*n:3*n]
                circles_candidate = np.vstack([xs, ys, rs]).T
                # Greedy inflation: multiple passes, smallest first
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
                    circles_candidate[:,2] = rs
                # Containment/overlap fix
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
        # Voronoi jittered starts
        for _ in range(vor_starts):
            if time.time() - start_time > time_limit-6:
                break
            rng_v = np.random.default_rng(seed + _)
            centers = voronoi_init_points(n, seed=seed + _)
            init_r = 0.065 + 0.025 * rng_v.normal(size=n)
            init_r = np.clip(init_r, 0.032, 0.13)
            x0 = np.concatenate([centers[:,0], centers[:,1], init_r])
            x0 = perturb_init(x0, n, rng_v)
            result = minimize(objective, x0, method='SLSQP',
                              constraints=cons, bounds=bounds,
                              options={'maxiter': 2300, 'ftol': 1e-8, 'disp': False})
            if result.success:
                xs = result.x[:n]
                ys = result.x[n:2*n]
                rs = result.x[2*n:3*n]
                circles_candidate = np.vstack([xs, ys, rs]).T
                for _ in range(2):
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
                    circles_candidate[:,2] = rs
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
    # fallback: robust greedy inflation and constraint fix, hex and voronoi, pick best
    lattice = hexagonal_lattice_positions(n, margin=0.03)
    init_r = 0.065 + 0.025 * np.random.randn(n)
    init_r = np.clip(init_r, 0.032, 0.13)
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
    # Voronoi jittered fallback
    try:
        centers = voronoi_init_points(n, seed=8888)
        init_r2 = 0.065 + 0.025 * np.random.randn(n)
        init_r2 = np.clip(init_r2, 0.032, 0.13)
        xs2, ys2 = centers[:,0], centers[:,1]
        rs2 = init_r2.copy()
        for _ in range(3):
            idx_order = np.argsort(rs2)
            for idx in idx_order:
                rmax = min(xs2[idx], ys2[idx], 1-xs2[idx], 1-ys2[idx])
                for j in range(n):
                    if idx != j:
                        d = np.hypot(xs2[idx]-xs2[j], ys2[idx]-ys2[j])
                        if d > rs2[j]:
                            rmax = min(rmax, d - rs2[j])
                rs2[idx] = min(rmax, 0.15)
                rs2[idx] = max(rs2[idx], 0.01)
        for _ in range(2):
            for i in range(n):
                rs2[i] = min(rs2[i], xs2[i], ys2[i], 1-xs2[i], 1-ys2[i], 0.15)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.hypot(xs2[i]-xs2[j], ys2[i]-ys2[j])
                    max_rij = max(0.01, (d - 1e-10)/2)
                    if rs2[i] + rs2[j] > d:
                        if rs2[i] > rs2[j]:
                            rs2[i] = min(rs2[i], max_rij)
                        else:
                            rs2[j] = min(rs2[j], max_rij)
        circles2 = np.vstack([xs2, ys2, rs2]).T
        if np.sum(rs2) > np.sum(rs):
            if best_circles is None or np.sum(rs2) > best_sum:
                return circles2
    except Exception:
        pass
    if best_circles is not None:
        return best_circles
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = optimize_radii_and_positions(n)
    return circles


# EVOLVE-BLOCK-END
