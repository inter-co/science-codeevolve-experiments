# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

def triangular_grid_circle_centers(n, square_side=1.0):
    """
    Generate circle centers using a triangular lattice inside a unit square,
    attempting to fit n circles and maximize the sum of radii.
    Returns: (centers, radius)
    """
    best_r = 0
    best_centers = None
    for rows in range(5, 10):
        for cols in range(5, 10):
            num = rows * cols
            if num < n:
                continue
            r_x = square_side / (cols + 0.5)
            r_y = square_side / (1 + (rows-1)*np.sqrt(3)/2)
            r = min(r_x, r_y)
            centers = []
            for i in range(rows):
                y = r + i * r * np.sqrt(3)/2
                for j in range(cols):
                    x = r + j * r * 2
                    if i % 2 == 1:
                        x += r
                    if x > square_side - r or y > square_side - r:
                        continue
                    centers.append((x, y))
            if len(centers) >= n and r > best_r:
                best_r = r
                best_centers = centers[:n]
    return best_centers, best_r

def local_grow(circles, box_poly, max_iters=12):
    """
    For each circle, maximize its radius locally given neighbors and box constraint.
    """
    n = len(circles)
    centers = np.array([[c[0], c[1]] for c in circles])
    radii = np.array([c[2] for c in circles])
    for _ in range(max_iters):
        for i in range(n):
            min_dist_box = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            min_dist_other = np.inf
            for j in range(n):
                if i == j:
                    continue
                dist = np.hypot(*(centers[i]-centers[j]))
                min_dist_other = min(min_dist_other, dist - radii[j])
            allowed = max(0.0, min(min_dist_box, min_dist_other))
            radii[i] = allowed
    return np.column_stack([centers, radii])

def force_directed_relax(circles, box_poly, max_iters=350, shrink_factor=0.98):
    """
    Physics-inspired repulsion and boundary enforcement.
    """
    n = len(circles)
    centers = np.array([[c[0],c[1]] for c in circles])
    radii = np.array([c[2] for c in circles])
    for _ in range(8):
        radii *= shrink_factor
        for it in range(max_iters):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    dist = np.hypot(dx, dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        overlap = min_dist - dist + 1e-3
                        if dist == 0:
                            angle = np.random.uniform(0,2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                        else:
                            dx /= dist
                            dy /= dist
                        centers[i,0] += dx * overlap * 0.5
                        centers[i,1] += dy * overlap * 0.5
                        centers[j,0] -= dx * overlap * 0.5
                        centers[j,1] -= dy * overlap * 0.5
                        moved = True
                centers[i,0] = np.clip(centers[i,0], radii[i], 1-radii[i])
                centers[i,1] = np.clip(centers[i,1], radii[i], 1-radii[i])
            if not moved:
                break
    return np.column_stack([centers, radii])

def pack_objective(x, n):
    # Minimize negative sum of radii
    return -np.sum(x[2::3])

def pack_constraint_circle_bounds(x, n):
    cons = []
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        cons.append(xi - ri)      # xi >= ri
        cons.append(1 - ri - xi)  # xi <= 1-ri
        cons.append(yi - ri)      # yi >= ri
        cons.append(1 - ri - yi)  # yi <= 1-ri
        cons.append(ri)           # ri >= 0
    return np.array(cons)

def pack_constraint_non_overlap(x, n):
    # Vectorized non-overlap constraint, much faster for SLSQP
    coords = x.reshape((n, 3))
    xy = coords[:, :2]
    r = coords[:, 2]
    # Only upper triangle (i<j)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dists = np.linalg.norm(xy[idx_i] - xy[idx_j], axis=1)
    min_sep = r[idx_i] + r[idx_j]
    return dists - min_sep

def _jittered_grid_initial(n, seed):
    rng = np.random.default_rng(seed)
    grid_size = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0.08, 0.92, grid_size)
    ys = np.linspace(0.08, 0.92, grid_size)
    coords = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    rng.shuffle(coords)
    coords = coords[:n]
    r0 = rng.uniform(0.07, 0.16, size=n)
    x0 = coords[:,0] + rng.uniform(-0.015, 0.015, size=n)
    y0 = coords[:,1] + rng.uniform(-0.015, 0.015, size=n)
    x0 = np.clip(x0, 0.01, 0.99)
    y0 = np.clip(y0, 0.01, 0.99)
    vars0 = np.zeros(3*n)
    for i in range(n):
        vars0[3*i:3*i+3] = [x0[i], y0[i], r0[i]]
    return vars0

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square, maximizing the sum of radii.
    Robust hybrid: physics-based/jittered multi-start, SLSQP, accepts feasible failures, and multi-polish refinement.
    """
    n = 32
    # Use more seeds for broader search (from best inspiration)
    seeds = [42, 123, 7, 99, 314, 2718, 2023, 888, 0, 5555, 77, 333, 1111, 2022, 404, 23, 45, 777, 8008, 17]
    best_sum = -np.inf
    best_circles = None

    for run_idx, seed in enumerate(seeds):
        if run_idx % 2 == 0:
            # Physics-based: triangular grid, force relax, local grow
            np.random.seed(seed)
            centers, r = triangular_grid_circle_centers(n)
            if centers is None:
                grid_n = int(np.ceil(np.sqrt(n)))
                grid = np.linspace(0.05, 0.95, grid_n)
                centers = [(x, y) for x in grid for y in grid][:n]
                r = 0.03
            circles = np.array([[x, y, r] for x, y in centers])
            circles[:,:2] += np.random.uniform(-0.02,0.02, circles[:,:2].shape)
            circles[:,:2] = np.clip(circles[:,:2], 0.01, 0.99)
            circles = force_directed_relax(circles, None)
            circles = local_grow(circles, None)
            x0 = circles.flatten()
        else:
            x0 = _jittered_grid_initial(n, seed=seed)

        def _nonoverlap_constraint_vec(vars, n=n):
            coords = vars.reshape((n, 3))
            xy = coords[:, :2]
            r = coords[:, 2]
            idx_i, idx_j = np.triu_indices(n, k=1)
            dists = np.linalg.norm(xy[idx_i] - xy[idx_j], axis=1)
            min_sep = r[idx_i] + r[idx_j]
            return dists - min_sep

        def _containment_constraint_vec(vars, n=n):
            coords = vars.reshape((n,3))
            x = coords[:,0]
            y = coords[:,1]
            r = coords[:,2]
            return np.concatenate([
                x - r,
                y - r,
                1 - (x + r),
                1 - (y + r),
                r - 0.01
            ])

        bnds = []
        for i in range(n):
            bnds += [(0,1), (0,1), (0.01,0.21)]
        bounds = Bounds(*zip(*bnds))
        nonoverlap = NonlinearConstraint(_nonoverlap_constraint_vec, 0, np.inf)
        containment = NonlinearConstraint(_containment_constraint_vec, 0, np.inf)
        def _objective(vars):
            return -np.sum(vars[2::3])

        res = minimize(_objective, x0, method='SLSQP',
                       constraints=[nonoverlap, containment], bounds=bounds,
                       options={'ftol':1e-8, 'maxiter':4000, 'disp': False})

        # Accept solution if SLSQP succeeded, or if feasible even if failed
        candidate = None
        if res.success:
            candidate = res.x.reshape((n,3))
        else:
            # Check feasibility with tighter tolerance
            x = res.x
            overlap_ok = np.all(_nonoverlap_constraint_vec(x) >= -1e-10)
            contain_ok = np.all(_containment_constraint_vec(x) >= -1e-10)
            if overlap_ok and contain_ok:
                candidate = x.reshape((n,3))
        if candidate is not None:
            candidate[:,0] = np.clip(candidate[:,0], candidate[:,2], 1 - candidate[:,2])
            candidate[:,1] = np.clip(candidate[:,1], candidate[:,2], 1 - candidate[:,2])
            candidate[:,2] = np.clip(candidate[:,2], 0.01, 0.2)
            total = candidate[:,2].sum()
            if total > best_sum:
                best_sum = total
                best_circles = candidate.copy()

    # --- Final aggressive SLSQP local polish: 8 runs with stronger jitter ---
    if best_circles is not None:
        x0 = best_circles.flatten()
        def _nonoverlap_constraint_vec(vars, n=n):
            coords = vars.reshape((n, 3))
            xy = coords[:, :2]
            r = coords[:, 2]
            idx_i, idx_j = np.triu_indices(n, k=1)
            dists = np.linalg.norm(xy[idx_i] - xy[idx_j], axis=1)
            min_sep = r[idx_i] + r[idx_j]
            return dists - min_sep
        def _containment_constraint_vec(vars, n=n):
            coords = vars.reshape((n,3))
            x = coords[:,0]
            y = coords[:,1]
            r = coords[:,2]
            return np.concatenate([
                x - r,
                y - r,
                1 - (x + r),
                1 - (y + r),
                r - 0.01
            ])
        bnds = []
        for i in range(n):
            bnds += [(0,1), (0,1), (0.01,0.21)]
        bounds = Bounds(*zip(*bnds))
        nonoverlap = NonlinearConstraint(_nonoverlap_constraint_vec, 0, np.inf)
        containment = NonlinearConstraint(_containment_constraint_vec, 0, np.inf)
        def _objective(vars):
            return -np.sum(vars[2::3])

        num_polish = 8
        polish_best = best_circles.copy()
        polish_sum = best_sum
        for polish in range(num_polish):
            if polish == 0:
                xstart = x0.copy()
            else:
                rng = np.random.default_rng(101 + polish)
                xstart = x0 + rng.normal(0, 3e-3, size=x0.shape)
            res = minimize(_objective, xstart, method='SLSQP',
                           constraints=[nonoverlap, containment], bounds=bounds,
                           options={'ftol':1e-10, 'maxiter':9000, 'disp': False})
            candidate = None
            if res.success:
                candidate = res.x.reshape((n,3))
            else:
                x = res.x
                overlap_ok = np.all(_nonoverlap_constraint_vec(x) >= -1e-10)
                contain_ok = np.all(_containment_constraint_vec(x) >= -1e-10)
                if overlap_ok and contain_ok:
                    candidate = x.reshape((n,3))
            if candidate is not None:
                candidate[:,0] = np.clip(candidate[:,0], candidate[:,2], 1 - candidate[:,2])
                candidate[:,1] = np.clip(candidate[:,1], candidate[:,2], 1 - candidate[:,2])
                candidate[:,2] = np.clip(candidate[:,2], 0.01, 0.2)
                t = candidate[:,2].sum()
                if t > polish_sum:
                    polish_sum = t
                    polish_best = candidate.copy()
        best_sum = polish_sum
        best_circles = polish_best.copy()

    if best_circles is None:
        best_circles = np.zeros((n,3))
    best_circles = best_circles[np.argsort(-best_circles[:,2])]
    return best_circles


# EVOLVE-BLOCK-END
