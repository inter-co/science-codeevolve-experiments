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
    Hybrid method: combines physics-based initialization, jittered grid multi-start, and aggressive SLSQP local refinement.
    """
    n = 32
    # List of seeds for multi-start (drawn from Inspiration 1 and 2)
    seeds = [42, 123, 7, 99, 314, 2718, 2023, 888, 0, 5555, 77, 333, 1111, 2022]
    best_sum = -np.inf
    best_circles = None

    # --- Multi-start: hybrid physics (triangular grid + repulsion) and jittered grid ---
    for run_idx, seed in enumerate(seeds):
        # Alternate between physics-based and jittered grid starts for diversity
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
            # Jittered grid
            x0 = _jittered_grid_initial(n, seed=seed)

        # Vectorized constraints
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
                x - r,           # x >= r
                y - r,           # y >= r
                1 - (x + r),     # x <= 1-r
                1 - (y + r),     # y <= 1-r
                r - 0.01         # r >= 0.01 (tighten for numerical stability)
            ])

        bnds = []
        for i in range(n):
            bnds += [(0,1), (0,1), (0.01,0.5)]
        bounds = Bounds(*zip(*bnds))
        nonoverlap = NonlinearConstraint(_nonoverlap_constraint_vec, 0, np.inf)
        containment = NonlinearConstraint(_containment_constraint_vec, 0, np.inf)
        def _objective(vars):
            return -np.sum(vars[2::3])
        res = minimize(_objective, x0, method='SLSQP',
                       constraints=[nonoverlap, containment], bounds=bounds,
                       options={'ftol':1e-8, 'maxiter':3000, 'disp': False})
        if res.success:
            out = res.x.reshape((n,3))
            out[:,0] = np.clip(out[:,0], out[:,2], 1 - out[:,2])
            out[:,1] = np.clip(out[:,1], out[:,2], 1 - out[:,2])
            out[:,2] = np.clip(out[:,2], 0.01, 0.5)
            total = out[:,2].sum()
            if total > best_sum:
                best_sum = total
                best_circles = out.copy()

    # --- Final aggressive SLSQP local polish from the best found so far ---
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
            bnds += [(0,1), (0,1), (0.01,0.5)]
        bounds = Bounds(*zip(*bnds))
        nonoverlap = NonlinearConstraint(_nonoverlap_constraint_vec, 0, np.inf)
        containment = NonlinearConstraint(_containment_constraint_vec, 0, np.inf)
        def _objective(vars):
            return -np.sum(vars[2::3])
        res = minimize(_objective, x0, method='SLSQP',
                       constraints=[nonoverlap, containment], bounds=bounds,
                       options={'ftol':1e-10, 'maxiter':6000, 'disp': False})
        # Only update if sum improved
        if res.success and -_objective(res.x) > best_sum:
            out = res.x.reshape((n,3))
            out[:,0] = np.clip(out[:,0], out[:,2], 1 - out[:,2])
            out[:,1] = np.clip(out[:,1], out[:,2], 1 - out[:,2])
            out[:,2] = np.clip(out[:,2], 0.01, 0.5)
            best_sum = out[:,2].sum()
            best_circles = out.copy()

    if best_circles is None:
        best_circles = np.zeros((n,3))

    # --- Final greedy local radius growth as in Inspiration 2 ---
    def _greedy_grow(circles, n, passes=4, rmin=0.01, rmax=0.5):
        # Greedily grow each radius as much as possible, one at a time, for several passes (from Inspiration 1)
        c = circles.copy()
        x, y, r = c[:, 0], c[:, 1], c[:, 2]
        for passnum in range(passes):
            idxs = np.arange(n)
            np.random.seed(2024 + passnum)
            np.random.shuffle(idxs)
            for idx in idxs:
                maxr = min(x[idx], 1 - x[idx], y[idx], 1 - y[idx])
                for j in range(n):
                    if j == idx:
                        continue
                    dist = np.hypot(x[idx] - x[j], y[idx] - y[j])
                    maxr = min(maxr, dist - r[j])
                r[idx] = max(rmin, min(maxr, rmax))
            c[:, 2] = r
        # Final safety: clip to bounds
        c[:, 0] = np.clip(c[:, 0], c[:, 2], 1 - c[:, 2])
        c[:, 1] = np.clip(c[:, 1], c[:, 2], 1 - c[:, 2])
        c[:, 2] = np.clip(c[:, 2], rmin, rmax)
        return c

    # Apply final greedy grow for a few passes to improve radii sum
    best_circles = _greedy_grow(best_circles, n, passes=4, rmin=0.01, rmax=0.5)

    # --- Penalty-based squeeze for final improvement (from Inspiration 2) ---
    def penalty_obj(z):
        c = z.reshape((n,3))
        penalty = 0.0
        for i in range(n):
            x_, y_, r_ = c[i]
            # Containment penalty
            if (x_ - r_ < 0) or (x_ + r_ > 1) or (y_ - r_ < 0) or (y_ + r_ > 1) or (r_ < 0.01):
                penalty += 1e3
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = c[i]
                xj, yj, rj = c[j]
                dist = np.hypot(xi - xj, yi - yj)
                if dist < ri + rj - 1e-8:
                    penalty += 1e2 * (ri + rj - dist)
        return -np.sum(c[:,2]) + penalty

    bnds = []
    for i in range(n):
        bnds += [(0,1), (0,1), (0.01,0.5)]
    bounds = Bounds(*zip(*bnds))
    from scipy.optimize import minimize as _lbfgs_minimize
    res2 = _lbfgs_minimize(
        penalty_obj, best_circles.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter':120, 'disp': False}
    )
    if res2.success:
        arr = res2.x.reshape((n,3))
        best_circles[:] = arr
        # Robust containment and radii clipping
        best_circles[:,0] = np.clip(best_circles[:,0], best_circles[:,2], 1 - best_circles[:,2])
        best_circles[:,1] = np.clip(best_circles[:,1], best_circles[:,2], 1 - best_circles[:,2])
        best_circles[:,2] = np.clip(best_circles[:,2], 0.01, 0.5)

    # --- Fallback: physics-inspired grow-and-relax if best_circles are degenerate ---
    if np.sum(best_circles[:,2]) < 0.05:
        def grow_and_relax_circles(
            n=32, n_grow_steps=3000, grow_step=0.0015, move_step=0.008, edge_margin=None, random_seed=42
        ):
            np.random.seed(random_seed)
            circles = np.zeros((n, 3))
            circles[:, 0:2] = np.random.uniform(0.1, 0.9, size=(n, 2))
            circles[:, 2] = 0.01  # Small initial radii

            for step in range(n_grow_steps):
                # Grow all circles
                circles[:,2] += grow_step

                # Containment (stay inside box)
                for i in range(n):
                    x, y, r = circles[i]
                    if x - r < 0:
                        circles[i,0] = r
                    if x + r > 1:
                        circles[i,0] = 1 - r
                    if y - r < 0:
                        circles[i,1] = r
                    if y + r > 1:
                        circles[i,1] = 1 - r

                # Overlap resolution (robust)
                for i in range(n):
                    xi, yi, ri = circles[i]
                    for j in range(i+1, n):
                        xj, yj, rj = circles[j]
                        dx, dy = xj - xi, yj - yi
                        dist = np.hypot(dx, dy)
                        min_dist = ri + rj
                        if dist < min_dist - 1e-12:
                            if dist < 1e-12:
                                angle = np.random.uniform(0, 2*np.pi)
                                dx, dy = np.cos(angle), np.sin(angle)
                                dist = 1e-3
                            shift = (min_dist - dist) / 2.0 + 1e-6
                            norm = np.array([dx, dy]) / (dist + 1e-12)
                            circles[i,0:2] -= norm * shift * move_step
                            circles[j,0:2] += norm * shift * move_step
                            for idx in [i, j]:
                                x, y, r = circles[idx]
                                if x - r < 0:
                                    circles[idx,0] = r
                                if x + r > 1:
                                    circles[idx,0] = 1 - r
                                if y - r < 0:
                                    circles[idx,1] = r
                                if y + r > 1:
                                    circles[idx,1] = 1 - r
                            overlap = min_dist - dist
                            if overlap > 1e-4:
                                circles[i,2] -= grow_step*0.1
                                circles[j,2] -= grow_step*0.1
                                circles[i,2] = max(circles[i,2], 0.001)
                                circles[j,2] = max(circles[j,2], 0.001)

                # Prevent circles from growing beyond containment
                for i in range(n):
                    x, y, r = circles[i]
                    max_r = min(x, 1-x, y, 1-y)
                    if r > max_r:
                        circles[i,2] = max_r - 1e-8

                min_room = np.min([
                    min(circles[i,0], 1-circles[i,0], circles[i,1], 1-circles[i,1]) - circles[i,2]
                    for i in range(n)
                ])
                overlap = False
                for i in range(n):
                    for j in range(i+1, n):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist = np.hypot(xi - xj, yi - yj)
                        if dist < ri + rj - 1e-7:
                            overlap = True
                if not overlap and min_room < 2e-4:
                    break
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    dist = np.hypot(xi - xj, yi - yj)
                    min_dist = ri + rj
                    if dist < min_dist:
                        overlap = (min_dist - dist) / 2
                        circles[i,2] -= overlap
                        circles[j,2] -= overlap
            circles[:,2] = np.clip(circles[:,2], 0.001, 0.5)
            circles[:,0] = np.clip(circles[:,0], circles[:,2], 1-circles[:,2])
            circles[:,1] = np.clip(circles[:,1], circles[:,2], 1-circles[:,2])
            return circles
        best_circles = grow_and_relax_circles(n=n, random_seed=2024)
        best_circles = _greedy_grow(best_circles, n, passes=4, rmin=0.01, rmax=0.5)
        # Penalty squeeze after fallback
        res2 = _lbfgs_minimize(
            penalty_obj, best_circles.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter':120, 'disp': False}
        )
        if res2.success:
            arr = res2.x.reshape((n,3))
            best_circles[:] = arr
            best_circles[:,0] = np.clip(best_circles[:,0], best_circles[:,2], 1 - best_circles[:,2])
            best_circles[:,1] = np.clip(best_circles[:,1], best_circles[:,2], 1 - best_circles[:,2])
            best_circles[:,2] = np.clip(best_circles[:,2], 0.01, 0.5)

    best_circles = best_circles[np.argsort(-best_circles[:,2])]
    return best_circles


# EVOLVE-BLOCK-END
