# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

# EVOLVE: Hybrid Voronoi-based initialization + global optimization.
# This algorithm first seeds 32 points using a low-discrepancy (Sobol) sequence for good initial spread,
# then constructs Voronoi regions for each seed.
# For each region, the maximum inscribed circle is placed at the centroid, with radius as large as possible
# (while contained in square and not overlapping neighbors).
# Then, a constrained global optimization (L-BFGS-B) refines (x,y,r) for each circle jointly to maximize sum(radii),
# subject to containment and non-overlap constraints.

# Helper: Sobol sequence for initial seeding
try:
    from scipy.stats import qmc
    def sobol_points(n, d, seed=42):
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        return sampler.random(n)
except ImportError:
    def sobol_points(n, d, seed=42):
        np.random.seed(seed)
        return np.random.rand(n, d)

def voronoi_centroids(points):
    # Compute Voronoi regions and find centroid for each region inside unit square
    vor = Voronoi(points)
    centroids = np.zeros_like(points)
    for i, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if -1 in vertices or len(vertices) == 0:
            # Use original point if region is unbounded
            centroids[i] = points[i]
            continue
        v = np.array([vor.vertices[j] for j in vertices])
        # Clip to unit square
        v = np.clip(v, 0, 1)
        centroids[i] = np.mean(v, axis=0)
    return centroids

# Overlap-aware greedy sequential initial radii assignment (from inspiration programs)
def initial_radii(points, min_dist=0.0125):
    n = points.shape[0]
    radii = np.zeros(n)
    def max_radius(x, y, circles):
        # Containment
        r_contain = min(x, 1 - x, y, 1 - y)
        # Non-overlap
        r_nonoverlap = r_contain
        for (xi, yi, ri) in circles:
            dist = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
            r_nonoverlap = min(r_nonoverlap, dist - ri)
        return max(min_dist, r_nonoverlap)
    circles = np.zeros((n, 3))
    circles[:, :2] = points
    for i in range(n):
        circles[i, 2] = max_radius(circles[i, 0], circles[i, 1], circles[:i])
        if circles[i, 2] < min_dist:
            circles[i, 2] = min_dist
    return circles[:,2]

def constraint_fun(x):
    n = 32
    # x: [x0,y0,r0, x1,y1,r1, ...]
    cons = []
    for i in range(n):
        xi, yi, ri = x[3*i:3*i+3]
        # Containment
        cons.append(xi - ri)      # xi >= ri  (xi - ri >= 0)
        cons.append(1 - xi - ri)  # xi <= 1-ri (1 - xi - ri >= 0)
        cons.append(yi - ri)      # yi >= ri
        cons.append(1 - yi - ri)  # yi <= 1-ri
        cons.append(ri)           # ri >= 0
    # Non-overlap
    for i in range(n):
        xi, yi, ri = x[3*i:3*i+3]
        for j in range(i+1, n):
            xj, yj, rj = x[3*j:3*j+3]
            dij = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            cons.append(dij - (ri + rj + 1e-6)) # dij >= ri + rj + epsilon (tightened)
    return np.array(cons)

def optimize_circles(init_xyz):
    n = 32
    def neg_sum_radii(x):
        return -np.sum(x[2::3])
    bounds = []
    for i in range(n):
        # x,y in [0,1], r in [0, 0.5]
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    cons = ({'type': 'ineq', 'fun': constraint_fun})
    result = minimize(neg_sum_radii, init_xyz, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol':1e-6,'maxiter':300})
    if not result.success:
        # fallback: use initial guess
        return init_xyz.reshape(n,3)
    return result.x.reshape(n,3)

# --- EVOLVE: Hybrid Lattice-Voronoi-Perturbation + Multi-Start SLSQP ---
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    seed = 42
    start_time = time.time()
    np.random.seed(seed)

    # Step 1: Hybrid Lattice + Sobol + Hexagonal + Voronoi-Lloyd initialization
    # We create several candidate initial layouts:
    #   - 1: Sobol points (as before)
    #   - 2: Hexagonal grid fitting 32 points
    #   - 3: Slightly perturbed lattice grid
    #   - 4: Random uniform (for diversity)
    #   - 5: Voronoi centroid-perturbed
    #   - 6: Lloyd's centroidal Voronoi refinement (from inspiration programs)

    candidates = []

    # 1. Sobol points
    points_sobol = sobol_points(n, 2, seed=seed)
    candidates.append(np.copy(points_sobol))

    # 2. Hexagonal grid fitting 32 points in [0,1]^2
    def hex_grid_points(n, margin=0.02):
        # Find rows and cols so rows*cols >= n, favor close-to-square
        best = None
        for rows in range(4, n+1):
            cols = int(np.ceil(n / rows))
            if rows*cols >= n:
                aspect = abs(rows - cols)
                if best is None or aspect < best[0]:
                    best = (aspect, rows, cols)
        _, rows, cols = best
        dx = (1 - 2*margin) / (cols - 1 if cols > 1 else 1)
        dy = (1 - 2*margin) / (rows - 1 if rows > 1 else 1)
        points = []
        for i in range(rows):
            y = margin + i*dy
            for j in range(cols):
                x = margin + j*dx + (dx/2)*(i%2)  # offset every other row
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y])
                if len(points) == n:
                    return np.array(points)
        return np.array(points)[:n]
    points_hex = hex_grid_points(n, margin=0.04)
    candidates.append(points_hex)

    # 3. Perturbed lattice grid
    grid_side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.linspace(0.05, 0.95, grid_side), np.linspace(0.05, 0.95, grid_side))
    grid_points = np.vstack([gx.ravel(), gy.ravel()]).T[:n]
    perturb = (np.random.rand(n,2)-0.5)*0.08
    points_grid = np.clip(grid_points + perturb, 0, 1)
    candidates.append(points_grid)

    # 4. Pure random uniform
    points_rand = np.random.rand(n,2)
    candidates.append(points_rand)

    # 5. Voronoi centroid-perturbed
    points_voronoi = voronoi_centroids(points_sobol)
    candidates.append(points_voronoi)

    # 6. Lloyd's centroidal Voronoi refinement (from inspiration programs)
    def lloyd_voronoi(points, max_iter=10):
        pts = np.copy(points)
        for _ in range(max_iter):
            vor = Voronoi(pts)
            new_pts = []
            for i, p in enumerate(pts):
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    new_pts.append(p)
                    continue
                poly = np.array([vor.vertices[j] for j in region])
                poly = np.clip(poly, 0, 1)
                if len(poly) < 3:
                    new_pts.append(p)
                else:
                    centroid = np.mean(poly, axis=0)
                    centroid = np.clip(centroid, 0, 1)
                    new_pts.append(centroid)
            pts = np.array(new_pts)
        return pts
    points_lloyd = lloyd_voronoi(points_sobol, max_iter=10)
    candidates.append(points_lloyd)

    # For each candidate: estimate initial radii and create initial guess
    def max_radius(x, y, circles):
        # Containment
        r_contain = min(x, 1 - x, y, 1 - y)
        # Non-overlap
        r_nonoverlap = r_contain
        for (xi, yi, ri) in circles:
            dist = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
            r_nonoverlap = min(r_nonoverlap, dist - ri)
        return max(0.0125, r_nonoverlap)

    init_xyzs = []
    for cand in candidates:
        # Improved initial radii: sequential greedy non-overlap
        circles = np.zeros((n, 3))
        circles[:, :2] = cand
        for i in range(n):
            circles[i, 2] = max_radius(circles[i, 0], circles[i, 1], circles[:i])
        xyz = circles.flatten()
        init_xyzs.append(xyz)

    # Add a centroidal Voronoi-improved candidate from the *best* initial guess so far (inspired by Inspiration 1)
    # This further increases diversity
    best_init_idx = np.argmax([np.sum(x[2::3]) for x in init_xyzs])
    best_seed = init_xyzs[best_init_idx].reshape(n, 3)[:, :2]
    # One round of centroidal Voronoi
    centroids = voronoi_centroids(best_seed)
    circles = np.zeros((n, 3))
    circles[:, :2] = centroids
    for i in range(n):
        circles[i, 2] = max_radius(circles[i, 0], circles[i, 1], circles[:i])
    xyz = circles.flatten()
    init_xyzs.append(xyz)

    # Step 2: Multi-start SLSQP global constrained optimization for all circles jointly
    best_circles = None
    best_sum = -np.inf
    candidate_results = []
    for idx, init_xyz in enumerate(init_xyzs):
        # Early exit if time is nearly up
        if time.time() - start_time > 55.0:
            break
        try:
            circles_opt = optimize_circles(init_xyz)
            s = np.sum(circles_opt[:,2])
            candidate_results.append((s, circles_opt))
            if s > best_sum:
                best_sum = s
                best_circles = circles_opt
        except Exception:
            continue

    # Take top 3 candidates for further local refinement, if time allows
    candidate_results.sort(reverse=True, key=lambda x: x[0])
    refined_best = None
    refined_best_sum = best_sum
    for s, circles in candidate_results[:3]:
        if time.time() - start_time > 57.0:
            break
        # Refine with more SLSQP iterations, tighter tolerance
        def neg_sum_radii(x):
            return -np.sum(x[2::3])
        nvar = 3*n
        bounds = []
        for i in range(n):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        cons = ({'type': 'ineq', 'fun': constraint_fun})
        try:
            # Allow more iterations for final refinement, but still watch time
            result = minimize(neg_sum_radii, circles.flatten(), bounds=bounds, constraints=cons, method='SLSQP',
                              options={'ftol':1e-10, 'maxiter':1500})
            if result.success:
                xopt = result.x.reshape(n,3)
                # If any radii < 0, reset to minimal
                xopt[:,2] = np.maximum(xopt[:,2], 0.0125)
                sopt = np.sum(xopt[:,2])
                if sopt > refined_best_sum:
                    refined_best_sum = sopt
                    refined_best = xopt
        except Exception:
            continue

    # Greedy local radius maximization for "last-mile" improvement
    def greedy_local_max_radii(circles, max_iter=7):
        # circles: (n,3)
        for it in range(max_iter):
            for idx in np.random.permutation(n):
                xi, yi, ri = circles[idx]
                max_r = min(xi, 1-xi, yi, 1-yi)
                for j in range(n):
                    if j==idx: continue
                    xj, yj, rj = circles[j]
                    dij = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    max_r = min(max_r, dij - rj - 1e-6)
                circles[idx,2] = max(0.0125, max_r)
        return circles

    final_circles = None
    if refined_best is not None:
        final_circles = refined_best
    else:
        final_circles = best_circles

    # Final greedy local pass (increase iterations for last-mile improvement)
    final_circles = greedy_local_max_radii(final_circles, max_iter=7)

    # Extra: shuffle and one more greedy pass if time remains
    if time.time() - start_time < 57.5:
        np.random.shuffle(final_circles)
        final_circles = greedy_local_max_radii(final_circles, max_iter=2)

        # Add another centroidal Voronoi candidate using the best configuration so far (from Inspiration 1)
        best_config = final_circles[:, :2]
        centroids2 = voronoi_centroids(best_config)
        circles2 = np.zeros((n, 3))
        circles2[:, :2] = centroids2
        for i in range(n):
            circles2[i, 2] = max_radius(circles2[i, 0], circles2[i, 1], circles2[:i])
        try:
            if time.time() - start_time < 58.5:
                opt_circles2 = optimize_circles(circles2.flatten())
                # Run a greedy pass after SLSQP for last-mile improvement
                opt_circles2 = greedy_local_max_radii(opt_circles2, max_iter=2)
                if np.sum(opt_circles2[:, 2]) > np.sum(final_circles[:, 2]):
                    final_circles = opt_circles2
        except Exception:
            pass

    # Fallback: use best initial guess if optimization fails
    if final_circles is None or np.isnan(final_circles).any():
        final_circles = init_xyzs[0].reshape(n,3)

    # If optimization took too long, fallback to best initial guess
    if time.time() - start_time > 59.0:
        final_circles = init_xyzs[0].reshape(n,3)

    # --- POST-GREEDY CENTROIDAL VORONOI REFINEMENT (multi-inspiration) ---
    if time.time() - start_time < 58.5:
        # Centroidal Voronoi relaxation (2 rounds) starting from best so far
        best_config2 = np.copy(final_circles[:, :2])
        for _ in range(2):
            best_config2 = voronoi_centroids(best_config2)
        circles_post = np.zeros((n, 3))
        circles_post[:, :2] = best_config2
        for i in range(n):
            circles_post[i, 2] = max_radius(circles_post[i, 0], circles_post[i, 1], circles_post[:i])
        try:
            opt_post = optimize_circles(circles_post.flatten())
            opt_post = greedy_local_max_radii(opt_post, max_iter=2)
            if np.sum(opt_post[:, 2]) > np.sum(final_circles[:, 2]):
                final_circles = opt_post
        except Exception:
            pass

        # --- EXTRA: One more centroidal Voronoi relaxation round if time allows ---
        if time.time() - start_time < 59.0:
            best_config3 = voronoi_centroids(final_circles[:, :2])
            circles_post2 = np.zeros((n, 3))
            circles_post2[:, :2] = best_config3
            for i in range(n):
                circles_post2[i, 2] = max_radius(circles_post2[i, 0], circles_post2[i, 1], circles_post2[:i])
            try:
                opt_post2 = optimize_circles(circles_post2.flatten())
                opt_post2 = greedy_local_max_radii(opt_post2, max_iter=2)
                if np.sum(opt_post2[:, 2]) > np.sum(final_circles[:, 2]):
                    final_circles = opt_post2
            except Exception:
                pass

        # --- EXTRA: Shuffle and perturb candidate (from inspiration) ---
        if time.time() - start_time < 59.2:
            perturbed = np.copy(final_circles[:, :2])
            np.random.shuffle(perturbed)
            perturbed += (np.random.rand(n,2)-0.5)*0.04
            perturbed = np.clip(perturbed, 0, 1)
            circles_pert = np.zeros((n, 3))
            circles_pert[:, :2] = perturbed
            for i in range(n):
                circles_pert[i, 2] = max_radius(circles_pert[i, 0], circles_pert[i, 1], circles_pert[:i])
            try:
                opt_pert = optimize_circles(circles_pert.flatten())
                opt_pert = greedy_local_max_radii(opt_pert, max_iter=2)
                if np.sum(opt_pert[:, 2]) > np.sum(final_circles[:, 2]):
                    final_circles = opt_pert
            except Exception:
                pass

        # --- EXTRA: Symmetry-flipped candidate for further diversity ---
        if time.time() - start_time < 59.4:
            for flipx, flipy in [(True, False), (False, True), (True, True)]:
                flipped = np.copy(final_circles[:, :2])
                if flipx:
                    flipped[:,0] = 1.0 - flipped[:,0]
                if flipy:
                    flipped[:,1] = 1.0 - flipped[:,1]
                circles_flip = np.zeros((n, 3))
                circles_flip[:, :2] = flipped
                for i in range(n):
                    circles_flip[i, 2] = max_radius(circles_flip[i, 0], circles_flip[i, 1], circles_flip[:i])
                try:
                    opt_flip = optimize_circles(circles_flip.flatten())
                    opt_flip = greedy_local_max_radii(opt_flip, max_iter=2)
                    if np.sum(opt_flip[:, 2]) > np.sum(final_circles[:, 2]):
                        final_circles = opt_flip
                except Exception:
                    pass

    # Robust final feasibility check: ensure no NaNs, negative radii, or overlaps
    def check_feasibility(circles):
        if circles is None or np.isnan(circles).any():
            return False
        # Radii non-negative
        if np.any(circles[:,2] < 0):
            return False
        # Containment
        if np.any(circles[:,0] < circles[:,2]) or np.any(circles[:,0] > 1 - circles[:,2]):
            return False
        if np.any(circles[:,1] < circles[:,2]) or np.any(circles[:,1] > 1 - circles[:,2]):
            return False
        # Non-overlap
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i+1, n):
                xj, yj, rj = circles[j]
                dij = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dij < ri + rj - 1e-7:
                    return False
        return True

    if not check_feasibility(final_circles):
        final_circles = init_xyzs[0].reshape(n,3)

    # Final Lloyd relaxation (inspired by both programs, post all greedy/CVT)
    if time.time() - start_time < 59.5:
        def lloyd_voronoi(points, max_iter=1):
            pts = np.copy(points)
            for _ in range(max_iter):
                vor = Voronoi(pts)
                new_pts = []
                for i, p in enumerate(pts):
                    region_idx = vor.point_region[i]
                    region = vor.regions[region_idx]
                    if -1 in region or len(region) == 0:
                        new_pts.append(p)
                        continue
                    poly = np.array([vor.vertices[j] for j in region])
                    poly = np.clip(poly, 0, 1)
                    if len(poly) < 3:
                        new_pts.append(p)
                    else:
                        centroid = np.mean(poly, axis=0)
                        centroid = np.clip(centroid, 0, 1)
                        new_pts.append(centroid)
                pts = np.array(new_pts)
            return pts

        best_feasible_circles = np.copy(final_circles)
        best_feasible_sum = np.sum(final_circles[:,2]) if check_feasibility(final_circles) else -np.inf

        lloyd_points = lloyd_voronoi(final_circles[:, :2], max_iter=1)
        circles_lloyd = np.zeros((n, 3))
        circles_lloyd[:, :2] = lloyd_points
        for i in range(n):
            circles_lloyd[i, 2] = max_radius(circles_lloyd[i, 0], circles_lloyd[i, 1], circles_lloyd[:i])
        try:
            opt_lloyd = optimize_circles(circles_lloyd.flatten())
            opt_lloyd = greedy_local_max_radii(opt_lloyd, max_iter=2)
            if check_feasibility(opt_lloyd):
                sum_lloyd = np.sum(opt_lloyd[:, 2])
                if sum_lloyd > best_feasible_sum:
                    best_feasible_circles = opt_lloyd
                    best_feasible_sum = sum_lloyd
        except Exception:
            pass

        final_circles = best_feasible_circles

    return final_circles


# EVOLVE-BLOCK-END
