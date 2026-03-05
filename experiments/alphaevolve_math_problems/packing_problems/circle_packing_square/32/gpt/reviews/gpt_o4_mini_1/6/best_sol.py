# EVOLVE-BLOCK-START
import numpy as np
import random
from numba import njit

def circle_packing32() -> np.ndarray:
    """
    Hybrid circle packing for n=32 using three distinct modules:
      1) KMeans-based Voronoi seeding with vectorized clearance.
      2) Combinatorial CP-SAT radius refinement.
      3) Continuous gradient-based local smoothing via SLSQP.
    """
    random.seed(42)
    np.random.seed(42)
    n = 32

    # Initialize circles with uniform random centers and small constant radius
    circles = np.zeros((n, 3))
    circles[:, :2] = np.random.rand(n, 2)
    circles[:, 2] = 0.01

    # Module 1: KMeans Voronoi Seeding + Local Hill‐Climbing + Force Relaxation
    from sklearn.cluster import KMeans

    @njit(nogil=True)
    def compute_clearance(x, y, circles):
        """
        Compute maximum feasible radius at (x, y) given existing circles.
        Vectorized implementation for speed (compiled with numba).
        """
        # boundary clearance
        r_bound = min(x, 1 - x, y, 1 - y)
        if circles.shape[0] > 0:
            # compute distance to each existing circle edge
            dxy0 = circles[:, 0] - x
            dxy1 = circles[:, 1] - y
            # numpy.hypot not supported in nopython mode; use manual sqrt
            min_dist = 1e9
            for i in range(circles.shape[0]):
                dist = (dxy0[i] * dxy0[i] + dxy1[i] * dxy1[i])**0.5 - circles[i, 2]
                if dist < min_dist:
                    min_dist = dist
            if min_dist < r_bound:
                r_bound = min_dist
        return r_bound if r_bound > 0.0 else 0.0

    # Seed with KMeans centers and initial radii
    M = 20000
    pts = np.random.rand(M, 2)
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(pts)
    centers = kmeans.cluster_centers_
    circles = []
    for cx, cy in centers:
        db = min(cx, 1 - cx, cy, 1 - cy)
        dists = np.linalg.norm(centers - [cx, cy], axis=1)
        dn = np.min(dists[dists > 0]) / 2.0
        r0 = min(db, dn)
        circles.append([cx, cy, r0])
    circles = np.array(circles)

    # Randomized hill-climbing (pos + radius)
    for _ in range(4000):
        idx = np.random.randint(n)
        x_old, y_old, r_old = circles[idx]
        x_new = np.clip(x_old + np.random.randn() * 0.02, 0, 1)
        y_new = np.clip(y_old + np.random.randn() * 0.02, 0, 1)
        r_new = compute_clearance(x_new, y_new, np.delete(circles, idx, axis=0))
        if r_new > r_old:
            circles[idx] = [x_new, y_new, r_new]

    # Radius-only smoothing
    for _ in range(2000):
        idx = np.random.randint(n)
        x, y, r_old = circles[idx]
        r_new = compute_clearance(x, y, np.delete(circles, idx, axis=0))
        if r_new > r_old:
            circles[idx, 2] = r_new

    # Force-based relaxation with adaptive step to redistribute and grow
    T = 500
    alpha0, alpha_min = 0.02, 0.001
    for t in range(T):
        alpha = alpha0 * (1 - t / T) + alpha_min * (t / T)
        disp = np.zeros((n, 2))
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dx, dy = xi - xj, yi - yj
                dist = np.hypot(dx, dy) + 1e-8
                overlap = ri + rj - dist
                if overlap > 0:
                    dirx, diry = dx / dist, dy / dist
                    disp[i] += [alpha * overlap * dirx, alpha * overlap * diry]
                    disp[j] -= [alpha * overlap * dirx, alpha * overlap * diry]
        circles[:, :2] += disp
        # Clip positions and recompute radii
        for i in range(n):
            x, y, _ = circles[i]
            circles[i, 0] = np.clip(x, circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(y, circles[i, 2], 1 - circles[i, 2])
            circles[i, 2] = compute_clearance(circles[i, 0], circles[i, 1],
                                              np.delete(circles, i, axis=0))
    # end Module 1

    # Module 2: Combinatorial Radius Refinement via OR-Tools CP-SAT
    from ortools.sat.python import cp_model
    from itertools import combinations

    def cp_sat_refine(circles, scale=2000, time_limit=5):
        """
        Module 2 (Combinatorial):
        Fix circle centers and optimize radii with discrete CP-SAT.
        Radii are integer variables scaled by 'scale'.
        """
        num = circles.shape[0]
        model = cp_model.CpModel()
        # Integer radius variables
        r_vars = [model.NewIntVar(0, scale, f"r_{i}") for i in range(num)]
        # Boundary constraints
        for i, (cx, cy, _) in enumerate(circles):
            bound = int(min(cx, cy, 1 - cx, 1 - cy) * scale)
            model.Add(r_vars[i] <= bound)
        # Non-overlap constraints
        for i, j in combinations(range(num), 2):
            xi, yi, _ = circles[i]
            xj, yj, _ = circles[j]
            # Precomputed center distance
            d = int(np.hypot(xi - xj, yi - yj) * scale)
            model.Add(r_vars[i] + r_vars[j] <= d)
        # Maximize total sum of radii
        model.Maximize(sum(r_vars))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(num):
                circles[i, 2] = solver.Value(r_vars[i]) / scale
        return circles

    circles = cp_sat_refine(circles)

    # Module 3: Local Gradient-based Smooth Refinement (SLSQP)
    from scipy.optimize import minimize

    def local_smooth(circles, maxiter=100):
        """
        Module 3 (Continuous/Gradient):
        Perform a local SLSQP optimize to smooth and slightly adjust positions and radii.
        """
        num = circles.shape[0]
        x0 = circles.flatten()

        # Constraints for containment and non-overlap
        cons = []
        # Boundary and non-negativity
        for i in range(num):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        # Pairwise non-overlap
        for i, j in combinations(range(num), 2):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.hypot(x[3*i] - x[3*j], x[3*i+1] - x[3*j+1]) - (x[3*i+2] + x[3*j+2])
            })

        # Objective: maximize sum of radii -> minimize negative sum
        def obj(x):
            r = x[2::3]
            return -np.sum(r)

        res = minimize(obj, x0, method='SLSQP', constraints=cons, options={'maxiter': maxiter})
        if res.success:
            return res.x.reshape((num, 3))
        else:
            return circles

    # Smooth with local gradient refine
    final = local_smooth(circles)
    return final


# EVOLVE-BLOCK-END
