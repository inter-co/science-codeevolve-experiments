# EVOLVE-BLOCK-START
import numpy as np
import random

def circle_packing32() -> np.ndarray:
    """
    Hybrid circle packing for n=32 using two distinct modules:
      1) Physics-based force relaxation to spread and grow circles initially.
      2) Differential Evolution for global fine-tuning of positions and radii.
    """
    random.seed(42)
    np.random.seed(42)
    n = 32

    # Initialize circles with uniform random centers and small constant radius
    circles = np.zeros((n, 3))
    circles[:, :2] = np.random.rand(n, 2)
    circles[:, 2] = 0.01

    # Module 1: KMeans-based Voronoi Seeding and local hill-climbing
    from sklearn.cluster import KMeans

    def compute_clearance(x, y, circles):
        """
        Compute maximum feasible radius at (x,y) given existing circles.
        """
        r = min(x, 1 - x, y, 1 - y)
        for cx, cy, cr in circles:
            d = np.hypot(x - cx, y - cy) - cr
            if d < r:
                r = d
            if r <= 0:
                return 0.0
        return r

    # Seed with KMeans centers and initial radii
    M = 30000  # increased sample size for improved KMeans seeding
    pts = np.random.rand(M, 2)
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(pts)
    centers = kmeans.cluster_centers_
    circles = []
    for cx, cy in centers:
        # Distance to square boundary
        db = min(cx, 1 - cx, cy, 1 - cy)
        # Half nearest-neighbor distance
        dists = np.linalg.norm(centers - [cx, cy], axis=1)
        dn = np.min(dists[dists > 0]) / 2.0
        r0 = min(db, dn)
        circles.append([cx, cy, r0])
    circles = np.array(circles)

    # Local randomized hill-climbing (position + radius)
    for _ in range(5000):  # extended hill-climb iterations for better exploration
        idx = np.random.randint(n)
        x_old, y_old, r_old = circles[idx]
        x_new = np.clip(x_old + np.random.randn() * 0.02, 0, 1)
        y_new = np.clip(y_old + np.random.randn() * 0.02, 0, 1)
        r_new = compute_clearance(x_new, y_new, np.delete(circles, idx, axis=0))
        if r_new > r_old:
            circles[idx] = [x_new, y_new, r_new]
    # Radius-only smoothing
    for _ in range(3000):  # extended radius-only smoothing
        idx = np.random.randint(n)
        x, y, r_old = circles[idx]
        r_new = compute_clearance(x, y, np.delete(circles, idx, axis=0))
        if r_new > r_old:
            circles[idx, 2] = r_new

    # Force-based relaxation to adjust overlaps and grow radii
    alpha = 0.01
    for _ in range(300):  # more relaxation steps
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
        # Clip and update radii
        for i in range(n):
            x, y, _ = circles[i]
            circles[i, 0] = np.clip(x, circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(y, circles[i, 2], 1 - circles[i, 2])
            circles[i, 2] = compute_clearance(circles[i, 0], circles[i, 1],
                                              np.delete(circles, i, axis=0))

    # Module 2: Combinatorial Radius Refinement via OR-Tools CP-SAT
    from ortools.sat.python import cp_model
    from itertools import combinations

    def cp_sat_refine(circles, scale=2000, time_limit=5):  # reduced CP-SAT time to speed up
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

    def local_smooth(circles, maxiter=50):  # reduce SLSQP iterations for runtime savings
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
    final = local_smooth(circles, maxiter=50)  # speedier local refinement

    # Module 4: LP-based radius refinement given final positions (inspired by Inspirations 1 & 2)
    pos = final[:, :2]
    n = pos.shape[0]
    # Compute boundary-based maximum radii
    rad_bound = np.minimum.reduce([pos[:,0], 1 - pos[:,0], pos[:,1], 1 - pos[:,1]])
    # Compute pairwise center distances
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    # Setup LP: maximize sum of radii <=> minimize negative sum
    c_lp = -np.ones(n)
    A1 = np.eye(n); b1 = rad_bound
    rows = []; bs = []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1; row[j] = 1
            rows.append(row); bs.append(dmat[i, j])
    if rows:
        A2 = np.vstack(rows); b2 = np.array(bs)
        A_ub = np.vstack([A1, A2]); b_ub = np.hstack([b1, b2])
    else:
        A_ub, b_ub = A1, b1

    # Solve LP for optimal radii
    from scipy.optimize import linprog
    res_lp = linprog(c_lp, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(0, None)] * n, method='highs')
    if res_lp.success:
        final[:, 2] = res_lp.x
    else:
        # Fallback: nearest-neighbor heuristic
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        dists, _ = tree.query(pos, k=2)
        final[:, 2] = np.minimum(dists[:,1] / 2.0, rad_bound)

    return final


# EVOLVE-BLOCK-END
