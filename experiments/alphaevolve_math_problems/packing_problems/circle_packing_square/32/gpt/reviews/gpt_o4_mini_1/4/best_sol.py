# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import KDTree

def hexagonal_init(n):
    cols = int(np.ceil(np.sqrt(n)))
    dx = 1.0/cols
    dy = np.sqrt(3)*dx/2.0
    pts = []
    row = 0
    while len(pts)<n:
        offset = (dx/2.0) if (row%2==1) else 0.0
        y = row*dy
        x = offset
        while x <= 1.0+1e-8 and len(pts)<n:
            if y <= 1.0+1e-8:
                pts.append((min(max(x,0.0),1.0), min(max(y,0.0),1.0)))
            x += dx
        row += 1
    return np.array(pts[:n])

def compute_radii_from_positions(pos):
    n = pos.shape[0]
    # distance to each of the four boundaries
    r_bound = np.minimum(np.minimum(pos[:,0],1.0-pos[:,0]),
                         np.minimum(pos[:,1],1.0-pos[:,1]))
    if n>1:
        tree = KDTree(pos)
        dists,_ = tree.query(pos, k=2)
        # first neighbor is itself at distance 0, second is closest other
        r_nei = dists[:,1]*0.5
    else:
        r_nei = r_bound
    return np.maximum(0.0, np.minimum(r_bound, r_nei))

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

    # Initial seeding: choose between hexagonal lattice or KMeans Voronoi
    pos_hex = hexagonal_init(n)
    r_hex = compute_radii_from_positions(pos_hex)
    sum_hex = r_hex.sum()
    M = 20000
    pts = np.random.rand(M, 2)
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(pts)
    pos_km = kmeans.cluster_centers_
    r_km = compute_radii_from_positions(pos_km)
    sum_km = r_km.sum()
    if sum_hex > sum_km:
        init_pos, init_r = pos_hex, r_hex
    else:
        init_pos, init_r = pos_km, r_km
    circles = np.column_stack((init_pos, init_r))

    # Local randomized hill-climbing (position + radius)
    for _ in range(2000):
        idx = np.random.randint(n)
        x_old, y_old, r_old = circles[idx]
        x_new = np.clip(x_old + np.random.randn() * 0.02, 0, 1)
        y_new = np.clip(y_old + np.random.randn() * 0.02, 0, 1)
        r_new = compute_clearance(x_new, y_new, np.delete(circles, idx, axis=0))
        if r_new > r_old:
            circles[idx] = [x_new, y_new, r_new]
    # Radius-only smoothing
    for _ in range(1000):
        idx = np.random.randint(n)
        x, y, r_old = circles[idx]
        r_new = compute_clearance(x, y, np.delete(circles, idx, axis=0))
        if r_new > r_old:
            circles[idx, 2] = r_new

    # Force-based relaxation to adjust overlaps and grow radii
    alpha = 0.005
    for _ in range(500):
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

    def cp_sat_refine(circles, scale=2000, time_limit=10):
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
