# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from scipy.optimize import linprog
from scipy.spatial import cKDTree  # For LP fallback and NN queries

def circle_packing32() -> np.ndarray:
    """
    Optimization via SciPy's SLSQP solver:
    Decision vector z = [x0, y0, r0, ..., x31, y31, r31]
    Maximize sum of radii under containment and non-overlap constraints.
    """
    n = 32
    # Seed for reproducibility
    np.random.seed(0)
    # KMeans-based seeding for initial centers and radii
    M = 30000
    pts = np.random.rand(M, 2)
    kmeans = KMeans(n_clusters=n, random_state=0, n_init=10).fit(pts)
    centers = kmeans.cluster_centers_
    # Compute initial radii based on boundary and nearest-neighbor distances
    init_r = []
    for idx, c in enumerate(centers):
        db = min(c[0], 1 - c[0], c[1], 1 - c[1])
        dists = np.linalg.norm(centers - c, axis=1)
        dn = np.min(dists[np.arange(len(dists)) != idx]) / 2.0
        init_r.append(min(db, dn))

    # Force-based relaxation on initial placements (inspired by Inspiration 1)
    circles_init = np.hstack((centers, np.array(init_r).reshape(-1,1)))
    for _ in range(100):
        disp = np.zeros((n, 2))
        for i in range(n):
            xi, yi, ri = circles_init[i]
            # boundary repulsion
            disp[i,0] += (max(ri - xi, 0) - max(ri - (1 - xi), 0))
            disp[i,1] += (max(ri - yi, 0) - max(ri - (1 - yi), 0))
            # pairwise repulsion
            for j in range(i+1, n):
                xj, yj, rj = circles_init[j]
                dx = xi - xj; dy = yi - yj
                dist = np.hypot(dx, dy) + 1e-6
                overlap = (ri + rj) - dist
                if overlap > 0:
                    shift = 0.01 * overlap * np.array([dx, dy]) / dist
                    disp[i] += shift
                    disp[j] -= shift
        circles_init[:,:2] += disp
        # clamp within the unit square
        circles_init[:,0] = np.clip(circles_init[:,0], circles_init[:,2], 1 - circles_init[:,2])
        circles_init[:,1] = np.clip(circles_init[:,1], circles_init[:,2], 1 - circles_init[:,2])
    # Update centers and init_r from relaxed circles
    centers = circles_init[:,:2]
    init_r = circles_init[:,2].tolist()

    # Greedy maximal inflation: inflate radii based on neighbor and boundary clearance before optimization
    pos = centers
    rad = np.array(init_r)
    tree = cKDTree(pos)
    for i in range(n):
        x, y = pos[i]
        # boundary clearance
        dmin = min(x, y, 1 - x, 1 - y)
        # neighbor-based clearance
        idxs = tree.query_ball_point((x, y), dmin + rad.max())
        for j in idxs:
            if j == i:
                continue
            dmin = min(dmin, np.hypot(x - pos[j,0], y - pos[j,1]) - rad[j])
        # update radius with maximal feasible
        rad[i] = max(rad[i], dmin - 1e-8)
    init_r = rad.tolist()

    # Build initial decision vector z = [x0,y0,r0,...]
    x0 = np.zeros(3 * n)
    for i, (cx, cy) in enumerate(centers):
        x0[3*i]   = cx
        x0[3*i+1] = cy
        x0[3*i+2] = init_r[i]

    # Objective: maximize sum of radii <=> minimize negative sum
    def objective(z):
        return -np.sum(z[2::3])

    # Collect inequality constraints
    constraints = []
    # Containment: ri ≤ xi ≤ 1-ri and ri ≤ yi ≤ 1-ri
    for i in range(n):
        idx = 3 * i
        constraints.extend([
            {'type': 'ineq', 'fun': lambda z, idx=idx: z[idx] - z[idx + 2]},         # xi - ri >= 0
            {'type': 'ineq', 'fun': lambda z, idx=idx: 1 - z[idx] - z[idx + 2]},     # xi + ri <= 1
            {'type': 'ineq', 'fun': lambda z, idx=idx: z[idx + 1] - z[idx + 2]},     # yi - ri >= 0
            {'type': 'ineq', 'fun': lambda z, idx=idx: 1 - z[idx + 1] - z[idx + 2]}, # yi + ri <= 1
        ])
    # Non-overlap: distance(pi, pj) - ri - rj >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: np.hypot(z[3*i] - z[3*j],
                                                    z[3*i+1] - z[3*j+1]) 
                                            - z[3*i+2] - z[3*j+2]
            })

    # Run the SLSQP solver
    # Run the SLSQP solver with tighter convergence settings
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-6, 'disp': False}
    )

    # Check solver success; if failed, fallback to LP radius optimization
    if not result.success:
        z = result.x if hasattr(result, 'x') else x0
        print(f"Warning: SLSQP failed ({result.message}), proceeding with LP radius optimization.")
    else:
        z = result.x

    # Reshape variables into (x, y, r)
    circles = z.reshape((n, 3))

    # Enhance radii via LP given fixed positions
    pos = circles[:, :2]
    rad_bound = np.minimum.reduce([pos[:,0], 1 - pos[:,0],
                                   pos[:,1], 1 - pos[:,1]])
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    c_lp = -np.ones(n)
    A1 = np.eye(n); b1 = rad_bound
    rows = []; bs = []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1; row[j] = 1
            rows.append(row); bs.append(dmat[i,j])
    if rows:
        A2 = np.vstack(rows); b2 = np.array(bs)
        A_ub = np.vstack([A1, A2]); b_ub = np.hstack([b1, b2])
    else:
        A_ub, b_ub = A1, b1

    res_lp = linprog(c_lp, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(0, None)] * n, method='highs')
    if res_lp.success:
        circles[:,2] = res_lp.x
    else:
        # Fallback: nearest-neighbor heuristic for radii
        tree = cKDTree(pos)
        dists, _ = tree.query(pos, k=2)
        circles[:,2] = np.minimum(dists[:,1] / 2, rad_bound)

    # Final local hill-climbing refinement (inspired by Inspiration Program 2)
    rng = np.random.default_rng(1)
    def compute_clearance(x, y, arr):
        rr = min(x, 1 - x, y, 1 - y)
        for ox, oy, orad in arr:
            d = np.hypot(x - ox, y - oy) - orad
            if d < rr:
                rr = d
            if rr <= 0:
                return 0.0
        return rr

    steps = 2000
    max_sigma = 0.005
    for t in range(steps):
        sigma = max_sigma * (1 - t / steps)
        i = rng.integers(n)
        xi, yi, ri = circles[i]
        x_new = np.clip(xi + rng.normal(0, sigma), ri, 1 - ri)
        y_new = np.clip(yi + rng.normal(0, sigma), ri, 1 - ri)
        others = np.delete(circles, i, axis=0)
        r_new = compute_clearance(x_new, y_new, others)
        if r_new > ri:
            circles[i] = [x_new, y_new, r_new]

    return circles


# EVOLVE-BLOCK-END
