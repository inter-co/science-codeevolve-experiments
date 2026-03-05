# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Voronoi, cKDTree

# Stochastic hill-climbing refinement (inspired by heuristic local search)
def refine_hill_climb(init_centers, init_radii, iters=3000, move_scale=0.005, seed=0):
    """
    Simple stochastic hill-climbing: perturb circles to increase sum of radii.
    """
    n = init_centers.shape[0]
    rng = np.random.RandomState(seed)
    centers = init_centers.copy()
    radii = init_radii.copy()
    best_sum = radii.sum()
    for _ in range(iters):
        i = rng.randint(n)
        # propose random move
        new_c = centers[i] + rng.normal(scale=move_scale, size=2)
        new_c = np.clip(new_c, 0, 1)
        # boundary clearance
        db = min(new_c[0], 1-new_c[0], new_c[1], 1-new_c[1])
        # neighbor clearance
        if n > 1:
            dists = np.linalg.norm(centers - new_c, axis=1) - radii
            dists[i] = db
            d_neighbor = dists.min()
        else:
            d_neighbor = db
        new_r = max(0.0, min(db, d_neighbor))
        new_sum = best_sum - radii[i] + new_r
        if new_sum > best_sum:
            centers[i] = new_c
            radii[i] = new_r
            best_sum = new_sum
    return np.hstack([centers, radii.reshape(-1,1)])

def pack_voronoi(n=32, lloyd_iters=50, seed=101):
    """
    Generate an initial set of n circles via approximate centroidal Voronoi tessellation.
    """
    rng = np.random.RandomState(seed)
    points = rng.rand(n, 2)
    # Lloyd-like relaxation
    for _ in range(lloyd_iters):
        vor = Voronoi(points)
        new_pts = []
        for idx, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                new_pts.append(points[idx])
            else:
                verts = vor.vertices[region]
                # clip to unit square
                verts = np.minimum(np.maximum(verts, 0.0), 1.0)
                # approximate centroid by average of clipped vertices
                c = verts.mean(axis=0)
                new_pts.append(c.tolist())
        points = np.array(new_pts)
    tree = cKDTree(points)
    radii = np.zeros(n)
    for i, p in enumerate(points):
        d_edge = min(p[0], 1-p[0], p[1], 1-p[1])
        dists, _ = tree.query(p, k=2)
        radii[i] = min(d_edge, dists[1]/2)
    return np.hstack([points, radii.reshape(-1,1)])

def pack_greedy(n=32, samples_per_iter=10000, seed=102):
    """
    Generate an initial set of n circles by greedy maximal-clearance sampling.
    """
    rng = np.random.RandomState(seed)
    centers, radii = [], []
    for _ in range(n):
        pts = rng.rand(samples_per_iter, 2)
        best_point, best_r = None, -1.0
        for p in pts:
            db = min(p[0], p[1], 1-p[0], 1-p[1])
            if centers:
                dcs = np.linalg.norm(np.array(centers)-p, axis=1) - np.array(radii)
                db = min(db, dcs.min())
            if db > best_r:
                best_r, best_point = db, p
        centers.append(best_point)
        radii.append(best_r)
    pts = np.array(centers)
    return np.hstack([pts, np.array(radii).reshape(-1,1)])

# Physics-based repulsion helper (inspired by Inspiration Programs 1 & 2)
def repulsive_force_packing(n=32, max_iter=2000, tol=1e-4, lr=0.1, seed=42):
    """
    Physics-inspired repulsive force relaxation plus greedy radii.
    """
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    radii = np.full(n, 0.01)
    for _ in range(max_iter):
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff) + 1e-9
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    dir_vec = diff / dist
                    forces[i] += dir_vec * overlap
                    forces[j] -= dir_vec * overlap
        # Centripetal force toward center
        forces += -(pos - 0.5)
        pos += lr * forces
        pos = np.clip(pos, radii[:, None], 1 - radii[:, None])
    # Greedy expansion of radii
    for i in range(n):
        max_r = min(pos[i,0], pos[i,1], 1 - pos[i,0], 1 - pos[i,1])
        for j in range(n):
            if i == j: continue
            d = np.linalg.norm(pos[i] - pos[j]) - radii[j]
            max_r = min(max_r, d)
        radii[i] = max(0.0, max_r)
    return pos, radii

# Using constrained nonlinear programming (SLSQP) for 32-circle packing
def circle_packing32() -> np.ndarray:
    """
    Solves a constrained nonlinear optimization to maximize the sum of radii
    for 32 circles in the unit square [0,1]×[0,1].
    Variables order: [x0..x31, y0..y31, r0..r31]
    Constraints:
      - 0 ≤ xi, yi ≤ 1
      - radius constraints: xi - ri ≥ 0, 1 - xi - ri ≥ 0; yi - ri ≥ 0, 1 - yi - ri ≥ 0
      - non-overlap: √((xi-xj)^2 + (yi-yj)^2) - (ri + rj) ≥ 0
    """
    n = 32
    # reproducible random start
    np.random.seed(42)
    # Improved hexagonal‐lattice initialization + max‐feasible radii
    m = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0, 1, m)
    ys = np.linspace(0, 1, m)
    hex_pts = []
    for i, yy in enumerate(ys):
        offset = (0.5 / m) if (i % 2) else 0.0
        for xx in xs:
            xpt = np.clip(xx + offset, 0, 1)
            hex_pts.append([xpt, yy])
    hex_pts = np.array(hex_pts)[:n]
    from scipy.spatial import KDTree
    tree = KDTree(hex_pts)
    dists, idxs = tree.query(hex_pts, k=2)
    half_nn = dists[:, 1] / 2.0
    boundary = np.minimum.reduce([
        hex_pts[:,0], 1 - hex_pts[:,0],
        hex_pts[:,1], 1 - hex_pts[:,1]
    ])
    init_r = np.minimum(half_nn, boundary)
    x0 = np.concatenate([
        hex_pts[:, 0],   # x coords
        hex_pts[:, 1],   # y coords
        init_r           # initial radii
    ])

    # bounds: xi, yi ∈ [0,1], ri ∈ [0,0.5]
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n

    # inequality constraints
    constraints = []
    # containment constraints
    for i in range(n):
        # xi - ri ≥ 0
        constraints.append({'type':'ineq', 'fun': lambda z, i=i: z[i] - z[2*n + i]})
        # 1 - xi - ri ≥ 0
        constraints.append({'type':'ineq', 'fun': lambda z, i=i: 1 - z[i] - z[2*n + i]})
        # yi - ri ≥ 0
        constraints.append({'type':'ineq', 'fun': lambda z, i=i: z[n + i] - z[2*n + i]})
        # 1 - yi - ri ≥ 0
        constraints.append({'type':'ineq', 'fun': lambda z, i=i: 1 - z[n + i] - z[2*n + i]})

    # non-overlap constraints between each pair (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: np.hypot(z[i] - z[j], z[n + i] - z[n + j]) 
                                            - (z[2*n + i] + z[2*n + j])
            })

    # objective: maximize Σ ri  ⇒  minimize negative sum
    def objective(z):
        return -np.sum(z[2*n:])

    # Multi-start SLSQP + hill-climb refinement
    best_sum = -np.inf
    best_circles = None
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    for restart in range(3):
        # jitter the initial guess on subsequent restarts
        if restart > 0:
            rng = np.random.RandomState(42 + restart)
            jitter = rng.normal(scale=0.002, size=x0.shape)
            x_init = np.minimum(np.maximum(x0 + jitter, lows), highs)
        else:
            x_init = x0.copy()

        res = minimize(
            objective, x_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
        )
        if not res.success:
            continue

        z = res.x
        cir = np.vstack([z[:n], z[n:2*n], z[2*n:]]).T
        cir_ref = refine_hill_climb(cir[:, :2], cir[:, 2], iters=3000, move_scale=0.005, seed=restart)
        total_r = cir_ref[:, 2].sum()
        if total_r > best_sum:
            best_sum = total_r
            best_circles = cir_ref

    # Combine with heuristic-based initial solutions
    try:
        # Voronoi-based heuristic + hill-climb
        hv = pack_voronoi()
        hv_ref = refine_hill_climb(hv[:, :2], hv[:, 2],
                                   iters=2000, move_scale=0.01, seed=200)
        sum_hv = hv_ref[:, 2].sum()
        if sum_hv > best_sum:
            best_sum = sum_hv
            best_circles = hv_ref
        # Greedy-based heuristic + hill-climb
        hg = pack_greedy()
        hg_ref = refine_hill_climb(hg[:, :2], hg[:, 2],
                                   iters=2000, move_scale=0.01, seed=201)
        sum_hg = hg_ref[:, 2].sum()
        if sum_hg > best_sum:
            best_sum = sum_hg
            best_circles = hg_ref

        # Physics-based repulsion heuristic + hill-climb
        pos_rep, r_rep = repulsive_force_packing(n, max_iter=1000, lr=0.05, seed=42)
        rep_ref = refine_hill_climb(pos_rep, r_rep,
                                    iters=2000, move_scale=0.005, seed=202)
        sum_rep = rep_ref[:, 2].sum()
        if sum_rep > best_sum:
            best_sum = sum_rep
            best_circles = rep_ref
    except Exception:
        pass  # fallback to SLSQP result on error

    # Final LP refinement on best centers to maximize radii (inspired by Inspiration Program 2)
    try:
        centers = best_circles[:, :2]
        # Build pairwise distance matrix
        dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        # Inequality constraints: r_i + r_j <= d_ij
        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i+1, n):
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dmat[i, j])
        # Boundary constraints: each radius ≤ distance to each side
        for i in range(n):
            xi, yi = centers[i]
            for ub in (xi, yi, 1 - xi, 1 - yi):
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(ub)
        A_ub = np.vstack(A_ub)
        b_ub = np.array(b_ub)
        # Objective: maximize Σ r ⇔ minimize –Σ r
        c = -np.ones(n)
        res_lp = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            bounds=[(0, None)] * n,
            method='highs',
            options={'tol': 1e-8}
        )
        if res_lp.success:
            best_circles = np.hstack([centers, res_lp.x.reshape(-1,1)])
    except Exception:
        pass
    return best_circles


# EVOLVE-BLOCK-END
