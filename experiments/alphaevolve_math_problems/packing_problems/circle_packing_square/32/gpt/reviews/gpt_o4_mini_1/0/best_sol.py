# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
from scipy.spatial import Voronoi
from scipy.spatial import KDTree

# Local Nonlinear Programming (NLP) solver using SciPy's SLSQP
def solve_nlp(init_sol=None):
    n = 32
    # Initialize [radii, x, y]
    init_r = np.full(n, 0.02)
    init_xy = np.random.rand(n, 2)
    # Warm start with provided initial solution
    if init_sol is not None:
        r_init = init_sol[:, 2]
        coords_init = init_sol[:, :2]
        x0 = np.concatenate([r_init, coords_init.flatten()])
    else:
        x0 = np.concatenate([init_r, init_xy.flatten()])

    def obj(v):
        # negative sum of radii for minimizer
        return -np.sum(v[:n])

    cons = []
    # Containment constraints: r <= x <= 1-r, r <= y <= 1-r
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+2*i] - v[i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[n+2*i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+2*i+1] - v[i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[n+2*i+1]})
    # Non-overlap constraints: distance >= ri+rj
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.hypot(
                    v[n+2*i] - v[n+2*j],
                    v[n+2*i+1] - v[n+2*j+1]
                ) - (v[i] + v[j])
            })
    # use tighter convergence and more iterations for SLSQP
    res = opt.minimize(obj, x0, method='SLSQP', constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-5, 'disp': False})
    sol = res.x
    coords = sol[n:].reshape((n, 2))
    radii = sol[:n]
    return np.hstack([coords, radii.reshape(n, 1)])

# Physics-based force relaxation solver
def solve_phys(seed=None):
    # optionally reseed for diverse runs
    if seed is not None:
        np.random.seed(seed)
    n = 32
    coords = np.random.rand(n, 2)
    radii = np.full(n, 0.01)
    velocities = np.zeros_like(coords)
    dt = 0.1
    damping = 0.9
    # longer dynamics for better jamming
    for _ in range(1000):
        forces = np.zeros_like(coords)
        # Pairwise adaptive repulsion
        for i in range(n):
            for j in range(i+1, n):
                delta = coords[i] - coords[j]
                dist = np.linalg.norm(delta) + 1e-12
                overlap = (radii[i] + radii[j]) - dist
                if overlap > 0:
                    f = (overlap / dist) * delta
                    forces[i] += f
                    forces[j] -= f
        # Soft boundary penalty
        for i in range(n):
            x, y = coords[i]
            r = radii[i]
            if x < r:
                forces[i, 0] += (r - x)
            elif x + r > 1:
                forces[i, 0] -= (x + r - 1)
            if y < r:
                forces[i, 1] += (r - y)
            elif y + r > 1:
                forces[i, 1] -= (y + r - 1)
        # velocity update and damping
        velocities = velocities * damping + forces * dt
        coords += velocities * dt
        # adapt timestep
        max_force = np.max(np.linalg.norm(forces, axis=1))
        dt = min(0.1, max(1e-2, 1.0/(1.0 + max_force)))
    return np.hstack([coords, radii.reshape(n, 1)])

def random_greedy(seed, trials=1000):
    rng = np.random.RandomState(seed)
    circs = []
    for i in range(32):
        best_r = -1.0
        best_xy = (0.5, 0.5)
        for _ in range(trials):
            x, y = rng.rand(2)
            r = min(x, 1 - x, y, 1 - y)
            for x2, y2, r2 in circs:
                r = min(r, np.hypot(x - x2, y - y2) - r2)
            if r > best_r:
                best_r, best_xy = r, (x, y)
        circs.append((best_xy[0], best_xy[1], max(0.0, best_r)))
    return np.array(circs)

def inflate(circs, iters=5):
    c = circs.copy()
    for _ in range(iters):
        for i in range(c.shape[0]):
            x, y, _ = c[i]
            rmax = min(x, 1 - x, y, 1 - y)
            for j in range(c.shape[0]):
                if i == j:
                    continue
                x2, y2, r2 = c[j]
                d = np.hypot(x - x2, y - y2) - r2
                rmax = min(rmax, d)
            c[i, 2] = max(0.0, rmax)
    return c

# Local coordinate perturbation search for further improvement
def local_search(sol, iters=500, sigma=0.02):
    best = sol.copy()
    best_sum = np.sum(best[:, 2])
    n = sol.shape[0]
    for _ in range(iters):
        cand = best.copy()
        i = np.random.randint(n)
        # jitter center for circle i
        cand[i, 0:2] += np.random.normal(scale=sigma, size=2)
        # ensure center stays within valid boundaries
        cand[i, 0] = np.clip(cand[i, 0], cand[i, 2], 1 - cand[i, 2])
        cand[i, 1] = np.clip(cand[i, 1], cand[i, 2], 1 - cand[i, 2])
        # re-inflate radii locally after moving
        cand = inflate(cand, iters=2)
        s = np.sum(cand[:, 2])
        if s > best_sum:
            best = cand
            best_sum = s
    return best

# Solve radii via Linear Programming given fixed centers
def solve_radii_lp(centers, prune_k=None):
    n = centers.shape[0]
    # minimize -sum(r) → maximize sum(r)
    c_obj = -np.ones(n)
    # radius bounds from distance to borders
    bd = np.minimum.reduce([centers[:,0], 1-centers[:,0],
                            centers[:,1], 1-centers[:,1]])
    bounds = [(0.0, bd_i) for bd_i in bd]

    # build pairwise constraints with optional KDTree neighbor pruning
    if prune_k is not None and prune_k < n:
        tree = KDTree(centers)
        neighs = tree.query(centers, k=min(prune_k+1, n))[1]
        pairs = set()
        for i, nbr in enumerate(neighs):
            for j in nbr[1:]:
                a, b = min(i, j), max(i, j)
                pairs.add((a, b))
        pairs = list(pairs)
    else:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    m = len(pairs)
    A = np.zeros((m, n))
    b = np.zeros(m)
    for idx, (i, j) in enumerate(pairs):
        dist = np.hypot(centers[i,0] - centers[j,0],
                        centers[i,1] - centers[j,1])
        A[idx, i] = 1.0
        A[idx, j] = 1.0
        b[idx] = dist

    res_lp = opt.linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if res_lp.success:
        return np.hstack([centers, res_lp.x.reshape(-1,1)])
    else:
        # fallback: zero radii if LP fails
        return np.hstack([centers, np.zeros((n,1))])

def cvt_strategy(n=32, iters_cvt=50):
    # CVT-based center distribution via Lloyd's algorithm
    pts = np.random.rand(n, 2)
    for _ in range(iters_cvt):
        vor = Voronoi(pts)
        new_pts = []
        # Recompute each site as centroid of its Voronoi cell
        for idx, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                new_pts.append(pts[idx])
            else:
                polygon = vor.vertices[region]
                centroid = polygon.mean(axis=0)
                centroid = np.clip(centroid, 0.0, 1.0)
                new_pts.append(centroid)
        pts = np.array(new_pts)
    # Solve radii for CVT centers
    return solve_radii_lp(pts)

def hex_grid_strategy(n=32):
    # Hexagonal-grid seeding with LP radius optimization
    rows = [6, 5, 6, 5, 6, 4]
    spacing_x = 1.0 / 6
    spacing_y = np.sqrt(3.0) / 12.0
    centers = []
    for i, cnt in enumerate(rows):
        y = (i + 0.5) * spacing_y
        x_offset = (1.0 - cnt * spacing_x) / 2.0 + spacing_x / 2.0
        for j in range(cnt):
            x = x_offset + j * spacing_x
            centers.append((x, y))
    centers = np.array(centers[:n])
    # Solve radii via LP using KDTree pruning
    return solve_radii_lp(centers, prune_k=8)

def circle_packing32() -> np.ndarray:
    """
    Greedy sampling-based circle packing:
    Sequentially place each of the 32 circles by sampling candidate positions,
    estimating maximum feasible radius, and selecting the best candidate each step.
    """
    np.random.seed(0)  # deterministic sampling
    n = 32
    circles = np.zeros((n, 3))
    existing = []
    num_samples = 20000  # increased sampling for better greedy placement

    for k in range(n):
        # Sample random candidate positions within [0,1]^2
        xs = np.random.rand(num_samples)
        ys = np.random.rand(num_samples)
        # Distance to the four boundaries
        border_dist = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))

        if k == 0:
            # Place the first circle at the point farthest from any boundary
            idx = np.argmax(border_dist)
            circles[k] = [xs[idx], ys[idx], border_dist[idx]]
        else:
            # Stack existing circles for vectorized distance computations
            ex = np.array(existing)
            xex, yex, rex = ex[:, 0], ex[:, 1], ex[:, 2]
            # Compute center-to-center distances minus existing radii
            dx = xs[:, None] - xex[None, :]
            dy = ys[:, None] - yex[None, :]
            dists = np.sqrt(dx * dx + dy * dy) - rex
            # Nearest circle distance for each candidate
            nearest = np.min(dists, axis=1)
            # Feasible radius is limited by both nearest neighbor gap and border
            feasible = np.minimum(nearest, border_dist)
            # Choose the candidate allowing the largest circle
            idx = np.argmax(feasible)
            circles[k] = [xs[idx], ys[idx], feasible[idx]]

        # Append the chosen circle to the existing list
        existing.append(circles[k].tolist())

    # Hybrid selection: compare greedy, grid, voronoi, random-greedy, NLP, and physics-based strategies with inflation
    sol_greedy = circles
    sum_greedy = np.sum(sol_greedy[:, 2])

    # Grid strategy for baseline
    def grid_strategy(h=8, w=4):
        xs = (np.arange(h) + 0.5) / h
        ys = (np.arange(w) + 0.5) / w
        pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        r = min(1.0/(2*h), 1.0/(2*w))
        return np.hstack([pts, np.full((n,1), r)])
    sol_grid = grid_strategy()
    sum_grid = np.sum(sol_grid[:, 2])

    # Voronoi-based seeding (multi-seed sweep)
    def voronoi_strategy(seed=0):
        rng = np.random.RandomState(seed)
        pts = rng.rand(n, 2)
        circs = np.zeros((n,3))
        for i, (x, y) in enumerate(pts):
            r = min(x, 1 - x, y, 1 - y)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            d = np.min(np.delete(dists, i))
            circs[i] = [x, y, min(r, d/2.0)]
        return circs
    # try several random seeds and pick the best Voronoi result
    sol_voro_candidates = [voronoi_strategy(seed=s) for s in range(5)]
    sum_voro_list = [np.sum(v[:,2]) for v in sol_voro_candidates]
    best_voro_idx = int(np.argmax(sum_voro_list))
    sol_voro = sol_voro_candidates[best_voro_idx]
    sum_voro = sum_voro_list[best_voro_idx]

    # Random sequential greedy strategies
    sol_rg0 = random_greedy(seed=0, trials=1000)
    sum_rg0 = np.sum(sol_rg0[:, 2])
    sol_rg1 = random_greedy(seed=1, trials=1000)
    sum_rg1 = np.sum(sol_rg1[:, 2])

    # NLP solution using greedy warm start
    try:
        sol_nlp = solve_nlp(init_sol=sol_greedy)
        sum_nlp = np.sum(sol_nlp[:, 2])
    except Exception:
        sol_nlp = sol_greedy; sum_nlp = sum_greedy

    # Physics-based solution (multi-seed ensemble)
    sol_phys_list = []
    sum_phys_list = []
    for s in range(3):
        try:
            sol_ph = solve_phys(seed=s)
            sol_phys_list.append(sol_ph)
            sum_phys_list.append(np.sum(sol_ph[:, 2]))
        except Exception:
            sol_phys_list.append(sol_greedy)
            sum_phys_list.append(sum_greedy)
    best_idx = int(np.argmax(sum_phys_list))
    sol_phys = sol_phys_list[best_idx]
    sum_phys = sum_phys_list[best_idx]

    # Aggregate and apply inflation
    # Insert CVT‐based and hex‐grid candidates
    sol_cvt = cvt_strategy(n, iters_cvt=50)
    sum_cvt = np.sum(sol_cvt[:, 2])
    sol_hex = hex_grid_strategy(n=n)
    sum_hex = np.sum(sol_hex[:, 2])

    candidates = [
        (sum_greedy, sol_greedy),
        (sum_grid, sol_grid),
        (sum_voro, sol_voro),
        (sum_cvt, sol_cvt),
        (sum_hex, sol_hex),
        (sum_rg0, sol_rg0),
        (sum_rg1, sol_rg1),
        (sum_nlp, sol_nlp),
        (sum_phys, sol_phys)
    ]
    inflated = []
    for s, sol in candidates:
        sol_inf = inflate(sol, iters=5)
        s_inf = np.sum(sol_inf[:, 2])
        inflated.append((s_inf if s_inf > s else s, sol_inf if s_inf > s else sol))
    best_config = max(inflated, key=lambda x: x[0])[1]
    # refine via local coordinate search
    try:
        refined = local_search(best_config, iters=300, sigma=0.02)
        if np.sum(refined[:, 2]) > np.sum(best_config[:, 2]):
            return refined
    except Exception:
        pass
    # Final LP-based radius re-optimization
    try:
        centers = best_config[:, :2]
        sol_lp = solve_radii_lp(centers)
        if np.sum(sol_lp[:, 2]) > np.sum(best_config[:, 2]):
            return sol_lp
    except Exception:
        pass
    return best_config


# EVOLVE-BLOCK-END
