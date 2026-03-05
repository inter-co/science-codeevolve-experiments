# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import linprog, minimize
import nevergrad as ng  # ES-based optimization import

# Modular circle packing strategies for methodological diversity.

def greedy_random_packing(n=32, iterations=500, seed=42):
    """
    Physics-based force-directed circle packing.
    Initializes random small circles, applies repulsive forces, and inflates radii.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Initialize positions and radii
    pts = np.random.rand(n, 2)
    radii = np.full(n, 0.01)
    # Simulation parameters
    k_repulse = 1.0
    lr = 0.01
    for _ in range(iterations):
        forces = np.zeros_like(pts)
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i+1, n):
                delta = pts[i] - pts[j]
                dist = np.linalg.norm(delta)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    direction = delta / (dist + 1e-6)
                    force = k_repulse * overlap * direction
                    forces[i] += force
                    forces[j] -= force
        # Boundary repulsion
        for i in range(n):
            for d in [0, 1]:
                if pts[i, d] - radii[i] < 0:
                    forces[i, d] += k_repulse * (radii[i] - pts[i, d])
                if pts[i, d] + radii[i] > 1:
                    forces[i, d] -= k_repulse * (radii[i] - (1 - pts[i, d]))
        # Position update and clipping
        pts += lr * forces
        pts = np.clip(pts, 0, 1)
        # Gradually inflate radii
        radii += lr * 0.05
    # Final radius trimming to enforce non-overlap and boundary constraints
    for i in range(n):
        radii[i] = min(radii[i], pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(pts[i] - pts[j])
                radii[i] = min(radii[i], dist - radii[j])
    radii = np.clip(radii, 1e-6, None)
    return np.hstack((pts, radii.reshape(-1, 1)))

def lattice_packing(n=32, rows=6, cols=6):
    """
    Simple lattice-based equal-radius packing as an analytic baseline.
    """
    positions = []
    r = 1.0 / (2 * max(rows, cols))
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) * 2 * r
            y = (i + 0.5) * 2 * r
            positions.append((x, y))
        if len(positions) >= n:
            break
    radii = [r] * len(positions)
    return np.column_stack((np.array(positions), np.array(radii)))

def cvt_packing(n=32, iterations=50, seed=42):
    """
    Centroidal Voronoi Tessellation (CVT) based packing.
    Iteratively moves points to region centroids and sets radii to half
    the minimum distance to neighbors or boundary.
    """
    import numpy as _np
    from scipy.spatial import Voronoi, KDTree
    _np.random.seed(seed)
    pts = _np.random.rand(n, 2)
    for _ in range(iterations):
        vor = Voronoi(pts)
        new_pts = pts.copy()
        for i, region_idx in enumerate(vor.point_region):
            vertices = vor.regions[region_idx]
            if not vertices or -1 in vertices:
                continue
            poly = _np.array([vor.vertices[v] for v in vertices])
            poly = _np.clip(poly, 0, 1)
            centroid = poly.mean(axis=0)
            new_pts[i] = centroid
        pts = new_pts
    # Compute radii
    radii = _np.minimum.reduce([pts[:, 0], 1 - pts[:, 0], pts[:, 1], 1 - pts[:, 1]])
    tree = KDTree(pts)
    for i in range(n):
        dists, idxs = tree.query(pts[i], k=2)
        if len(dists) > 1:
            neigh_dist = dists[1]
            radii[i] = min(radii[i], neigh_dist / 2)
    return _np.hstack((pts, radii.reshape(-1, 1)))

# LP-based radius optimization for fixed positions
def optimize_radii(xs, ys):
    """
    Optimize circle radii for given positions using linear programming.
    xs, ys: arrays of shape (n,)
    Returns array of shape (n,3): x, y, optimized r
    """
    n = len(xs)
    # Objective: maximize sum of r_i => minimize -sum r_i
    c = [-1.0] * n
    A = []
    b = []
    # Boundary constraints: r_i <= x_i, 1-x_i, y_i, 1-y_i
    for i in range(n):
        ai = [0.0]*n
        ai[i] = 1.0
        A.append(ai.copy()); b.append(xs[i])
        A.append(ai.copy()); b.append(1.0 - xs[i])
        A.append(ai.copy()); b.append(ys[i])
        A.append(ai.copy()); b.append(1.0 - ys[i])
    # Pairwise non-overlap: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            dij = math.hypot(xs[i]-xs[j], ys[i]-ys[j])
            aij = [0.0]*n
            aij[i] = 1.0
            aij[j] = 1.0
            A.append(aij)
            b.append(dij)
    # Solve LP
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    if res.success:
        rs = res.x
    else:
        # Fallback to minimal boundary radii
        rs = np.minimum(xs, np.minimum(1-xs, np.minimum(ys, 1-ys)))
    return np.column_stack((xs, ys, rs))

# Local stochastic search: jitter circle positions and re-optimize radii
def local_search(xs, ys, rs, iters=200, noise_scale=0.01, seed=42):
    """
    Perform stochastic local refinement by randomly perturbing one circle at a time
    and re-optimizing radii. Retains improvements in total radius sum.
    """
    random.seed(seed)
    best_xs, best_ys = xs.copy(), ys.copy()
    best_pack = np.column_stack((best_xs, best_ys, rs.copy()))
    best_sum = np.sum(rs)
    for _ in range(iters):
        i = random.randrange(len(xs))
        # Perturb the i-th circle's position
        cand_xs, cand_ys = best_xs.copy(), best_ys.copy()
        cand_xs[i] = min(max(cand_xs[i] + random.uniform(-noise_scale, noise_scale), 0.0), 1.0)
        cand_ys[i] = min(max(cand_ys[i] + random.uniform(-noise_scale, noise_scale), 0.0), 1.0)
        # Re-optimize radii
        cand_pack = optimize_radii(cand_xs, cand_ys)
        cand_sum = np.sum(cand_pack[:, 2])
        if cand_sum > best_sum:
            best_sum = cand_sum
            best_xs, best_ys = cand_pack[:,0].copy(), cand_pack[:,1].copy()
            best_pack = cand_pack.copy()
    return best_pack

# Evolutionary strategy-based packing using CMA-ES from Nevergrad
def es_packing(n=32, seed=42, budget=1000):
    """
    Evolutionary strategy optimization of circle positions.
    Positions are optimized via CMA-ES to maximize sum of LP-optimized radii.
    """
    random.seed(seed)
    np.random.seed(seed)
    instr = ng.p.Array(shape=(n,2)).set_bounds(0.0, 1.0)
    optimizer = ng.optimizers.CMA(parametrization=instr, budget=budget)
    for _ in range(budget):
        x = optimizer.ask()
        positions = x.value
        # Obtain optimized radii for the candidate positions
        pack = optimize_radii(positions[:,0], positions[:,1])
        score = -np.sum(pack[:,2])
        optimizer.tell(x, score)
    best_positions = optimizer.provide_recommendation().value
    best_pack = optimize_radii(best_positions[:,0], best_positions[:,1])
    return best_pack

# SLSQP-based continuous refinement using SciPy minimize (further precision polish)
def slsqp_refine(circles, maxiter=200):
    n = len(circles)
    # Flatten variables: x positions, y positions, radii
    x0 = np.hstack([circles[:,0], circles[:,1], circles[:,2]])
    def obj(x):
        # maximize sum of radii => minimize negative
        return -np.sum(x[2*n:])
    cons = []
    # boundary constraints: r_i ≤ x_i ≤ 1−r_i, r_i ≤ y_i ≤ 1−r_i
    for i in range(n):
        cons.append({'type':'ineq','fun': lambda x, i=i: x[i] - x[2*n+i]})
        cons.append({'type':'ineq','fun': lambda x, i=i: 1 - x[i] - x[2*n+i]})
        cons.append({'type':'ineq','fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        cons.append({'type':'ineq','fun': lambda x, i=i: 1 - x[n+i] - x[2*n+i]})
    # non-overlap constraints: (x_i−x_j)^2+(y_i−y_j)^2 ≥ (r_i+r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type':'ineq',
                'fun': lambda x, i=i, j=j: 
                    (x[i]-x[j])**2 + (x[n+i]-x[n+j])**2 
                    - (x[2*n+i] + x[2*n+j])**2
            })
    res = minimize(obj, x0, method='SLSQP', constraints=cons,
                   options={'ftol':1e-6, 'maxiter':maxiter, 'disp':False})
    if res.success:
        sol = res.x
        return np.vstack([sol[:n], sol[n:2*n], sol[2*n:]]).T
    return circles

def circle_packing32() -> np.ndarray:
    """
    Aggregates multiple strategies: physics, lattice, CVT, and ES-based optimization.
    Introduces evolutionary strategy (CMA-ES via Nevergrad) for positions search.
    """
    # Strategy 1: physics-based packing
    pack1 = greedy_random_packing()
    # Strategy 2: lattice-based packing
    pack2 = lattice_packing()
    # Strategy 3: centroidal Voronoi tessellation packing
    pack3 = cvt_packing()
    # Strategy 4: evolutionary strategy-based packing
    pack4 = es_packing()

    # Additional Strategy 5: extended local hill-climbing on CVT+LP result
    # Core idea: start from the CVT centers after LP-based radius optimization,
    # then do a deeper stochastic local search. Trade-off: more iterations for
    # diminishing returns but often lifts benchmarks.
    pack5_init = optimize_radii(pack3[:,0], pack3[:,1])
    pack5 = local_search(pack5_init[:,0], pack5_init[:,1], pack5_init[:,2],
                         iters=500, noise_scale=0.02)

    # Optimize and locally refine each configuration
    configs = [pack1, pack2, pack3, pack4]
    packs = []
    for idx, cfg in enumerate(configs):
        if idx < 3:
            opt = optimize_radii(cfg[:,0], cfg[:,1])
            refined = local_search(opt[:,0], opt[:,1], opt[:,2],
                                   iters=200, noise_scale=0.01)
        else:
            # es_packing already returns LP-optimized pack; just refine
            refined = local_search(cfg[:,0], cfg[:,1], cfg[:,2],
                                   iters=200, noise_scale=0.01)
        packs.append(refined)
    # Include the new CVT+LP hill-climbing candidate
    packs.append(pack5)

    # Evaluate total radii sum and select the best
    sums = [np.sum(p[:, 2]) for p in packs]
    best_index = int(np.argmax(sums))
    best = packs[best_index]
    # final continuous SLSQP refinement
    return slsqp_refine(best, maxiter=200)


# EVOLVE-BLOCK-END
