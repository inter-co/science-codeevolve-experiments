# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import scipy.spatial

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
    res = opt.minimize(obj, x0, method='SLSQP', constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-4})
    sol = res.x
    coords = sol[n:].reshape((n, 2))
    radii = sol[:n]
    return np.hstack([coords, radii.reshape(n, 1)])

# Physics-based force relaxation solver
def solve_phys():
    n = 32
    coords = np.random.rand(n, 2)
    radii = np.full(n, 0.01)
    for _ in range(500):
        forces = np.zeros_like(coords)
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                delta = coords[i] - coords[j]
                dist = np.linalg.norm(delta)
                target = radii[i] + radii[j]
                if dist < target and dist > 1e-6:
                    vec = delta / dist
                    f = (target - dist) * vec
                    forces[i] += f
                    forces[j] -= f
        # Boundary forces
        for i in range(n):
            x, y = coords[i]
            r = radii[i]
            forces[i, 0] += max(0, r - x) - max(0, x + r - 1)
            forces[i, 1] += max(0, r - y) - max(0, y + r - 1)
        coords += 0.01 * forces
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
def solve_radii_lp(centers):
    n = centers.shape[0]
    # maximize sum(r) → minimize -sum(r)
    c_obj = -np.ones(n)
    # radius bounds from distance to borders
    bd = np.minimum.reduce([centers[:,0], 1-centers[:,0],
                            centers[:,1], 1-centers[:,1]])
    bounds = [(0.0, bd_i) for bd_i in bd]
    # pairwise non-overlap constraints: r_i + r_j ≤ dist(center_i, center_j)
    A = []
    b = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.hypot(centers[i,0]-centers[j,0],
                            centers[i,1]-centers[j,1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A.append(row)
            b.append(dist)
    A = np.vstack(A)
    b = np.array(b)
    res_lp = opt.linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if res_lp.success:
        return np.hstack([centers, res_lp.x.reshape(-1,1)])
    else:
        # fallback: zero radii if LP fails
        return np.hstack([centers, np.zeros((n,1))])

def circle_packing32() -> np.ndarray:
    """
    Multi-seed CVT-based circle packing: run CVT with several random starts and pick the best.
    """
    import scipy.spatial
    seeds = [0, 7, 42, 99, 123]  # expanded seed list for multi-start CVT
    best_sol = None
    best_sum = -1.0
    for seed in seeds:
        np.random.seed(seed)
        n = 32
        # 1) Initialize random points
        pts = np.random.rand(n, 2)
        # 2) Lloyd's algorithm for CVT
        for _ in range(100):  # increased CVT iterations for better convergence
            vor = scipy.spatial.Voronoi(pts)
            new_pts = []
            for i in range(n):
                region = vor.regions[vor.point_region[i]]
                if not region or -1 in region:
                    new_pts.append(pts[i])
                else:
                    poly = vor.vertices[region]
                    # clip polygon to unit square
                    poly[:,0] = np.clip(poly[:,0], 0,1)
                    poly[:,1] = np.clip(poly[:,1], 0,1)
                    new_pts.append(poly.mean(axis=0))
            pts = np.array(new_pts)
        # 3) Solve radii via LP given fixed centers
        sol = solve_radii_lp(pts)
        # 4) Local inflation to exploit gaps
        sol_inf = inflate(sol, iters=10)
        # 5) Refinement via local coordinate perturbation
        refined = local_search(sol_inf, iters=500, sigma=0.01)
        # final NLP-based refinement
        try:
            sol_final = solve_nlp(init_sol=refined)
            if np.sum(sol_final[:,2]) > np.sum(refined[:,2]):
                refined = sol_final
        except Exception:
            pass
        total_r = np.sum(refined[:,2])
        if total_r > best_sum:
            best_sum = total_r
            best_sol = refined
    # Final NLP-based refinement to polish the best multi-seed CVT solution
    try:
        sol_final = solve_nlp(init_sol=best_sol)
        if np.sum(sol_final[:, 2]) > np.sum(best_sol[:, 2]):
            best_sol = sol_final
    except Exception:
        pass
    return best_sol


# EVOLVE-BLOCK-END
