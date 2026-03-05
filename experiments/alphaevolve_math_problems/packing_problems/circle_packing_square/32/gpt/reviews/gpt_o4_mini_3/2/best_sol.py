# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import KDTree
import time, math
from scipy.optimize import minimize

# Hybrid simulated annealing on positions with exact radius computation
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Simulated annealing on positions, computing radii exactly
    via nearest-neighbor and boundary distances (KDTree).
    """
    n = 32
    # grid-based initialization with jitter
    grid = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0.1, 0.9, grid)
    ys = xs
    pts = np.array([(x, y) for x in xs for y in ys][:n])
    pts += np.random.uniform(-0.01, 0.01, pts.shape)  # Reduced jitter for tighter initial layout
    pts = np.clip(pts, 0.0, 1.0)

    def compute_radii(pts):
        tree = KDTree(pts)
        radii = np.zeros(n)
        for i, p in enumerate(pts):
            # boundary distance
            d_edge = min(p[0], p[1], 1 - p[0], 1 - p[1])
            # nearest neighbor distance (skip self at index 0)
            dists, idxs = tree.query(p, k=2)
            d_nn = dists[1] if len(dists) > 1 else np.inf
            radii[i] = min(d_edge, d_nn / 2)
        return radii

    # initialize current & best states
    curr_rs = compute_radii(pts)
    curr_pts = pts.copy()
    curr_score = curr_rs.sum()
    best_pts, best_rs, best_score = curr_pts.copy(), curr_rs.copy(), curr_score

    # SA parameters (inspired by Inspiration 1)
    T0, Tf = 0.2, 1e-4      # lower initial temp, deeper cooling
    steps = 5000            # fewer but more focused annealing steps
    start_time = time.time()

    for k in range(steps):
        # time cutoff (keep under 60s)
        if time.time() - start_time > 55:
            break
        T = T0 * (Tf/T0) ** (k / steps)
        i = random.randrange(n)
        old_p = curr_pts[i].copy()
        # propose move
        curr_pts[i] += np.random.randn(2) * 0.02
        curr_pts[i] = np.clip(curr_pts[i], 0.0, 1.0)
        # recompute radii & score
        new_rs = compute_radii(curr_pts)
        new_score = new_rs.sum()
        delta = new_score - curr_score
        # Metropolis on current → candidate
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-8)):
            curr_score, curr_rs = new_score, new_rs
            # update best if improved
            if curr_score > best_score:
                best_score, best_pts, best_rs = curr_score, curr_pts.copy(), curr_rs.copy()
        else:
            curr_pts[i] = old_p

    # Force‐relaxation to remove overlaps from best configuration
    pts = best_pts.copy()
    radii = best_rs.copy()
    tree = KDTree(pts)
    for _ in range(100):
        moved = False
        for ii in range(n):
            idxs = tree.query_ball_point(pts[ii], radii[ii] + radii.max() + 1e-6)
            for jj in idxs:
                if jj <= ii: continue
                d = pts[jj] - pts[ii]
                dist = np.linalg.norm(d)
                overlap = radii[ii] + radii[jj] - dist
                if overlap > 0:
                    shift = (d / (dist + 1e-8)) * (overlap/2 + 1e-4)
                    pts[ii] -= shift
                    pts[jj] += shift
                    moved = True
        pts = np.clip(pts, 0.0, 1.0)
        tree = KDTree(pts)
        if not moved:
            break

    # Post‐relaxation radius inflation
    radii = compute_radii(pts)
    for _ in range(50):
        grown = 0
        tree = KDTree(pts)
        for ii in range(n):
            dists, _ = tree.query(pts[ii], k=n)
            max_r = min(
                pts[ii][0], pts[ii][1],
                1-pts[ii][0], 1-pts[ii][1],
                dists[1:].min()/2
            )
            if max_r > radii[ii] + 1e-4:
                radii[ii] = (radii[ii] + max_r)/2
                grown += 1
        if grown == 0:
            break

    # Polishing with SLSQP nonlinear optimization (inspired by inspirations)
    # pack current best into flat vector
    v0 = np.hstack((pts, radii.reshape(-1,1))).ravel()
    # objective: maximize sum of radii → minimize neg sum
    def _obj(v):
        return -np.sum(v[2::3])
    # build constraints
    cons = []
    # boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
    # non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j:
                    (v[3*i] - v[3*j])**2
                  + (v[3*i+1] - v[3*j+1])**2
                  - (v[3*i+2] + v[3*j+2])**2
            })
    # Analytical Jacobian for fast convergence
    def _jac(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g

    # Multi-restart SLSQP on the SA-derived seed
    best_sol = None
    best_score = -np.inf
    last_res = None
    for seed in range(3):
        if seed == 0:
            v_try = v0.copy()
        else:
            rng = np.random.RandomState(42 + seed)
            v_try = v0 + rng.normal(scale=1e-3, size=v0.shape)
        last_res = minimize(_obj, v_try, jac=_jac, method='SLSQP',
                            constraints=cons,
                            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
        if last_res.success:
            sol_try = last_res.x.reshape((n, 3))
            score_try = sol_try[:, 2].sum()
            if score_try > best_score:
                best_score = score_try
                best_sol = sol_try.copy()
    # Fallback if none succeeded
    if best_sol is None and last_res is not None:
        best_sol = last_res.x.reshape((n, 3))
        best_score = best_sol[:, 2].sum()

    # Greedy geometric insertion initializer (inspired by Inspiration 1)
    def greedy_initializer(n, max_iters=20000):
        circles = []
        for _ in range(max_iters):
            if len(circles) >= n:
                break
            x, y = random.random(), random.random()
            r = min(x, 1 - x, y, 1 - y)
            for cx, cy, cr in circles:
                dist = math.hypot(x - cx, y - cy)
                r = min(r, dist - cr)
            if r > 1e-4:
                circles.append([x, y, r])
        # Fill any remaining slots on a small placeholder grid
        if len(circles) < n:
            missing = n - len(circles)
            grid = int(math.ceil(math.sqrt(missing)))
            small_r = 1.0 / (2 * grid + 2)
            for idx in range(missing):
                row, col = divmod(idx, grid)
                xx = (col + 1) / (grid + 1)
                yy = (row + 1) / (grid + 1)
                circles.append([xx, yy, small_r])
        return np.array(circles[:n])

    # Greedy‐SLSQP refinement
    greed_circles = greedy_initializer(n)
    v_greed = np.hstack((greed_circles[:,0], greed_circles[:,1], greed_circles[:,2]))
    res_greed = minimize(_obj, v_greed, method='SLSQP', constraints=cons,
                         options={'maxiter': 100, 'ftol': 1e-7, 'disp': False})
    sol_greed = res_greed.x.reshape((n, 3))
    score_greed = sol_greed[:,2].sum()
    if score_greed > best_score:
        print(f"[circle_packing32] Greedy‐SLSQP improved sum_radii from {best_score:.6f} to {score_greed:.6f}")
        best_sol = sol_greed
        best_score = score_greed

    # Physics‐based repulsive initializer (inspired by Inspiration 2)
    def physics_initializer(n, init_r=best_score/n, steps=1000, dt=0.01):
        pts = np.random.rand(n, 2)
        for _ in range(steps):
            forces = np.zeros_like(pts)
            for i in range(n):
                for j in range(i+1, n):
                    d = pts[i] - pts[j]
                    dist = math.hypot(d[0], d[1])
                    if dist < 1e-8:
                        continue
                    overlap = 2 * init_r - dist
                    if overlap > 0:
                        f = d / dist * overlap
                        forces[i] += f
                        forces[j] -= f
            pts += dt * forces
            pts = np.clip(pts, init_r, 1 - init_r)
        tree_phys = KDTree(pts)
        dists, _ = tree_phys.query(pts, k=2)
        d_nn = dists[:, 1]
        d_bd = np.minimum.reduce([pts[:,0], 1-pts[:,0], pts[:,1], 1-pts[:,1]])
        radii_phys = 0.5 * np.minimum(d_nn, d_bd)
        return np.hstack((pts, radii_phys.reshape(-1,1)))

    phys_circles = physics_initializer(n)
    v_phys = phys_circles.ravel()
    res_phys = minimize(_obj, v_phys, method='SLSQP', constraints=cons,
                        options={'maxiter': 100, 'ftol': 1e-7, 'disp': False})
    sol_phys = res_phys.x.reshape((n, 3))
    score_phys = sol_phys[:,2].sum()
    if score_phys > best_score:
        print(f"[circle_packing32] Physics‐SLSQP improved sum_radii from {best_score:.6f} to {score_phys:.6f}")
        best_sol = sol_phys
        best_score = score_phys

    # Final greedy radius-inflation polish (half-step relaxations)
    sol = best_sol
    pts = sol[:, :2]
    radii = sol[:, 2].copy()
    for _ in range(100):
        tree = KDTree(pts)
        grown = 0
        for i in range(n):
            p = pts[i]
            # distance to boundary
            d_edge = min(p[0], p[1], 1 - p[0], 1 - p[1])
            # nearest neighbor margins
            dists, idxs = tree.query(p, k=n)
            min_sep = np.inf
            for kk in range(1, n):
                j = idxs[kk]
                sep = dists[kk] - radii[j]
                if sep < min_sep:
                    min_sep = sep
            max_r = max(0.0, min(d_edge, min_sep))
            if max_r > radii[i] + 1e-6:
                radii[i] = 0.5 * (radii[i] + max_r)
                grown += 1
        if grown == 0:
            break
    # return polished solution
    return np.column_stack((pts, radii.reshape(-1, 1)))


# EVOLVE-BLOCK-END
