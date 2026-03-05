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
    # run SLSQP on SA‐derived seed
    res = minimize(_obj, v0, method='SLSQP', constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})  # Further increased iterations & tighter tolerance
    sol = res.x.reshape((n, 3))
    # keep best solution and its score
    best_sol = sol
    best_score = sol[:, 2].sum()

    # Greedy‐SLSQP refinement (inspired by Inspiration 1)
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
        # fill any remaining slots on a small placeholder grid
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

    # generate and refine greedy seed
    greed_circles = greedy_initializer(n)
    v_greed = greed_circles.ravel()
    res_greed = minimize(_obj, v_greed, method='SLSQP', constraints=cons,
                         options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
    sol_greed = res_greed.x.reshape((n, 3))
    score_greed = sol_greed[:, 2].sum()
    if score_greed > best_score:
        best_sol = sol_greed

    # -------------------------------------------------------------------------
    # Additional hexagonal‐lattice seed + SLSQP (inspired by Inspirations 1/2)
    # -------------------------------------------------------------------------
    def hex_initializer(n):
        spacing = (2 / (math.sqrt(3) * n))**0.5
        dx = spacing
        dy = spacing * math.sqrt(3) / 2
        pts = []
        row = 0
        y = 0.0
        while y <= 1 and len(pts) < n:
            offset = (spacing / 2) if (row % 2) else 0.0
            x = offset
            while x <= 1 and len(pts) < n:
                pts.append((x, y))
                x += dx
            row += 1
            y = row * dy
        pts = np.array(pts[:n])
        tree = KDTree(pts)
        # nearest‐neighbor distances
        d_nn = np.array([tree.query(p, k=2)[0][1] for p in pts])
        # border distances
        d_border = np.minimum.reduce([pts[:,0], 1-pts[:,0], pts[:,1], 1-pts[:,1]])
        radii = 0.5 * np.minimum(d_nn, d_border)
        return np.hstack([pts, radii.reshape(-1,1)])

    hex_circles = hex_initializer(n)
    v_hex = hex_circles.ravel()
    res_hex = minimize(_obj, v_hex, method='SLSQP', constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-8, 'disp': False})
    sol_hex = res_hex.x.reshape((n, 3))
    score_hex = sol_hex[:, 2].sum()
    if score_hex > best_score:
        best_sol = sol_hex

    return best_sol


# EVOLVE-BLOCK-END
