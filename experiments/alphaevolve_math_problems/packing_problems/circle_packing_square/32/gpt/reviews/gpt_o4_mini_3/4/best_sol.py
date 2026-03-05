# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import math
from scipy.spatial import KDTree
# Use low-discrepancy Sobol sequence for better initial coverage
from scipy.stats import qmc

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square using a physics-inspired
    repulsion simulation, then computes maximal radii based on nearest distances
    and boundaries.
    Returns:
        circles: np.ndarray of shape (32,3), where each row is (x, y, r).
    """
    # Fixed seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    n = 32
    # Initialize positions inside [0.1, 0.9] via Sobol low-discrepancy sampling
    sampler = qmc.Sobol(d=2, scramble=True, seed=42)
    # random_base2(5) → 2^5 = 32 points
    # Widen initialization region toward edges (inspired by Inspiration 2)
    positions = qmc.scale(sampler.random_base2(m=5), [0.05, 0.05], [0.95, 0.95])
    # Add small jitter for initial dispersion (inspired by hybrid grid‐jitter)
    # Stronger jitter for initial variability (inspired by Inspiration 1)
    positions += np.random.uniform(-0.01, 0.01, positions.shape)
    positions = np.clip(positions, 0.0, 1.0)
    # Radii placeholder
    radii = np.zeros(n)

    # Simulation parameters (finer repulsion control - from Inspiration 2)
    k_repulse = 7e-3    # further increased repulsion strength
    dt = 0.015          # finer time step for stability under stronger forces
    iterations = 3000   # more steps for deeper dispersion

    # Physics-based repulsion loop
    for _ in range(iterations):
        tree = KDTree(positions)
        forces = np.zeros_like(positions)

        # Compute pairwise repulsive forces
        for i in range(n):
            neighbors = tree.query_ball_point(positions[i], r=0.2)
            for j in neighbors:
                if j == i:
                    continue
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff) + 1e-6
                # Inverse-square repulsion
                f_mag = k_repulse / (dist**2)
                forces[i] += (diff / dist) * f_mag

            # Boundary repulsion (soft walls—with larger margin)
            x, y = positions[i]
            # Reduced margin so circles can hug the edges for larger radii (Inspiration 2)
            margin = 0.05
            # left wall
            if x < margin:
                forces[i][0] += k_repulse * (margin - x)**2
            # right wall
            if x > 1 - margin:
                forces[i][0] -= k_repulse * (x - (1 - margin))**2
            # bottom wall
            if y < margin:
                forces[i][1] += k_repulse * (margin - y)**2
            # top wall
            if y > 1 - margin:
                forces[i][1] -= k_repulse * (y - (1 - margin))**2

        # Update positions
        positions += dt * forces
        # Keep inside [0,1]
        positions = np.clip(positions, 0.0, 1.0)

    # Build final KDTree for radius computation
    tree = KDTree(positions)

    # === Global radius optimization via LP (maximize sum of r_i) ===
    coords = positions
    n = coords.shape[0]
    # Pairwise center distances
    D = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((D ** 2).sum(axis=2))
    # Distances to the four walls
    boundary = np.minimum.reduce([
        coords[:, 0],
        1 - coords[:, 0],
        coords[:, 1],
        1 - coords[:, 1]
    ])
    # Build LP: minimize -sum(r) subject to r_i + r_j <= dist_ij,  r_i <= boundary_i
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist_matrix[i, j])
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(boundary[i])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if res.success:
        r_opt = res.x
    else:
        # Fallback to greedy assignment if LP fails
        # half nearest-neighbor on the final KDTree
        r_opt = np.zeros(n)
        for i in range(n):
            dists, _ = tree.query(coords[i], k=2)
            bd = min(coords[i,0], 1-coords[i,0], coords[i,1], 1-coords[i,1])
            r_opt[i] = min(dists[1]/2.0, bd)
    circles = np.hstack((coords, r_opt.reshape(-1, 1)))

    # Quick simulated-annealing on positions to refine LP result (inspired by Inspiration 1)
    from scipy.spatial import KDTree as _KDTree
    def _compute_radii(pts):
        tree_loc = _KDTree(pts)
        rs_loc = np.zeros(n)
        for ii, p in enumerate(pts):
            d_edge = min(p[0], p[1], 1-p[0], 1-p[1])
            dists, _ = tree_loc.query(p, k=2)
            d_nn = dists[1] if len(dists)>1 else np.inf
            rs_loc[ii] = min(d_edge, d_nn/2)
        return rs_loc

    # Simulated annealing on positions with temperature schedule
    steps_sa = 5000
    T0, Tf = 0.1, 1e-4
    pts_sa = coords.copy()
    rs_sa = r_opt.copy()
    sa_score = rs_sa.sum()
    for k in range(steps_sa):
        T = T0 * (Tf / T0) ** (k / steps_sa)
        i = random.randrange(n)
        old_p = pts_sa[i].copy()
        pts_sa[i] += np.random.randn(2) * 0.005
        pts_sa[i] = np.clip(pts_sa[i], 0.0, 1.0)
        new_rs = _compute_radii(pts_sa)
        new_score = new_rs.sum()
        delta = new_score - sa_score
        # Metropolis acceptance
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-8)):
            sa_score, rs_sa = new_score, new_rs
        else:
            pts_sa[i] = old_p

    coords, r_opt = pts_sa, rs_sa
    # Post–SA radius inflation (inspired by Inspiration 1 & 2)
    for _ in range(100):
        tree = KDTree(coords)
        grown = False
        for i in range(n):
            p = coords[i]
            dists, _ = tree.query(p, k=n)
            max_r = min(p[0], p[1], 1-p[0], 1-p[1], dists[1:].min()/2)
            if max_r > r_opt[i] + 1e-6:
                r_opt[i] = (r_opt[i] + max_r) / 2
                grown = True
        if not grown:
            break
    circles = np.hstack((coords, r_opt.reshape(-1,1)))

    # Polishing with SLSQP nonlinear optimization (inspired by inspiration programs)
    from scipy.optimize import minimize
    n = coords.shape[0]
    # Pack into flat vector [x0, y0, r0, x1, y1, r1, ...]
    v0 = circles.ravel()
    def _obj(v):
        return -np.sum(v[2::3])  # maximize sum of radii → minimize negative sum

    cons = []
    # Boundary constraints: r_i ≤ x_i ≤ 1-r_i and r_i ≤ y_i ≤ 1-r_i
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
    # Non-overlap constraints: (x_i−x_j)^2 + (y_i−y_j)^2 ≥ (r_i+r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j:
                    (v[3*i] - v[3*j])**2
                  + (v[3*i+1] - v[3*j+1])**2
                  - (v[3*i+2] + v[3*j+2])**2
            })

    res2 = minimize(
        _obj, v0, method='SLSQP', constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
    )
    if res2.success:
        circles = res2.x.reshape((n, 3))
    return circles


# EVOLVE-BLOCK-END
