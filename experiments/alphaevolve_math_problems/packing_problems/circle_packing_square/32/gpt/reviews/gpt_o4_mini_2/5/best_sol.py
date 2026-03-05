# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.stats import qmc  # Sobol sequence for diverse restarts
import random
import math
from scipy.spatial import cKDTree
from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform

# Stochastic hill-climbing refinement with adaptive move scale decay
def refine_hill_climb(init_centers, init_radii, iters=5000, move_scale=0.01, seed=0):
    """
    Stochastic hill-climbing with decaying move_scale to escape local traps.
    """
    n = init_centers.shape[0]
    rng = np.random.RandomState(seed)
    centers = init_centers.copy()
    radii = init_radii.copy()
    best_sum = radii.sum()
    for t in range(iters):
        i = rng.randint(n)
        # adaptive move scale decays over iterations
        scale = move_scale * (1 - t / iters)
        # propose random move
        new_c = centers[i] + rng.normal(scale=scale, size=2)
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


# Physics-inspired inflation and repulsion based relaxation (inspired by Inspiration 1 & 2)
def inflate_and_relax(centers: np.ndarray, radii: np.ndarray, max_iter: int = 200, alpha: float = 0.05):
    """
    Inflation and repulsive relaxation to further expand radii without overlap.
    """
    n = centers.shape[0]
    for _ in range(max_iter):
        # compute distance to walls
        d_wall = np.minimum.reduce([
            centers[:, 0],
            1 - centers[:, 0],
            centers[:, 1],
            1 - centers[:, 1]
        ])
        # pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + np.eye(n)
        separation = dist - (radii[:, None] + radii[None, :])
        # allowable growth per circle
        d_circle = np.min(separation + np.diag([1e6]*n), axis=1)
        growth = alpha * np.minimum(d_wall, d_circle)
        radii += growth
        # repulsion for overlaps
        overlap = separation < 0
        for i in range(n):
            for j in range(i+1, n):
                if overlap[i, j]:
                    vec = diff[i, j]
                    norm = np.linalg.norm(vec)
                    if norm < 1e-8:
                        vec = np.random.randn(2)
                        norm = np.linalg.norm(vec)
                    push = 0.5 * (radii[i] + radii[j] - norm) * (vec / norm)
                    centers[i] += push
                    centers[j] -= push
        # enforce boundaries
        centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
        alpha *= 0.995
    return centers, radii

# LP-based radii refinement for fixed centers (improve objective exactly)
def refine_radii_lp(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    # objective: maximize sum(r) ⇒ minimize -sum(r)
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    # boundary constraints: ri ≤ xi, ri ≤ 1−xi, ri ≤ yi, ri ≤ 1−yi
    for i, (xi, yi) in enumerate(pos):
        vec = np.zeros(n); vec[i] = 1.0
        A_ub.append(vec.copy()); b_ub.append(xi)
        A_ub.append(vec.copy()); b_ub.append(1 - xi)
        A_ub.append(vec.copy()); b_ub.append(yi)
        A_ub.append(vec.copy()); b_ub.append(1 - yi)
    # non-overlap: ri + rj ≤ dist(i,j)
    dmat = squareform(pdist(pos))
    for i in range(n):
        for j in range(i+1, n):
            vec = np.zeros(n)
            vec[i], vec[j] = 1.0, 1.0
            A_ub.append(vec); b_ub.append(dmat[i,j])
    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)
    bounds_lp = [(0, None)] * n
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    if res_lp.success:
        radii_lp = res_lp.x
    else:
        # fallback: boundary-limited radii
        radii_lp = np.minimum.reduce([pos[:,0], 1-pos[:,0], pos[:,1], 1-pos[:,1]])
    return np.hstack([pos, radii_lp.reshape(-1,1)])

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
    random.seed(42)
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

    # consolidated NonlinearConstraint for containment & non-overlap
    def constraint_vector(z):
        x = z[:n]
        y = z[n:2*n]
        r = z[2*n:]
        c = []
        # containment: xi - ri ≥ 0, (1-xi) - ri ≥ 0, yi - ri ≥ 0, (1-yi) - ri ≥ 0
        c.extend(x - r)
        c.extend((1 - x) - r)
        c.extend(y - r)
        c.extend((1 - y) - r)
        # non-overlap: dist_ij - (ri + rj) ≥ 0
        for i in range(n):
            for j in range(i + 1, n):
                c.append(np.hypot(x[i] - x[j], y[i] - y[j]) - (r[i] + r[j]))
        return np.array(c)

    nl_cons = NonlinearConstraint(constraint_vector, 0, np.inf)
    constraints = [nl_cons]

    # objective: maximize Σ ri  ⇒  minimize negative sum
    def objective(z):
        return -np.sum(z[2*n:])

    # analytical gradient for the objective (zeros except −1 for each radius)
    def objective_grad(z):
        grad = np.zeros_like(z)
        grad[2*n:] = -1.0
        return grad

    # Quick Simulated Annealing for global positioning (from inspirations)
    def compute_radii_positions(pts):
        # wall distances
        d_wall = np.minimum.reduce([pts[:,0], 1-pts[:,0], pts[:,1], 1-pts[:,1]])
        # nearest neighbor via cKDTree
        tree_sa = cKDTree(pts)
        d_sa, _ = tree_sa.query(pts, k=2)
        neigh = d_sa[:,1] / 2.0
        return np.minimum(d_wall, neigh)

    # run short SA to get a diverse starting arrangement
    pos_sa = hex_pts.copy()
    rad_sa = compute_radii_positions(pos_sa)
    best_sa_sum = rad_sa.sum()
    best_sa_pos = pos_sa.copy()
    best_sa_rad = rad_sa.copy()
    curr_pos = pos_sa.copy()
    curr_rad = rad_sa.copy()
    curr_sum = best_sa_sum
    T0, Tf, sa_iters = 0.02, 1e-4, 20000
    for it in range(sa_iters):
        T = T0 * ((Tf/T0) ** (it/sa_iters))
        i_sa = random.randrange(n)
        old = curr_pos[i_sa].copy()
        step = math.sqrt(T)
        # random move
        curr_pos[i_sa, 0] = np.clip(old[0] + random.uniform(-step, step), 0, 1)
        curr_pos[i_sa, 1] = np.clip(old[1] + random.uniform(-step, step), 0, 1)
        new_rad = compute_radii_positions(curr_pos)
        new_sum = new_rad.sum()
        dE = new_sum - curr_sum
        if dE > 0 or random.random() < math.exp(dE / max(T, 1e-8)):
            curr_rad = new_rad
            curr_sum = new_sum
            if new_sum > best_sa_sum:
                best_sa_sum = new_sum
                best_sa_pos = curr_pos.copy()
                best_sa_rad = curr_rad.copy()
        else:
            curr_pos[i_sa] = old
    x_sa = np.concatenate([best_sa_pos[:,0], best_sa_pos[:,1], best_sa_rad])

    # Multi-start SLSQP + hill-climb refinement (now 4 restarts, include SA)
    best_sum = -np.inf
    best_circles = None
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    for restart in range(4):
        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            rng = np.random.RandomState(42)
            jitter = rng.normal(scale=0.002, size=x0.shape)
            x_init = np.minimum(np.maximum(x0 + jitter, lows), highs)
        elif restart == 2:
            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            pts = sobol.random(n)
            tree2 = cKDTree(pts)
            d2, _ = tree2.query(pts, k=2)
            half_nn2 = d2[:,1] / 2.0
            boundary2 = np.minimum.reduce([pts[:,0], 1-pts[:,0], pts[:,1], 1-pts[:,1]])
            r2 = np.minimum(half_nn2, boundary2)
            x_init = np.concatenate([pts[:,0], pts[:,1], r2])
        else:
            # SA-derived start
            x_init = x_sa.copy()
        # enforce bounds
        x_init = np.minimum(np.maximum(x_init, lows), highs)

        res = minimize(
            objective, x_init,
            jac=objective_grad,           # supply exact gradient
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
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

    # Final inflation and relaxation to pack additional radius
    centers = best_circles[:, :2]
    radii = best_circles[:, 2]
    centers, radii = inflate_and_relax(centers, radii, max_iter=200, alpha=0.05)
    # Final LP-based radii refinement for best centers
    circles_lp = refine_radii_lp(centers)
    # Micro hill-climbing on LP result for final tweaks
    cir_final = refine_hill_climb(circles_lp[:, :2], circles_lp[:, 2],
                                  iters=2000, move_scale=0.001, seed=0)
    return cir_final


# EVOLVE-BLOCK-END
