# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import minimize, linprog  # added for local SLSQP refinement and LP radii refinement
from scipy.spatial import cKDTree  # for fast nearest-neighbor queries
from scipy.spatial.distance import pdist, squareform  # for LP-based radii refinement
from scipy.stats.qmc import Halton  # for Halton sequence initialization

def inflate_and_relax(centers: np.ndarray, radii: np.ndarray, max_iter: int = 200, alpha: float = 0.05):
    """
    Inflation and repulsive relaxation based on physics-inspired method.
    """
    n = centers.shape[0]
    for t in range(max_iter):
        # distance to walls
        d_wall = np.minimum.reduce([
            centers[:, 0],
            1 - centers[:, 0],
            centers[:, 1],
            1 - centers[:, 1]
        ])

        # pairwise differences and distances
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2)) + np.eye(n)
        separation = dist - (radii[:, None] + radii[None, :])

        # determine growth allowance and inflate radii
        d_circle = np.min(separation + np.diag([1e6] * n), axis=1)
        growth = alpha * np.minimum(d_wall, d_circle)
        radii += growth

        # repulsive relaxation to resolve any overlaps
        overlap = separation < 0
        for i in range(n):
            for j in range(i + 1, n):
                if overlap[i, j]:
                    vec = diff[i, j]
                    norm = np.linalg.norm(vec)
                    if norm == 0:
                        vec = np.random.randn(2)
                        norm = np.linalg.norm(vec)
                    push = 0.5 * (radii[i] + radii[j] - norm) * (vec / norm)
                    centers[i] += push
                    centers[j] -= push

        # enforce boundary constraints
        centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
        alpha *= 0.995

    return centers, radii

def pack_with_greedy(n, samples=1000):
    """
    Greedy incremental filling: sample random points and choose best radius.
    """
    placed = []
    for _ in range(n):
        best_r = -1.0
        best_pt = None
        for _ in range(samples):
            x0, y0 = np.random.rand(), np.random.rand()
            r0 = min(x0, 1-x0, y0, 1-y0)
            for xi, yi, ri in placed:
                r0 = min(r0, np.hypot(x0-xi, y0-yi) - ri)
            if r0 > best_r:
                best_r, best_pt = r0, (x0, y0)
        placed.append((best_pt[0], best_pt[1], best_r))
    return np.array(placed)

def refine_radii_lp(pos: np.ndarray) -> np.ndarray:
    """
    LP-based radii refinement for fixed centers.
    """
    n = pos.shape[0]
    c = -np.ones(n)
    A_ub, b_ub = [], []
    # boundary constraints
    for i, (xi, yi) in enumerate(pos):
        vec = np.zeros(n); vec[i] = 1
        A_ub.append(vec.copy()); b_ub.append(xi)
        A_ub.append(vec.copy()); b_ub.append(1 - xi)
        A_ub.append(vec.copy()); b_ub.append(yi)
        A_ub.append(vec.copy()); b_ub.append(1 - yi)
    # non-overlap constraints
    dmat = squareform(pdist(pos))
    for i in range(n):
        for j in range(i+1, n):
            vec = np.zeros(n); vec[i] = 1; vec[j] = 1
            A_ub.append(vec); b_ub.append(dmat[i, j])
    A_ub = np.vstack(A_ub); b_ub = np.array(b_ub)
    bounds_lp = [(0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method="highs")
    if res.success:
        radii = res.x
    else:
        radii = np.zeros(n)
    return np.column_stack((pos, radii))

# Halton-based pipeline: low-discrepancy initialization + LP radii refinement
def pack_with_halton(n, seed=42):
    sampler = Halton(d=2, scramble=True, seed=seed)
    pts = sampler.random(n)
    # refine radii via LP
    return refine_radii_lp(pts)

# Exploration: Use Simulated Annealing for dynamic circle placement and radius adjustment
# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Simulated Annealing based circle packing: dynamic positioning and radius adjustment for 32 circles.
    """
    # reproducibility
    np.random.seed(42)
    random.seed(42)
    n = 32
    max_iters = 120000  # further increased iterations leveraging faster radii evaluation
    T_initial = 0.05
    T_final = 1e-3

    # Initial positions (random uniform)
    pos = np.random.rand(n, 2)

    # Function to compute radii given positions
    def compute_radii(pts: np.ndarray) -> np.ndarray:
        # vectorized distance to square boundary
        bound_dist = np.minimum.reduce([
            pts[:, 0], 1 - pts[:, 0],
            pts[:, 1], 1 - pts[:, 1]
        ])
        # nearest neighbor distances via KD-tree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)  # returns [self, nearest_other]
        neigh_dist = 0.5 * dists[:, 1]
        # radius is limited by both boundary and nearest neighbor
        radii = np.minimum(bound_dist, neigh_dist)
        return np.maximum(radii, 0.0)

    # Initialize radii and best-known configuration
    radii = compute_radii(pos)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)

    # Simulated Annealing loop
    sum_current = best_sum
    for it in range(1, max_iters + 1):
        # exponential cooling for smoother decay
        T = T_initial * ((T_final / T_initial) ** (it / max_iters))

        # occasional shake to escape local minima
        if it % 8000 == 0:  # more frequent shake to escape local minima
            pos += np.random.uniform(-0.01, 0.01, size=pos.shape)
            pos = np.clip(pos, 0.0, 1.0)
            radii = compute_radii(pos)
            sum_current = np.sum(radii)

        # Randomly perturb one circle's position
        i = random.randrange(n)
        old_x, old_y = pos[i].copy()
        step = np.sqrt(T)  # step size ∝ sqrt(temperature)
        pos[i, 0] = np.clip(old_x + random.uniform(-step, step), 0.0, 1.0)
        pos[i, 1] = np.clip(old_y + random.uniform(-step, step), 0.0, 1.0)

        # Recompute radii and objective
        new_radii = compute_radii(pos)
        new_sum = np.sum(new_radii)
        delta = new_sum - sum_current

        # Accept or reject move
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-8)):
            sum_current = new_sum
            radii = new_radii
            # Update best if improved
            if new_sum > best_sum:
                best_sum = new_sum
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # Revert move on rejection
            pos[i] = [old_x, old_y]

    # Further refine best configuration via inflation-relaxation
    best_pos, best_radii = inflate_and_relax(
        best_pos.copy(), best_radii.copy(),
        max_iter=200, alpha=0.05
    )

    # Construct result array and then locally refine via SLSQP
    z0 = np.concatenate([
        best_pos[:, 0],
        best_pos[:, 1],
        best_radii
    ])
    n = 32
    # bounds for xi, yi ∈ [0,1] and ri ∈ [0,0.5]
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n

    # build containment and non-overlap constraints
    constraints = []
    for i in range(n):
        # xi - ri ≥ 0, 1 - xi - ri ≥ 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i:  z[i]         - z[2*n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i]     - z[2*n + i]})
        # yi - ri ≥ 0, 1 - yi - ri ≥ 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i:  z[n + i]    - z[2*n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[n + i] - z[2*n + i]})
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: np.hypot(z[i] - z[j], z[n+i] - z[n+j]) 
                                           - (z[2*n + i] + z[2*n + j])
            })

    # objective: maximize sum of radii ⇒ minimize negative sum
    def obj(z):
        return -np.sum(z[2*n:])

    # run a short SLSQP refine
    result = minimize(
        obj, z0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
    )
    z_opt = result.x
    circles = np.vstack([
        z_opt[:n],       # x coords
        z_opt[n:2*n],    # y coords
        z_opt[2*n:]      # radii
    ]).T
    # Final physics-based inflation + relaxation to squeeze out extra radius
    centers = circles[:, :2]
    radii_opt = circles[:, 2].copy()
    centers, radii_opt = inflate_and_relax(
        centers, radii_opt,
        max_iter=200,    # extended extra pass
        alpha=0.03       # slightly larger growth step
    )
    # SA-based result
    sa_circles = np.hstack((centers, radii_opt[:, None]))
    sa_sum = sa_circles[:, 2].sum()

    # Greedy+LP pipeline
    greedy_pos = pack_with_greedy(n, samples=2000)[:, :2]
    greedy_circles = refine_radii_lp(greedy_pos)
    greedy_sum = greedy_circles[:, 2].sum()

    # Halton+LP pipeline
    halton_circles = pack_with_halton(n)
    halton_sum = halton_circles[:, 2].sum()

    # Select best configuration among pipelines
    best_circles = sa_circles
    best_sum = sa_sum
    if greedy_sum > best_sum:
        best_circles, best_sum = greedy_circles, greedy_sum
    if halton_sum > best_sum:
        best_circles, best_sum = halton_circles, halton_sum

    # Final inflation-relaxation on the selected best solution
    centers, radii = best_circles[:, :2], best_circles[:, 2].copy()
    centers, radii = inflate_and_relax(centers, radii,
                                       max_iter=150,   # extra light refinement
                                       alpha=0.04)     # small growth step
    best_circles = np.hstack((centers, radii[:, None]))

    # Multi-start SLSQP polishing on the refined best solution
    z_init = np.concatenate([best_circles[:,0],
                             best_circles[:,1],
                             best_circles[:,2]])
    best_z = z_init.copy()
    best_multi_sum = best_circles[:, 2].sum()
    for k in range(3):
        if k > 0:
            # jitter previous best
            z0 = best_z + np.random.normal(scale=0.005, size=z_init.shape)
            # enforce bounds: [0,1] for x,y and [0,0.5] for r
            z0[:2*n] = np.clip(z0[:2*n], 0.0, 1.0)
            z0[2*n:] = np.clip(z0[2*n:], 0.0, 0.5)
        else:
            z0 = z_init
        res = minimize(obj, z0,
                       method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 200,
                                'ftol': 1e-7,
                                'disp': False})
        if res.success:
            z = res.x
            s = z[2*n:].sum()
            if s > best_multi_sum:
                best_multi_sum = s
                best_z = z.copy()

    # rebuild circles from best_z
    best_circles = np.vstack([
        best_z[:n],
        best_z[n:2*n],
        best_z[2*n:]
    ]).T

    return best_circles


# EVOLVE-BLOCK-END
