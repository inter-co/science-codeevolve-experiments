# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog  # LP solver for local radius optimization
from scipy.stats import qmc         # Sobol sampler for initial positions
from scipy.spatial import KDTree    # KDTree for neighbor search in annealing

# Physics-inspired Simulated Annealing approach for 32-circle packing
# Implements a stochastic local search with annealing to maximize total radius

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square using simulated annealing.

    Returns:
        circles: np.array of shape (32,3), where each row is (x, y, r).
    """
    n = 32
    np.random.seed(42)
    # Initialize positions using Sobol low-discrepancy sampling, scaled to avoid boundaries
    sampler = qmc.Sobol(d=2, scramble=True, seed=42)
    xy = qmc.scale(sampler.random_base2(m=5), [0.1, 0.1], [0.9, 0.9])
    # Physics-based repulsion initialization
    k_repulse = 1e-2
    dt = 0.02
    init_iters = 2000
    for _ in range(init_iters):
        tree_init = KDTree(xy)
        forces = np.zeros_like(xy)
        for ii in range(n):
            neighbors = tree_init.query_ball_point(xy[ii], 0.5)
            for jj in neighbors:
                if jj == ii:
                    continue
                diff = xy[ii] - xy[jj]
                dist = np.linalg.norm(diff) + 1e-6
                # Inverse-square repulsion
                f = (diff / dist) * (k_repulse / (dist**2))
                forces[ii] += f
            # Soft wall repulsion with margin
            margin = 0.1
            if xy[ii,0] < margin:
                forces[ii,0] += k_repulse * (margin - xy[ii,0])**2
            if xy[ii,0] > 1 - margin:
                forces[ii,0] -= k_repulse * (xy[ii,0] - (1 - margin))**2
            if xy[ii,1] < margin:
                forces[ii,1] += k_repulse * (margin - xy[ii,1])**2
            if xy[ii,1] > 1 - margin:
                forces[ii,1] -= k_repulse * (xy[ii,1] - (1 - margin))**2
        xy += dt * forces
        xy = np.clip(xy, 0.0, 1.0)
    # Initialize radii as half the nearest‐neighbor or boundary distance
    D_init = xy[:, None, :] - xy[None, :, :]
    dist_init = np.sqrt((D_init**2).sum(axis=2))
    np.fill_diagonal(dist_init, np.inf)
    nn_dist = dist_init.min(axis=1)
    boundary_init = np.minimum.reduce([xy[:,0], 1 - xy[:,0], xy[:,1], 1 - xy[:,1]])
    r = 0.5 * np.minimum(nn_dist, boundary_init)
    best_xy, best_r = xy.copy(), r.copy()
    current_score = r.sum()
    best_score = current_score

    # Annealing parameters
    T0, Tf = 1.0, 1e-3
    max_iter = 30000  # increased iterations for deeper exploration

    for k in range(max_iter):
        # Periodic LP‐based radius refinement every 5000 steps
        if k > 0 and k % 5000 == 0:
            D_loop = xy[:, None, :] - xy[None, :, :]
            dist_loop = np.sqrt((D_loop**2).sum(axis=2))
            boundary_loop = np.minimum.reduce([xy[:,0], 1 - xy[:,0], xy[:,1], 1 - xy[:,1]])
            # Build LP constraints
            A_lp, b_lp = [], []
            for ii in range(n):
                for jj in range(ii+1, n):
                    row_lp = np.zeros(n)
                    row_lp[ii] = 1; row_lp[jj] = 1
                    A_lp.append(row_lp); b_lp.append(dist_loop[ii, jj])
            for ii in range(n):
                row_lp = np.zeros(n)
                row_lp[ii] = 1
                A_lp.append(row_lp); b_lp.append(boundary_loop[ii])
            A_lp = np.array(A_lp); b_lp = np.array(b_lp)
            # Setup LP objective: maximize sum of radii -> minimize -sum r_i
            c_lp = -np.ones(n)
            res_lp = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=(0, None), method='highs')
            if res_lp.success:
                r = res_lp.x
                current_score = r.sum()
                if current_score > best_score:
                    best_score = current_score
                    best_xy, best_r = xy.copy(), r.copy()

        # Mid-annealing LP refinement at the halfway point
        if k == max_iter // 2:
            # Build LP constraints for current centers to re-optimize radii
            D_mid = xy[:, None, :] - xy[None, :, :]
            dist_mid = np.sqrt((D_mid**2).sum(axis=2))
            boundary_mid = np.minimum.reduce([
                xy[:,0], 1 - xy[:,0], xy[:,1], 1 - xy[:,1]
            ])
            A_mid, b_mid = [], []
            for ii in range(n):
                for jj in range(ii+1, n):
                    row_mid = np.zeros(n)
                    row_mid[ii] = 1; row_mid[jj] = 1
                    A_mid.append(row_mid); b_mid.append(dist_mid[ii, jj])
            for ii in range(n):
                row_mid = np.zeros(n)
                row_mid[ii] = 1
                A_mid.append(row_mid); b_mid.append(boundary_mid[ii])
            A_mid = np.array(A_mid); b_mid = np.array(b_mid)
            c_mid = -np.ones(n)
            res_mid = linprog(
                c_mid, A_ub=A_mid, b_ub=b_mid,
                bounds=(0, None), method='highs'
            )
            if res_mid.success:
                r = res_mid.x
                current_score = r.sum()
                if current_score > best_score:
                    best_score = current_score
                    best_xy, best_r = xy.copy(), r.copy()

        # Exponential cooling
        T = T0 * (Tf / T0) ** (k / (max_iter - 1))
        # Pick a circle to perturb, biasing toward smaller radii
        inv_r = 1.0 / (r + 1e-6)
        probs = inv_r / inv_r.sum()
        i = np.random.choice(n, p=probs)
        old_xy, old_r = xy[i].copy(), r[i]

        # Propose temperature‐adaptive random shift in position and radius
        step_scale = np.sqrt(T / T0)
        delta_xy = (np.random.rand(2) - 0.5) * 0.05 * (1 + step_scale)
        delta_r  = (np.random.rand()    - 0.5) * 0.05 * (1 + step_scale)
        new_r = max(0.0, old_r + delta_r)
        new_xy = old_xy + delta_xy

        # Enforce containment in [0,1] respecting new radius
        new_xy = np.minimum(np.maximum(new_xy, new_r), 1 - new_r)

        # Apply proposal
        xy[i], r[i] = new_xy, new_r

        # Check feasibility with KDTree neighbor search
        tree = KDTree(xy)
        # radius for neighbor search: new_r + maximum current radius
        max_rad = r.max()
        neighbors = tree.query_ball_point(xy[i], new_r + max_rad)
        overlap = False
        for j in neighbors:
            if j == i:
                continue
            if np.linalg.norm(xy[i] - xy[j]) < new_r + r[j]:
                overlap = True
                break

        if not overlap:
            new_score = r.sum()
            delta_score = new_score - current_score
            # Metropolis acceptance criterion
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_xy, best_r = xy.copy(), r.copy()
            else:
                # Reject move
                xy[i], r[i] = old_xy, old_r
        else:
            # Revert on overlap
            xy[i], r[i] = old_xy, old_r

    # Local radius optimization via LP to refine best configuration
    coords = best_xy
    n = coords.shape[0]
    # Precompute all non‐overlap index‐pairs for LP constraint construction
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    # Compute distance matrix between circle centers
    D = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((D ** 2).sum(axis=2))
    # Boundary distances for each center
    boundary = np.minimum.reduce([coords[:,0], 1 - coords[:,0], coords[:,1], 1 - coords[:,1]])
    # Setup LP: maximize sum r_i -> minimize -sum r_i
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    # Non-overlap constraints using precomputed pairs
    for (i, j) in pairs:
        row = np.zeros(n)
        row[i] = 1; row[j] = 1
        A_ub.append(row)
        b_ub.append(dist_matrix[i, j])
    # Boundary constraints: r_i <= boundary_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(boundary[i])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    # Solve LP with SciPy
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if res.success:
        r_opt = res.x
    else:
        r_opt = best_r.copy()

    # Quick hill-climbing on positions with LP-based radii optimization
    best_coords = coords.copy()
    best_radii = r_opt.copy()
    best_score = best_radii.sum()
    hill_iters = 800  # deeper local search now that LP cost is amortized
    for _ in range(hill_iters):
        # Pick one circle and perturb its center slightly
        ih = np.random.randint(n)
        new_coords = best_coords.copy()
        delta = (np.random.rand(2) - 0.5) * 0.02
        new_coords[ih] = np.clip(new_coords[ih] + delta, 0, 1)

        # Rebuild LP constraints for new_coords
        D_h = new_coords[:, None, :] - new_coords[None, :, :]
        dist_h = np.sqrt((D_h**2).sum(axis=2))
        boundary_h = np.minimum.reduce([
            new_coords[:,0], 1 - new_coords[:,0],
            new_coords[:,1], 1 - new_coords[:,1]
        ])
        A_h, b_h = [], []
        # Non-overlap constraints using precomputed pairs
        for (ii, jj) in pairs:
            row = np.zeros(n)
            row[ii] = 1; row[jj] = 1
            A_h.append(row); b_h.append(dist_h[ii, jj])
        # Boundary constraints
        for ii in range(n):
            row = np.zeros(n)
            row[ii] = 1
            A_h.append(row); b_h.append(boundary_h[ii])
        A_h = np.array(A_h)
        b_h = np.array(b_h)

        # Solve LP for these new centers
        res_h = linprog(c, A_ub=A_h, b_ub=b_h, bounds=(0, None), method='highs')
        if res_h.success:
            r_h = res_h.x
            score_h = r_h.sum()
            if score_h > best_score:
                best_score = score_h
                best_coords = new_coords
                best_radii = r_h

    circles = np.hstack((best_coords, best_radii.reshape(-1, 1)))
    return circles


# EVOLVE-BLOCK-END
