# EVOLVE-BLOCK-START
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import numpy as np
from scipy.optimize import linprog  # Added for LP-based radius optimization
from scipy.optimize import minimize  # Added for nonlinear SLSQP refinement
import math

# Helper: physics-inspired repulsion + inflation refinement
def _physics_refine(pos, r, iterations=500, lr=0.1, dr=1e-3):
    n = pos.shape[0]
    for _ in range(iterations):
        # pairwise deltas and distances
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-8
        # overlap matrix: (r_i + r_j) - dist
        overlap = (r[:, None] + r[None, :]) - dist
        mask = overlap > 0
        # compute repulsive forces
        forces = (overlap[..., None] / dist[..., None] * delta) * (mask[..., None] * lr)
        net_force = np.sum(forces, axis=1)
        pos += net_force
        # enforce containment
        pos = np.minimum(np.maximum(pos, r[:, None]), 1 - r[:, None])
        # inflation step
        d_wall = np.minimum.reduce([pos[:,0], 1-pos[:,0], pos[:,1], 1-pos[:,1]])
        pair_clear = dist - r[None, :]
        min_pair_clear = np.min(pair_clear, axis=1)
        clearance = np.minimum(d_wall, min_pair_clear)
        r = np.minimum(r + dr, clearance)
    return pos, r

# Helper: deterministic hexagonal lattice seed for initial circle centers
def _hexagonal_seed(n):
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    dy = 1.0 / (rows + 1)
    dx = dy * math.sqrt(3) / 2
    points = []
    for i in range(rows):
        y = (i + 1) * dy
        x_offset = (i % 2) * dx
        for j in range(cols):
            x = (j + 1) * dx + x_offset
            if 0 < x < 1 and 0 < y < 1 and len(points) < n:
                points.append((x, y))
        if len(points) >= n:
            break
    return np.array(points[:n])

# Helper: local nonlinear SLSQP refinement on circles array
def _slsqp_refine(circles, maxiter=800):  # Increased SLSQP budget for finer tuning
    n = circles.shape[0]
    x0 = circles.flatten()
    # Objective: maximize sum of radii → minimize negative sum
    def obj(x):
        return -np.sum(x[2::3])
    cons = []
    # Boundary constraints for each circle
    for i in range(n):
        cons += [
            {'type': 'ineq', 'fun': lambda x, i=i: x[3*i+0] - x[3*i+2]},       # x - r ≥ 0
            {'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[3*i+0] + x[3*i+2])}, # x + r ≤ 1
            {'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]},       # y - r ≥ 0
            {'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[3*i+1] + x[3*i+2])}  # y + r ≤ 1
        ]
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[3*i+0] - x[3*j+0])**2 +
                    (x[3*i+1] - x[3*j+1])**2
                ) - (x[3*i+2] + x[3*j+2])**2
            })
    res = minimize(obj, x0, method='SLSQP', constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-9})  # Finer convergence tolerance
    if res.success:
        return res.x.reshape(-1,3)
    else:
        return circles

# Exploration: Using Centroidal Voronoi Tessellation (CVT) to seed circle centers
# followed by maximal inscribed radius computation per cell for packing.
def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square using CVT-based seeding:
    1. Initialize 32 random points.
    2. Iteratively perform Lloyd relaxations via bounded Voronoi cells.
    3. Compute each circle's radius as the maximal inscribed circle in its Voronoi cell
       constrained by neighbors and box boundaries.
    Returns:
        circles: np.array of shape (32,3), where each row is (x, y, r).
    """
    n = 32
    # Initialize deterministic hexagonal lattice seed for initial points
    rng_init = np.random.default_rng(123)  # deterministic jitter seed
    base_points = _hexagonal_seed(n)
    points = np.clip(
        base_points + rng_init.uniform(-0.005, 0.005, size=(n,2)),
        0.0, 1.0
    )
    # Define unit square for clipping infinite cells
    bbox = box(0, 0, 1, 1)

    # Create mirrored copies for bounded Voronoi approximation
    def mirror_points(pts):
        left = pts.copy(); left[:, 0] = -pts[:, 0]
        right = pts.copy(); right[:, 0] = 2 - pts[:, 0]
        bottom = pts.copy(); bottom[:, 1] = -pts[:, 1]
        top = pts.copy(); top[:, 1] = 2 - pts[:, 1]
        return np.vstack([pts, left, right, bottom, top])

    points_ext = mirror_points(points)

    # Lloyd-like CVT iterations
    for _ in range(800):  # Further increased CVT iterations for finer centroids
        vor = Voronoi(points_ext)
        new_pts = []
        for i in range(n):
            region = vor.regions[vor.point_region[i]]
            if not region or -1 in region:
                new_pts.append(points[i])
                continue
            # Build region polygon and clip to unit box
            poly_coords = [vor.vertices[v] for v in region]
            poly = Polygon(poly_coords).intersection(bbox)
            if poly.is_empty:
                new_pts.append(points[i])
            else:
                new_pts.append(np.array(poly.centroid.coords[0]))
        points = np.clip(np.array(new_pts), 0.0, 1.0)
        points_ext = mirror_points(points)

    # Phase 2: LP-based optimal radius assignment given positions
    # Maximize sum of radii subject to non-overlap and boundary constraints
    # Variables: radii r_i for each circle
    dists_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    # Build inequality constraints A_ub * r <= b_ub
    A_ub = []
    b_ub = []
    # Non-overlap constraints: r_i + r_j <= d_ij for i<j
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dists_matrix[i, j])
    # Boundary constraints: r_i <= x_i, 1-x_i, y_i, 1-y_i
    for i in range(n):
        xi, yi = points[i]
        for bound in (xi, 1 - xi, yi, 1 - yi):
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(bound)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    # Objective: maximize sum(r) <=> minimize -sum(r)
    c = -np.ones(n)
    bounds_lp = [(0, None)] * n
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    if res_lp.success:
        radii = res_lp.x
    else:
        # fallback to naive inscribed radii
        radii = np.empty(n)
        for i in range(n):
            xi, yi = points[i]
            r_min = min(xi, 1 - xi, yi, 1 - yi)
            dists = np.linalg.norm(points - points[i], axis=1)
            r_min = min(r_min, np.min(np.delete(dists, i)) / 2)
            radii[i] = r_min

    # Local hill-climbing with multi-candidate proposals for deeper search
    curr_pts = points.copy()
    curr_r = radii.copy()
    curr_sum = curr_r.sum()
    rng = np.random.default_rng(42)
    for _ in range(300):  # Further increased hill-climb iterations
        i = rng.integers(n)
        best_trial_sum = curr_sum
        best_trial_pts = curr_pts
        best_trial_r = curr_r
        # Try multiple jitter proposals and pick the best
        for _m in range(8):  # More candidate proposals per hill-climb
            cand_pts = curr_pts.copy()
            cand_pts[i] = np.clip(
                curr_pts[i] + rng.normal(0, 0.07, size=2),  # increased jitter
                0.0, 1.0
            )
            # rebuild LP constraints
            dmat_c = np.linalg.norm(cand_pts[:, None, :] - cand_pts[None, :, :], axis=2)
            A_c, b_c = [], []
            for ii in range(n):
                for jj in range(ii+1, n):
                    row = np.zeros(n); row[ii]=1; row[jj]=1
                    A_c.append(row); b_c.append(dmat_c[ii, jj])
            for ii in range(n):
                xi, yi = cand_pts[ii]
                for bound in (xi, 1-xi, yi, 1-yi):
                    row = np.zeros(n); row[ii]=1
                    A_c.append(row); b_c.append(bound)
            A_c = np.array(A_c); b_c = np.array(b_c)
            res_c = linprog(c, A_ub=A_c, b_ub=b_c, bounds=bounds_lp, method='highs')
            if not res_c.success:
                continue
            r_c = res_c.x; sum_c = r_c.sum()
            if sum_c > best_trial_sum:
                best_trial_sum, best_trial_pts, best_trial_r = sum_c, cand_pts, r_c
        if best_trial_sum > curr_sum:
            curr_sum, curr_pts, curr_r = best_trial_sum, best_trial_pts, best_trial_r

    points = curr_pts
    radii = curr_r

    circles = np.column_stack([points, radii])
    # Phase 3: physics-based repulsion+inflation refinement
    pos_phys, r_phys = _physics_refine(points.copy(), radii.copy(),
                                       iterations=500, lr=0.1, dr=1e-3)
    circles = np.column_stack([pos_phys, r_phys])
    # Phase 4: nonlinear local SLSQP refinement to fine-tune positions/radii
    circles = _slsqp_refine(circles)
    # Phase 5: final LP-based radius optimization for refined positions
    pos_final = circles[:, :2]
    r_initial = circles[:, 2]
    # Build LP constraints for final radii
    dmat_final = np.linalg.norm(pos_final[:, None, :] - pos_final[None, :, :], axis=2)
    A_final = []
    b_final = []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1; row[j] = 1
            A_final.append(row); b_final.append(dmat_final[i, j])
    for i in range(n):
        xi, yi = pos_final[i]
        for bound in (xi, 1 - xi, yi, 1 - yi):
            row = np.zeros(n)
            row[i] = 1
            A_final.append(row); b_final.append(bound)
    A_final = np.array(A_final); b_final = np.array(b_final)
    res_final = linprog(c, A_ub=A_final, b_ub=b_final, bounds=bounds_lp, method='highs')
    if res_final.success:
        circles = np.column_stack([pos_final, res_final.x])
    else:
        circles = np.column_stack([pos_final, r_initial])
    return circles


# EVOLVE-BLOCK-END
