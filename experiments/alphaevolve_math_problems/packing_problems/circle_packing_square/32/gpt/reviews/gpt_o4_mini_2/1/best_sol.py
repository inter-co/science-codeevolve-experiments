# EVOLVE-BLOCK-START
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import numpy as np
from scipy.optimize import linprog  # Added for LP-based radius optimization
from scipy.optimize import minimize  # Added for nonlinear SLSQP refinement
import math

# Helper: physics-inspired repulsion + inflation refinement
def _physics_refine(pos, r, iterations=200, lr=0.01, dr=1e-4):
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
def _slsqp_refine(circles, maxiter=300):  # More SLSQP iterations by default
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
                   options={'maxiter': maxiter, 'ftol': 1e-7})  # Tighter tolerance
    if res.success:
        return res.x.reshape(-1,3)
    else:
        return circles

# Greedy radius expansion to fill any remaining gaps (imported from inspiration)
def _greedy_expand(circles):
    import math
    n = circles.shape[0]
    for i in range(n):
        x, y, _ = circles[i]
        # max radius limited by boundaries
        max_r = min(x, y, 1 - x, 1 - y)
        # further limited by neighbor distances
        for j in range(n):
            if j == i:
                continue
            dx = x - circles[j, 0]
            dy = y - circles[j, 1]
            d = math.hypot(dx, dy) - circles[j, 2]
            if d < max_r:
                max_r = d
        circles[i, 2] = max(0.0, max_r)
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
    points = _hexagonal_seed(n)
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
    for _ in range(200):  # Increased CVT iterations
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

    # Local hill-climbing on positions to improve radii sum via LP
    curr_pts = points.copy()
    curr_r = radii.copy()
    curr_sum = curr_r.sum()
    rng = np.random.default_rng(42)
    for _ in range(100):  # Increased hill-climb iterations
        i = rng.integers(n)
        new_pts = curr_pts.copy()
        jitter = rng.normal(0, 0.03, size=2)  # Increased jitter for exploration
        new_pts[i] = np.clip(curr_pts[i] + jitter, 0.0, 1.0)
        # recompute LP for new positions
        dmat = np.linalg.norm(new_pts[:, None, :] - new_pts[None, :, :], axis=2)
        A = []
        b = []
        # Non-overlap constraints
        for ii in range(n):
            for jj in range(ii+1, n):
                row = np.zeros(n)
                row[ii] = 1
                row[jj] = 1
                A.append(row)
                b.append(dmat[ii, jj])
        # Boundary constraints
        for ii in range(n):
            xi, yi = new_pts[ii]
            for bound in (xi, 1 - xi, yi, 1 - yi):
                row = np.zeros(n)
                row[ii] = 1
                A.append(row)
                b.append(bound)
        A = np.array(A)
        b = np.array(b)
        res2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds_lp, method='highs')
        if not res2.success:
            continue
        r2 = res2.x
        sum2 = r2.sum()
        if sum2 > curr_sum:
            curr_pts = new_pts
            curr_r = r2
            curr_sum = sum2

    points = curr_pts
    radii = curr_r

    circles = np.column_stack([points, radii])
    # Phase 3: physics-based repulsion+inflation refinement
    pos_phys, r_phys = _physics_refine(points.copy(), radii.copy(),
                                       iterations=300, lr=0.05, dr=5e-4)
    circles = np.column_stack([pos_phys, r_phys])
    # Phase 4: nonlinear local SLSQP refinement to fine-tune positions/radii
    circles = _slsqp_refine(circles)
    # Final greedy expansion to fill any remaining gaps and push radii upward
    circles = _greedy_expand(circles)
    return circles


# EVOLVE-BLOCK-END
