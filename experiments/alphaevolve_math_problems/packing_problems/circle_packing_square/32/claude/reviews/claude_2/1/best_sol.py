# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# EXPLORATION: Synthesized multi-phase optimization combining best practices
# Integrates multi-candidate initialization, extended growth, position swaps, and aggressive refinement

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    
    Comprehensive Strategy (synthesized from best approaches):
    1. Multi-candidate initialization (8 patterns) with quick evaluation
    2. Vectorized force-directed relaxation on winning candidate
    3. Extended radius growth with adaptive rates (2200+ iterations)
    4. Position swap exploration with simulated annealing
    5. Multi-start SLSQP optimization (10 trials with diverse perturbations)
    6. Directional micro-refinement
    7. Extended fine-tuning phase (1500+ iterations)

    Returns:
        circles: np.array of shape (32,3), where row i is (x, y, r) for circle i
    """
    np.random.seed(42)
    n = 32
    
    # Phase 1: Multi-candidate initialization (from INSPIRATION 1 & 2)
    def create_grid_init(rows, cols, n_circles):
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        initial_radius = min(spacing_x, spacing_y) * 0.35
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) < n_circles:
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    positions.append([x, y, initial_radius])
        return np.array(positions[:n_circles])
    
    def create_hex_init(n_circles):
        positions = []
        rows, cols = 6, 6
        spacing_x = 1.0 / (cols + 0.5)
        spacing_y = 1.0 / (rows + 1)
        initial_radius = min(spacing_x, spacing_y) * 0.35
        for i in range(rows):
            offset = 0.5 if i % 2 == 1 else 0.0
            for j in range(cols):
                if len(positions) < n_circles:
                    x = spacing_x * (j + offset + 1)
                    y = spacing_y * (i + 1)
                    x = np.clip(x, 0.1, 0.9)
                    positions.append([x, y, initial_radius])
        return np.array(positions[:n_circles])
    
    # Generate 8 candidate initializations
    candidates = [
        create_grid_init(6, 6, n),
        create_grid_init(4, 8, n),
        create_grid_init(8, 4, n),
        create_grid_init(5, 7, n),
        create_grid_init(7, 5, n),
        create_hex_init(n),
        create_hex_init(n),
        create_grid_init(6, 7, n)
    ]
    
    # Phase 2: Quick evaluation to select best candidate
    def quick_evaluate(circles_init):
        circles = circles_init.copy()
        for _ in range(80):
            forces = np.zeros((n, 2))
            for i in range(n):
                for j in range(i + 1, n):
                    dx = circles[j, 0] - circles[i, 0]
                    dy = circles[j, 1] - circles[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist * 1.3 and dist > 1e-6:
                        overlap = min_dist - dist
                        force_mag = overlap * 0.5
                        fx, fy = force_mag * dx / dist, force_mag * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
            circles[:, :2] += forces * 0.01
            for i in range(n):
                r = circles[i, 2]
                circles[i, 0] = np.clip(circles[i, 0], r + 0.001, 1 - r - 0.001)
                circles[i, 1] = np.clip(circles[i, 1], r + 0.001, 1 - r - 0.001)
        for _ in range(200):
            grown = False
            for i in range(n):
                max_r = min(circles[i, 0], 1 - circles[i, 0], circles[i, 1], 1 - circles[i, 1])
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[j, 0] - circles[i, 0])**2 + (circles[j, 1] - circles[i, 1])**2)
                        max_r = min(max_r, dist - circles[j, 2] - 1e-6)
                if max_r > circles[i, 2] + 0.001:
                    circles[i, 2] += 0.001
                    grown = True
            if not grown:
                break
        return circles, circles[:, 2].sum()
    
    best_circles = None
    best_sum = 0
    for candidate in candidates:
        optimized, sum_radii = quick_evaluate(candidate)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = optimized
    circles = best_circles
    
    # Phase 3: Vectorized force relaxation (from INSPIRATION 1)
    for iteration in range(200):
        positions = circles[:, :2]
        radii = circles[:, 2]
        dists = squareform(pdist(positions))
        np.fill_diagonal(dists, np.inf)
        forces = np.zeros((n, 2))
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = dists[:, :, np.newaxis]
        min_dist_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :])[:, :, np.newaxis]
        force_threshold = min_dist_matrix * 1.4
        mask = (dist_matrix < force_threshold) & (dist_matrix > 1e-6)
        direction = np.where(mask, diff / dist_matrix, 0)
        force_mag = np.where(mask[:,:,0], (force_threshold[:,:,0] - dists) * 0.2, 0)
        forces = np.sum(direction * force_mag[:, :, np.newaxis], axis=1)
        margin = radii[:, np.newaxis]
        boundary_forces = np.zeros((n, 2))
        boundary_forces[:, 0] += np.where(positions[:, 0] < margin[:, 0], (margin[:, 0] - positions[:, 0]) * 0.4, 0)
        boundary_forces[:, 0] -= np.where(positions[:, 0] > 1 - margin[:, 0], (positions[:, 0] - (1 - margin[:, 0])) * 0.4, 0)
        boundary_forces[:, 1] += np.where(positions[:, 1] < margin[:, 0], (margin[:, 0] - positions[:, 1]) * 0.4, 0)
        boundary_forces[:, 1] -= np.where(positions[:, 1] > 1 - margin[:, 0], (positions[:, 1] - (1 - margin[:, 0])) * 0.4, 0)
        forces += boundary_forces
        damping = 0.7 * (1 - iteration / 200 * 0.6)
        circles[:, :2] += forces * damping
        circles[:, :2] = np.clip(circles[:, :2], radii[:, None], 1 - radii[:, None])
    
    # Phase 4: Ultra-high precision binary search for optimal radii (from INSPIRATION 1)
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(circles[:, :2], circles[:, :2])
    np.fill_diagonal(dist_matrix, np.inf)
    
    for i in range(n):
        x, y = circles[i, 0], circles[i, 1]
        boundary_limit = min(x, y, 1 - x, 1 - y)
        neighbor_limit = np.min(dist_matrix[i]) / 2
        r_max = min(boundary_limit, neighbor_limit)
        r_min = 0.0
        
        # Ultra-high precision: 30 iterations (from INSPIRATION 1)
        for _ in range(30):
            r_mid = (r_min + r_max) / 2
            if r_mid <= boundary_limit and r_mid <= neighbor_limit:
                r_min = r_mid
            else:
                r_max = r_mid
        
        # Less conservative (0.999 from INSPIRATION 1)
        circles[i, 2] = r_min * 0.999
    
    # Phase 4b: Optimized radius growth with FIXED position swaps (reduced iterations)
    base_growth_rate = 0.0006
    for iteration in range(1500):
        grown = False
        growth_rate = base_growth_rate * (1.0 + 4.0 * np.exp(-iteration / 350))
        
        # Position swaps with simulated annealing - FIXED acceptance logic
        if iteration % 60 == 0 and iteration > 0:
            temperature = 0.0005 * np.exp(-iteration / 400)
            for i in range(min(10, n)):
                for j in range(i + 1, min(i + 10, n)):
                    old_sum = circles[:, 2].sum()
                    temp_pos = circles[i, :2].copy()
                    circles[i, :2] = circles[j, :2]
                    circles[j, :2] = temp_pos
                    for k in [i, j]:
                        max_r = min(circles[k, 0], 1 - circles[k, 0], circles[k, 1], 1 - circles[k, 1])
                        for m in range(n):
                            if k != m:
                                dist = np.sqrt((circles[m, 0] - circles[k, 0])**2 + (circles[m, 1] - circles[k, 1])**2)
                                max_r = min(max_r, dist - circles[m, 2] - 1e-6)
                        if max_r > 1e-6:
                            circles[k, 2] = max(circles[k, 2], max_r - 1e-6)
                    new_sum = circles[:, 2].sum()
                    delta = new_sum - old_sum
                    # FIXED: Accept if improved or probabilistically (from INSPIRATION 2)
                    if delta >= 0 or np.random.rand() < np.exp(delta / max(temperature, 1e-10)):
                        pass  # Keep swap
                    else:
                        # Revert swap
                        circles[i, :2] = circles[j, :2]
                        circles[j, :2] = temp_pos
        
        order = np.argsort(circles[:, 2])
        for idx in order:
            i = int(idx)
            max_r = min(circles[i, 0], 1 - circles[i, 0], circles[i, 1], 1 - circles[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.sqrt((circles[j, 0] - circles[i, 0])**2 + (circles[j, 1] - circles[i, 1])**2)
                    max_r = min(max_r, dist - circles[j, 2] - 1e-6)
            if max_r > circles[i, 2] + growth_rate:
                circles[i, 2] += growth_rate
                grown = True
            elif max_r > circles[i, 2] + 1e-6:
                circles[i, 2] = max_r - 1e-6
                grown = True
        if not grown:
            break
    
    # Phase 5: Multi-start SLSQP optimization (10 trials)
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    def constraint_overlap(params):
        pos_x, pos_y, radii = params[:n], params[n:2*n], params[2*n:]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((pos_x[i] - pos_x[j])**2 + (pos_y[i] - pos_y[j])**2)
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def constraint_boundary(params):
        pos_x, pos_y, radii = params[:n], params[n:2*n], params[2*n:]
        constraints = []
        for i in range(n):
            constraints.extend([pos_x[i] - radii[i], 1 - pos_x[i] - radii[i],
                              pos_y[i] - radii[i], 1 - pos_y[i] - radii[i]])
        return np.array(constraints)
    
    best_circles = circles.copy()
    best_sum = circles[:, 2].sum()
    
    for trial in range(7):
        try:
            if trial == 0:
                x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
            else:
                perturb = circles.copy()
                perturbation_scale = 0.005 * (1.0 + 0.5 / trial)
                perturb[:, :2] += np.random.randn(n, 2) * perturbation_scale
                perturb[:, 2] *= (1.0 + np.random.randn(n) * 0.001)
                for i in range(n):
                    r = perturb[i, 2]
                    perturb[i, 0] = np.clip(perturb[i, 0], r, 1 - r)
                    perturb[i, 1] = np.clip(perturb[i, 1], r, 1 - r)
                    perturb[i, 2] = np.clip(perturb[i, 2], 0.001, 0.5)
                x0 = np.concatenate([perturb[:, 0], perturb[:, 1], perturb[:, 2]])
            
            bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)] * n
            constraints = [
                {'type': 'ineq', 'fun': constraint_overlap},
                {'type': 'ineq', 'fun': constraint_boundary}
            ]
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints, options={'maxiter': 300, 'ftol': 1e-9})
            if result.success and result.fun < -best_sum:
                best_sum = -result.fun
                best_circles[:, 0] = result.x[:n]
                best_circles[:, 1] = result.x[n:2*n]
                best_circles[:, 2] = result.x[2*n:]
        except:
            pass
    circles = best_circles
    
    # Phase 6: CRITICAL - Coordinate descent refinement (breakthrough from both inspirations!)
    for iteration in range(8):
        improved = False
        for i in range(n):
            x, y, r = circles[i]
            
            def objective_i(params):
                return -params[2]
            
            def constraints_i(params):
                xi, yi, ri = params
                violations = [
                    ri - xi + 3e-8,
                    ri - yi + 3e-8,
                    ri - (1 - xi) + 3e-8,
                    ri - (1 - yi) + 3e-8
                ]
                
                for j in range(n):
                    if i != j:
                        xj, yj, rj = circles[j]
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        violations.append(ri + rj - dist + 3e-8)
                
                return np.array(violations)
            
            result = minimize(objective_i, [x, y, r], method='SLSQP',
                            bounds=[(0.005, 0.995), (0.005, 0.995), (0.00001, 0.5)],
                            constraints={'type': 'ineq', 'fun': lambda p: -constraints_i(p)},
                            options={'maxiter': 50, 'ftol': 1e-11})
            
            if result.success and -result.fun > r * 0.9999:
                circles[i] = result.x
                improved = True
        
        if not improved:
            break
    
    # Phase 6b: Streamlined fine-tuning (reduced from 1500 to 800 iterations)
    for iteration in range(800):
        grown = False
        fine_growth_rate = 0.0003 * (1.0 + 1.5 * np.exp(-iteration / 200))
        order = np.argsort(circles[:, 2])
        for idx in order:
            i = int(idx)
            max_r = min(circles[i, 0], 1 - circles[i, 0], circles[i, 1], 1 - circles[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.sqrt((circles[j, 0] - circles[i, 0])**2 + (circles[j, 1] - circles[i, 1])**2)
                    max_r = min(max_r, dist - circles[j, 2] - 1e-6)
            if max_r > circles[i, 2] + fine_growth_rate:
                circles[i, 2] += fine_growth_rate
                grown = True
            elif max_r > circles[i, 2] + 1e-9:
                circles[i, 2] = max_r - 1e-9
                grown = True
        if not grown:
            break
    
    # Phase 7: Progressive tightening SLSQP refinement (from INSPIRATION 1)
    try:
        x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
        bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)] * n
        constraints = [
            {'type': 'ineq', 'fun': constraint_overlap},
            {'type': 'ineq', 'fun': constraint_boundary}
        ]
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                        constraints=constraints, options={'maxiter': 150, 'ftol': 1e-9})
        if result.success and -result.fun > circles[:, 2].sum() * 0.9999:
            circles[:, 0] = result.x[:n]
            circles[:, 1] = result.x[n:2*n]
            circles[:, 2] = result.x[2*n:]
    except:
        pass
    
    # Phase 8: Ultra-fine squeeze (from INSPIRATION 1 - extended to 80 iterations)
    for squeeze_iter in range(80):
        improved = False
        for i in range(n):
            for growth in [0.000002, 0.000001, 0.0000005, 0.0000002, 0.0000001]:
                test_r = circles[i, 2] + growth
                x, y = circles[i, 0], circles[i, 1]
                
                if test_r > min(x, y, 1-x, 1-y) - 1e-9:
                    continue
                
                valid = True
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(circles[i, :2] - circles[j, :2])
                        if dist < test_r + circles[j, 2] - 1e-9:
                            valid = False
                            break
                
                if valid:
                    circles[i, 2] = test_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
