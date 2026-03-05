# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# EXPLORATION: Multi-candidate initialization with extensive growth and optimization
# Synthesizes best practices from inspiration programs for maximum performance

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    
    Comprehensive Strategy (inspired by best-performing approaches):
    1. Multi-candidate initialization (8 patterns) with quick evaluation
    2. Vectorized force-directed relaxation on winning candidate
    3. Extended radius growth with adaptive rates (2000+ iterations)
    4. Position swap exploration with simulated annealing
    5. Multi-start SLSQP optimization (6+ trials)
    6. Final fine-tuning phase

    Returns:
        circles: np.array of shape (32,3), where row i is (x, y, r) for circle i
    """
    np.random.seed(42)
    n = 32
    
    # Phase 1: Advanced initialization with binary search for optimal radii (KEY INNOVATION from INSPIRATIONS)
    def initialize_hexagonal(n, spacing=0.148, offset_x=0.074, offset_y=0.074):
        """Hexagonal packing initialization (from INSPIRATION 1 & 2)"""
        positions = []
        row = 0
        while len(positions) < n:
            x_offset = (spacing / 2) if row % 2 == 1 else 0
            y = offset_y + row * spacing * np.sqrt(3) / 2
            
            col = 0
            while y < 0.93:
                x = offset_x + x_offset + col * spacing
                if x < 0.93 and len(positions) < n:
                    positions.append([x, y])
                col += 1
                if x >= 0.93:
                    break
            row += 1
            if y >= 0.93:
                break
        
        return np.array(positions[:n])
    
    def initialize_boundary_focused(n):
        """Boundary-focused initialization (from INSPIRATION 1 & 2)"""
        positions = []
        boundary_count = min(16, n // 2)
        
        for i in range(boundary_count):
            t = i / boundary_count
            if t < 0.25:
                x, y = t * 4, 0.05
            elif t < 0.5:
                x, y = 0.95, (t - 0.25) * 4
            elif t < 0.75:
                x, y = 1.0 - (t - 0.5) * 4, 0.95
            else:
                x, y = 0.05, 1.0 - (t - 0.75) * 4
            positions.append([x, y])
        
        interior_count = n - boundary_count
        grid_size = int(np.ceil(np.sqrt(interior_count)))
        for i in range(interior_count):
            row, col = i // grid_size, i % grid_size
            x = 0.2 + (col / (grid_size + 1)) * 0.6
            y = 0.2 + (row / (grid_size + 1)) * 0.6
            positions.append([x, y])
        
        return np.array(positions[:n])
    
    def compute_optimal_radii_binary(positions):
        """Binary search for optimal radii (CRITICAL INNOVATION from INSPIRATIONS)"""
        from scipy.spatial.distance import cdist
        n = len(positions)
        radii = np.zeros(n)
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        
        for i in range(n):
            x, y = positions[i]
            boundary_limit = min(x, y, 1 - x, 1 - y)
            neighbor_limit = np.min(dist_matrix[i]) / 2
            r_max = min(boundary_limit, neighbor_limit)
            r_min = 0.0
            
            # Binary search for maximum feasible radius
            for _ in range(24):
                r_mid = (r_min + r_max) / 2
                if r_mid <= boundary_limit and r_mid <= neighbor_limit:
                    r_min = r_mid
                else:
                    r_max = r_mid
            
            radii[i] = r_min * 0.9985  # Safety factor
        
        return radii
    
    # Generate diverse candidates with perturbations (from INSPIRATIONS)
    init_strategies = [
        ('hexagonal', 0.0, 42),
        ('hexagonal', 0.008, 43),
        ('boundary_focused', 0.0, 44),
        ('hexagonal', 0.015, 45)
    ]
    
    candidates = []
    
    for strategy, perturbation, seed in init_strategies:
        np.random.seed(seed)
        
        if strategy == 'hexagonal':
            positions = initialize_hexagonal(n)
        else:
            positions = initialize_boundary_focused(n)
        
        if perturbation > 0:
            positions += np.random.randn(n, 2) * perturbation
            positions = np.clip(positions, 0.015, 0.985)
        
        # Binary search for optimal radii (KEY IMPROVEMENT)
        radii = compute_optimal_radii_binary(positions)
        candidates.append(np.column_stack([positions, radii]))
    
    # Phase 2: Quick evaluation with vectorized force relaxation (from INSPIRATIONS)
    def quick_evaluate(circles_init):
        circles = circles_init.copy()
        
        # Efficient vectorized force relaxation
        for _ in range(100):
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            dists = squareform(pdist(positions))
            np.fill_diagonal(dists, np.inf)
            
            forces = np.zeros((n, 2))
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_matrix = dists[:, :, np.newaxis]
            min_dist_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :])[:, :, np.newaxis]
            
            mask = (dist_matrix < min_dist_matrix * 1.3) & (dist_matrix > 1e-6)
            direction = np.where(mask, diff / dist_matrix, 0)
            force_mag = np.where(mask[:,:,0], (min_dist_matrix[:,:,0] * 1.3 - dists) * 0.3, 0)
            forces = np.sum(direction * force_mag[:, :, np.newaxis], axis=1)
            
            circles[:, :2] += forces * 0.02
            circles[:, :2] = np.clip(circles[:, :2], radii[:, None], 1 - radii[:, None])
        
        return circles, circles[:, 2].sum()
    
    # Evaluate all candidates
    best_circles = None
    best_sum = 0
    for candidate in candidates:
        optimized, sum_radii = quick_evaluate(candidate)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = optimized
    
    circles = best_circles
    
    # Phase 3: Enhanced vectorized force relaxation
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
        
        # Boundary forces
        margin = radii[:, np.newaxis]
        boundary_forces = np.zeros((n, 2))
        boundary_forces[:, 0] += np.where(positions[:, 0] < margin[:, 0], 
                                         (margin[:, 0] - positions[:, 0]) * 0.4, 0)
        boundary_forces[:, 0] -= np.where(positions[:, 0] > 1 - margin[:, 0], 
                                         (positions[:, 0] - (1 - margin[:, 0])) * 0.4, 0)
        boundary_forces[:, 1] += np.where(positions[:, 1] < margin[:, 0], 
                                         (margin[:, 0] - positions[:, 1]) * 0.4, 0)
        boundary_forces[:, 1] -= np.where(positions[:, 1] > 1 - margin[:, 0], 
                                         (positions[:, 1] - (1 - margin[:, 0])) * 0.4, 0)
        
        forces += boundary_forces
        damping = 0.7 * (1 - iteration / 200 * 0.6)
        circles[:, :2] += forces * damping
        circles[:, :2] = np.clip(circles[:, :2], radii[:, None], 1 - radii[:, None])
    
    # Phase 4: Extended radius growth with adaptive rates and position swaps (from INSPIRATION 2)
    base_growth_rate = 0.0006
    for iteration in range(2000):
        grown = False
        growth_rate = base_growth_rate * (1.0 + 4.0 * np.exp(-iteration / 350))
        
        # Periodic position swaps with simulated annealing
        if iteration % 50 == 0 and iteration > 0:
            temperature = 0.0005 * np.exp(-iteration / 400)
            for i in range(min(12, n)):
                for j in range(i + 1, min(i + 12, n)):
                    old_sum = circles[:, 2].sum()
                    temp_pos = circles[i, :2].copy()
                    circles[i, :2] = circles[j, :2]
                    circles[j, :2] = temp_pos
                    
                    # Recalculate max radii for swapped circles
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
                    if delta < 0 or (delta < temperature and np.random.rand() < np.exp(-delta/max(temperature, 1e-8))):
                        pass  # Accept
                    else:
                        # Reject
                        circles[i, :2] = circles[j, :2]
                        circles[j, :2] = temp_pos
        
        # Grow circles in order of smallest first
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
    
    # Phase 5: Multi-start SLSQP optimization (6 trials from INSPIRATION 2)
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
    
    # Increased from 6 to 10 trials with improved perturbation strategy
    for trial in range(10):
        try:
            if trial == 0:
                x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
            else:
                perturb = circles.copy()
                # Better perturbation scaling (from INSPIRATION 1)
                if trial <= 4:
                    perturbation_scale = 0.005 * (1.0 + 0.5 / trial)
                else:
                    perturbation_scale = 0.003 * (1.0 + 0.3 / (trial - 4))
                perturb[:, :2] += np.random.randn(n, 2) * perturbation_scale
                perturb[:, 2] *= (1.0 + np.random.randn(n) * 0.0008)
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
                            constraints=constraints, options={'maxiter': 400, 'ftol': 1e-10})
            
            if result.success and result.fun < -best_sum:
                best_sum = -result.fun
                best_circles[:, 0] = result.x[:n]
                best_circles[:, 1] = result.x[n:2*n]
                best_circles[:, 2] = result.x[2*n:]
        except:
            pass
    
    circles = best_circles
    
    # Phase 6: Directional micro-refinement (from INSPIRATION 1 & 2)
    for refinement_pass in range(2):
        improved_any = False
        step_size = 0.004 if refinement_pass == 0 else 0.002
        for i in range(n):
            best_local_sum = circles[:, 2].sum()
            best_pos = circles[i, :2].copy()
            # Test 8 directions
            for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                          (step_size*0.7, step_size*0.7), (step_size*0.7, -step_size*0.7),
                          (-step_size*0.7, step_size*0.7), (-step_size*0.7, -step_size*0.7)]:
                test_x = circles[i, 0] + dx
                test_y = circles[i, 1] + dy
                if test_x < 0.01 or test_x > 0.99 or test_y < 0.01 or test_y > 0.99:
                    continue
                max_r = min(test_x, 1 - test_x, test_y, 1 - test_y)
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[j, 0] - test_x)**2 + (circles[j, 1] - test_y)**2)
                        max_r = min(max_r, dist - circles[j, 2] - 1e-6)
                if max_r > 1e-6:
                    test_circles = circles.copy()
                    test_circles[i, 0] = test_x
                    test_circles[i, 1] = test_y
                    test_circles[i, 2] = max_r - 1e-6
                    test_sum = test_circles[:, 2].sum()
                    if test_sum > best_local_sum:
                        best_local_sum = test_sum
                        best_pos = np.array([test_x, test_y])
                        circles[i, :2] = best_pos
                        circles[i, 2] = max_r - 1e-6
                        improved_any = True
                        break
        if not improved_any:
            break
    
    # Phase 7: Extended fine-tuning (1500 iterations from INSPIRATION 1 & 2)
    for iteration in range(1500):
        grown = False
        if iteration < 400:
            fine_growth_rate = 0.0004 * (1.0 + 2.0 * np.exp(-iteration / 150))
        else:
            fine_growth_rate = 0.0003 * (1.0 + 1.5 * np.exp(-iteration / 250))
        
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
    
    # Phase 8: Additional SLSQP trials with tighter tolerances (from INSPIRATIONS)
    best_circles = circles.copy()
    best_sum = circles[:, 2].sum()
    
    for trial in range(3):
        try:
            if trial == 0:
                x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
            else:
                perturb = circles.copy()
                perturbation_scale = 0.003 * (1.0 + 0.4 / trial)
                perturb[:, :2] += np.random.randn(n, 2) * perturbation_scale
                perturb[:, 2] *= (1.0 + np.random.randn(n) * 0.0005)
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
                            constraints=constraints, options={'maxiter': 400, 'ftol': 1e-10})
            
            if result.success and result.fun < -best_sum:
                best_sum = -result.fun
                best_circles[:, 0] = result.x[:n]
                best_circles[:, 1] = result.x[n:2*n]
                best_circles[:, 2] = result.x[2*n:]
        except:
            pass
    
    circles = best_circles
    
    # Phase 9: Final SLSQP polish (from INSPIRATIONS)
    try:
        x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
        bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)] * n
        constraints = [
            {'type': 'ineq', 'fun': constraint_overlap},
            {'type': 'ineq', 'fun': constraint_boundary}
        ]
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                        constraints=constraints, options={'maxiter': 250, 'ftol': 1e-10})
        if result.success and -result.fun > circles[:, 2].sum():
            circles[:, 0] = result.x[:n]
            circles[:, 1] = result.x[n:2*n]
            circles[:, 2] = result.x[2*n:]
    except:
        pass
    
    # Phase 10: Ultra-fine final growth (from INSPIRATIONS)
    for iteration in range(250):
        grown = False
        ultra_fine_rate = 0.0001 * (1.0 + 0.5 * np.exp(-iteration / 80))
        
        order = np.argsort(circles[:, 2])
        for idx in order:
            i = int(idx)
            max_r = min(circles[i, 0], 1 - circles[i, 0], circles[i, 1], 1 - circles[i, 1])
            
            for j in range(n):
                if i != j:
                    dist = np.sqrt((circles[j, 0] - circles[i, 0])**2 + (circles[j, 1] - circles[i, 1])**2)
                    max_r = min(max_r, dist - circles[j, 2] - 1e-7)
            
            if max_r > circles[i, 2] + ultra_fine_rate:
                circles[i, 2] += ultra_fine_rate
                grown = True
            elif max_r > circles[i, 2] + 1e-10:
                circles[i, 2] = max_r - 1e-10
                grown = True
        
        if not grown:
            break
    
    return circles


# EVOLVE-BLOCK-END
