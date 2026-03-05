# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform

# EXPLORATION: Enhanced multi-phase optimization with adaptive strategies
# Combines improved initialization, vectorized force dynamics, and multi-start optimization

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    
    Enhanced Strategy:
    1. Smart initialization with spatial awareness
    2. Vectorized force-directed relaxation with adaptive damping
    3. Greedy radius expansion with priority queue
    4. Multi-start gradient-based optimization
    5. Differential evolution for global refinement

    Returns:
        circles: np.array of shape (32,3), where row i is (x, y, r) for circle i
    """
    np.random.seed(42)
    n = 32
    
    # Helper functions for diverse initializations (from INSPIRATION 1)
    def initialize_hexagonal(n, spacing=0.148, offset_x=0.074, offset_y=0.074):
        """Hexagonal packing initialization."""
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
        """Boundary-focused initialization (from INSPIRATION 1)."""
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
        """Binary search for optimal radii (from INSPIRATION 1)."""
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
            
            # Increased precision from 24 to 30 iterations (from INSPIRATION 1)
            for _ in range(30):
                r_mid = (r_min + r_max) / 2
                if r_mid <= boundary_limit and r_mid <= neighbor_limit:
                    r_min = r_mid
                else:
                    r_max = r_mid
            
            radii[i] = r_min * 0.999  # Slightly more conservative (from INSPIRATION 1)
        
        return radii
    
    # Phase 1: Multi-start with diverse initializations (from INSPIRATION 1)
    init_strategies = [
        ('hexagonal', 0.0, 42),
        ('hexagonal', 0.008, 43),
        ('boundary_focused', 0.0, 44),
        ('hexagonal', 0.015, 45)
    ]
    
    candidates = []
    circles = None
    
    # Phase 2: Force relaxation function (reusable across starts)
    def apply_forces(circles, n_iter=200):
        velocities = np.zeros((n, 2))
        convergence_count = 0
        
        for iteration in range(n_iter):
            progress = iteration / n_iter
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Vectorized distance calculation
            dists = squareform(pdist(positions))
            np.fill_diagonal(dists, np.inf)
            
            # Track maximum violation for adaptive behavior
            max_violation = 0.0
            
            # Vectorized force calculation
            forces = np.zeros((n, 2))
            
            # Adaptive force strength based on progress (from INSPIRATION 2)
            if progress < 0.3:
                base_force = 0.18
            elif progress < 0.7:
                base_force = 0.15 * (1.0 - progress * 0.5)
            else:
                base_force = 0.10 * (1.0 - progress)
            
            # Repulsive forces (vectorized where possible)
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_matrix = dists[:, :, np.newaxis]
            min_dist_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :])[:, :, np.newaxis]
            
            # Calculate violations for adaptive force magnitude
            violations = np.maximum(0, min_dist_matrix[:,:,0] - dists)
            max_violation = max(max_violation, np.max(violations))
            
            # Enhanced force with violation-dependent magnitude (from INSPIRATION 2)
            force_threshold = min_dist_matrix * 1.3
            mask = (dist_matrix < force_threshold) & (dist_matrix > 1e-6)
            
            direction = np.where(mask, diff / dist_matrix, 0)
            # Violation-dependent force: stronger for larger overlaps
            force_mag = np.where(mask[:,:,0], 
                                (force_threshold[:,:,0] - dists) * base_force * (1.0 + violations * 5.0), 
                                0)
            
            forces = np.sum(direction * force_mag[:, :, np.newaxis], axis=1)
            
            # Boundary forces (vectorized)
            margin = radii[:, np.newaxis]
            boundary_forces = np.zeros((n, 2))
            
            # Enhanced boundary forces with violation tracking
            left_violation = np.maximum(0, margin[:, 0] - positions[:, 0])
            right_violation = np.maximum(0, positions[:, 0] + margin[:, 0] - 1)
            top_violation = np.maximum(0, margin[:, 0] - positions[:, 1])
            bottom_violation = np.maximum(0, positions[:, 1] + margin[:, 0] - 1)
            
            max_violation = max(max_violation, np.max(left_violation), np.max(right_violation),
                              np.max(top_violation), np.max(bottom_violation))
            
            # Left/right boundaries
            boundary_forces[:, 0] += np.where(positions[:, 0] < margin[:, 0], 
                                             left_violation * 0.35, 0)
            boundary_forces[:, 0] -= np.where(positions[:, 0] > 1 - margin[:, 0], 
                                             right_violation * 0.35, 0)
            
            # Top/bottom boundaries
            boundary_forces[:, 1] += np.where(positions[:, 1] < margin[:, 0], 
                                             top_violation * 0.35, 0)
            boundary_forces[:, 1] -= np.where(positions[:, 1] > 1 - margin[:, 0], 
                                             bottom_violation * 0.35, 0)
            
            forces += boundary_forces
            
            # Adaptive damping based on violation magnitude (from INSPIRATION 2)
            if max_violation > 0.02:
                damping = 0.55
            elif max_violation > 0.01:
                damping = 0.70
            else:
                damping = 0.82 * (1 - progress * 0.3)
            
            velocities = velocities * damping + forces
            circles[:, :2] += velocities
            
            # Ensure within bounds
            circles[:, :2] = np.clip(circles[:, :2], radii[:, None], 1 - radii[:, None])
            
            # Convergence tracking (from INSPIRATION 2)
            if max_violation < 1e-6 and np.max(np.abs(velocities)) < 1e-6:
                convergence_count += 1
                if convergence_count >= 3:
                    break
            else:
                convergence_count = 0
        
        return circles
    
    # Phase 3: Multi-scale greedy expansion with ultra-fine levels (from INSPIRATION 2)
    def expand_radii_multiscale(circles, n_iter=60, use_ultrafine=False):
        no_improvement_count = 0
        for attempt in range(n_iter):
            improved = False
            
            # Order by smallest radius first (from INSPIRATION 2)
            order = np.argsort(circles[:, 2])
            
            for idx in order:
                i = int(idx)
                original_r = circles[i, 2]
                
                # 8-level growth for ultra-fine mode (from INSPIRATION 2), else standard 5-level
                if use_ultrafine:
                    growth_levels = [0.001, 0.0005, 0.0003, 0.00015, 0.00008, 0.00004, 0.00002, 0.00001]
                    tolerance = 6e-8  # Ultra-tight from INSPIRATION 2
                else:
                    growth_levels = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
                    tolerance = 1e-7
                
                for growth in growth_levels:
                    test_r = original_r + growth
                    
                    # Boundary check with adaptive tolerance
                    x, y = circles[i, 0], circles[i, 1]
                    if test_r > min(x, y, 1-x, 1-y) - tolerance:
                        continue
                    
                    # Overlap check with adaptive tolerance
                    valid = True
                    for j in range(n):
                        if i != j:
                            dist = np.linalg.norm(circles[i, :2] - circles[j, :2])
                            if dist < test_r + circles[j, 2] - tolerance:
                                valid = False
                                break
                    
                    if valid:
                        circles[i, 2] = test_r
                        improved = True
                        break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    break
            else:
                no_improvement_count = 0
        
        return circles
    
    # Execute multi-start optimization
    for strategy, perturbation, seed in init_strategies:
        np.random.seed(seed)
        
        if strategy == 'hexagonal':
            positions = initialize_hexagonal(n)
        else:
            positions = initialize_boundary_focused(n)
        
        if perturbation > 0:
            positions += np.random.randn(n, 2) * perturbation
            positions = np.clip(positions, 0.015, 0.985)
        
        # Binary search for optimal radii (from INSPIRATION 1)
        radii = compute_optimal_radii_binary(positions)
        trial_circles = np.column_stack([positions, radii])
        
        # Force relaxation
        trial_circles = apply_forces(trial_circles, n_iter=200)
        
        # Multi-scale greedy expansion
        trial_circles = expand_radii_multiscale(trial_circles, n_iter=60)
        
        current_sum = np.sum(trial_circles[:, 2])
        candidates.append((current_sum, trial_circles.copy()))
        
        if circles is None or current_sum > np.sum(circles[:, 2]):
            circles = trial_circles.copy()
        
        # Aggressive early termination if we beat benchmark by 0.1% (from INSPIRATION 1)
        if current_sum > 2.937944526205518 * 1.001:
            break
    
    # Progressive refinement on top 2 candidates (from INSPIRATION 1)
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Phase 4: Progressive SLSQP refinement on top candidates
    def refine_with_slsqp(circles, iterations=100, margin=1e-7):
        """SLSQP refinement with configurable margin (from INSPIRATION 1)."""
        n = len(circles)
        
        def objective(params):
            radii = params[2*n:]
            return -np.sum(radii)
        
        def constraints_func(params):
            positions = params[:2*n].reshape(n, 2)
            radii = params[2*n:]
            violations = []
            
            for i in range(n):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]
                violations.extend([r - x + margin, r - y + margin, 
                                 r - (1 - x) + margin, r - (1 - y) + margin])
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    violations.append(radii[i] + radii[j] - dist + margin)
            
            return np.array(violations)
        
        x0 = np.concatenate([circles[:, :2].flatten(), circles[:, 2]])
        constraints = {'type': 'ineq', 'fun': lambda p: -constraints_func(p)}
        bounds = [(0.005, 0.995)] * (2*n) + [(0.00001, 0.5)] * n
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, 
                             options={'maxiter': iterations, 'ftol': 1e-10, 'eps': 1e-9})
            
            if result.success or result.fun <= objective(x0) * 1.0004:
                positions = result.x[:2*n].reshape(n, 2)
                radii = result.x[2*n:]
                return np.column_stack([positions, radii])
        except:
            pass
        
        return circles
    
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    for rank, (init_sum, trial_circles) in enumerate(candidates[:2]):
        # Progressive SLSQP with tightening constraints (from INSPIRATION 1)
        trial_circles = refine_with_slsqp(trial_circles, iterations=120, margin=1e-7)
        trial_circles = refine_with_slsqp(trial_circles, iterations=150, margin=5e-8)
        
        # Standard multi-scale expansion for all candidates
        trial_circles = expand_radii_multiscale(trial_circles, n_iter=40, use_ultrafine=False)
        
        # Enhanced refinement for top candidate (optimized)
        if rank == 0:
            trial_circles = refine_with_slsqp(trial_circles, iterations=200, margin=3e-8)
            
            # Optimized ultra-fine growth: 70+50=120 iterations (better balance)
            trial_circles = expand_radii_multiscale(trial_circles, n_iter=70, use_ultrafine=True)
            
            # Final ultra-tight SLSQP with 2e-8 margin
            trial_circles = refine_with_slsqp(trial_circles, iterations=240, margin=2e-8)
            
            # Additional ultra-fine pass with longer duration
            trial_circles = expand_radii_multiscale(trial_circles, n_iter=50, use_ultrafine=True)
        
        current_sum = np.sum(trial_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = trial_circles.copy()
        
        # Smart early stopping if rank-0 beats benchmark by 0.15%
        if rank == 0 and best_sum > 2.937944526205518 * 1.0015:
            break
    
    circles = best_circles
    
    # Additional local optimization with standard approach
    def objective(x):
        pos_radii = x.reshape(n, 3)
        return -np.sum(pos_radii[:, 2])
    
    def is_feasible(x):
        pos_radii = x.reshape(n, 3)
        
        # Check containment
        for i in range(n):
            if pos_radii[i, 2] > min(pos_radii[i, 0], 1 - pos_radii[i, 0],
                                     pos_radii[i, 1], 1 - pos_radii[i, 1]):
                return False
        
        # Check non-overlap
        positions = pos_radii[:, :2]
        radii = pos_radii[:, 2]
        dists = squareform(pdist(positions))
        np.fill_diagonal(dists, np.inf)
        min_dists = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        if np.any(dists < min_dists - 1e-6):
            return False
        
        return True
    
    def constraint_all(x):
        pos_radii = x.reshape(n, 3)
        constraints = []
        
        # Containment
        for i in range(n):
            constraints.append(pos_radii[i, 0] - pos_radii[i, 2])
            constraints.append(1 - pos_radii[i, 0] - pos_radii[i, 2])
            constraints.append(pos_radii[i, 1] - pos_radii[i, 2])
            constraints.append(1 - pos_radii[i, 1] - pos_radii[i, 2])
            constraints.append(pos_radii[i, 2] - 0.001)
        
        # Non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(pos_radii[i, :2] - pos_radii[j, :2])
                min_dist = pos_radii[i, 2] + pos_radii[j, 2]
                constraints.append(dist - min_dist)
        
        return np.array(constraints)
    
    # Try 2 additional local optimizations with small perturbations
    for attempt in range(2):
        try:
            if attempt > 0:
                # Small random perturbation with re-initialization
                perturbed = circles.copy()
                perturbed[:, :2] += np.random.randn(n, 2) * 0.008
                perturbed[:, 2] *= np.random.uniform(0.985, 1.0, n)
                
                # Re-apply forces very briefly
                perturbed = apply_forces(perturbed, n_iter=25)
                start_point = perturbed
            else:
                start_point = circles
            
            constraints = [{'type': 'ineq', 'fun': constraint_all}]
            
            result = minimize(
                objective,
                start_point.flatten(),
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 250, 'ftol': 1e-8}
            )
            
            if result.success or is_feasible(result.x):
                candidate = result.x.reshape(n, 3)
                candidate_sum = np.sum(candidate[:, 2])
                if candidate_sum > np.sum(circles[:, 2]):
                    circles = candidate
        except:
            pass
    
    # Directional micro-refinement (from INSPIRATION 1, phase 6)
    for refinement_pass in range(2):
        improved_any = False
        step_size = 0.003 if refinement_pass == 0 else 0.0015
        
        for i in range(n):
            best_local_sum = circles[:, 2].sum()
            best_pos = circles[i, :2].copy()
            best_radius = circles[i, 2]
            
            # 8-directional search (from INSPIRATION 1)
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
                        dist = np.linalg.norm(circles[j, :2] - np.array([test_x, test_y]))
                        max_r = min(max_r, dist - circles[j, 2] - 5e-8)
                
                if max_r > 1e-6:
                    test_circles = circles.copy()
                    test_circles[i, 0] = test_x
                    test_circles[i, 1] = test_y
                    test_circles[i, 2] = max_r - 5e-8
                    test_sum = test_circles[:, 2].sum()
                    if test_sum > best_local_sum:
                        best_local_sum = test_sum
                        best_pos = np.array([test_x, test_y])
                        best_radius = max_r - 5e-8
                        circles[i, :2] = best_pos
                        circles[i, 2] = best_radius
                        improved_any = True
                        break
        
        if not improved_any:
            break
    
    # Local perturbation search for final polish (adapted from INSPIRATION 2)
    for perturbation_pass in range(2):
        improved_global = False
        step_sizes = [0.002, 0.001] if perturbation_pass == 0 else [0.0008, 0.0004]
        
        for i in range(n):
            best_local_config = circles.copy()
            best_local_sum = circles[:, 2].sum()
            
            for step_size in step_sizes:
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    dx = step_size * np.cos(angle)
                    dy = step_size * np.sin(angle)
                    
                    test_circles = circles.copy()
                    test_circles[i, 0] += dx
                    test_circles[i, 1] += dy
                    
                    # Compute max radius at new position with ultra-tight tolerance
                    max_r = min(test_circles[i, 0], 1 - test_circles[i, 0],
                               test_circles[i, 1], 1 - test_circles[i, 1])
                    for j in range(n):
                        if i != j:
                            dist = np.linalg.norm(test_circles[j, :2] - test_circles[i, :2])
                            max_r = min(max_r, dist - test_circles[j, 2] - 6e-8)
                    
                    if max_r > 1e-6:
                        test_circles[i, 2] = max_r - 6e-8
                        test_sum = test_circles[:, 2].sum()
                        if test_sum > best_local_sum:
                            best_local_sum = test_sum
                            best_local_config = test_circles.copy()
                            improved_global = True
            
            if improved_global:
                circles = best_local_config
        
        if not improved_global:
            break
    
    # Coordinate descent with ultra-tight constraints (two-pass approach)
    # High-impact targeted optimization before final maximization
    for pass_num in range(2):
        # First pass: all circles; Second pass: focus on smallest radii
        if pass_num == 0:
            indices = range(n)
        else:
            indices = np.argsort(circles[:, 2])[:n//2]  # Focus on smallest half
        
        for i in indices:
            x, y, r = circles[i]
            
            def objective_i(params):
                return -params[2]
            
            def constraints_i(params):
                xi, yi, ri = params
                margin = 1.5e-8 if pass_num == 1 else 2e-8  # Tighter on second pass
                violations = [
                    ri - xi + margin,
                    ri - yi + margin,
                    ri - (1 - xi) + margin,
                    ri - (1 - yi) + margin
                ]
                
                for j in range(n):
                    if i != j:
                        xj, yj, rj = circles[j]
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        violations.append(ri + rj - dist + margin)
                
                return np.array(violations)
            
            try:
                result = minimize(objective_i, [x, y, r], method='SLSQP',
                                bounds=[(0.005, 0.995), (0.005, 0.995), (0.00001, 0.5)],
                                constraints={'type': 'ineq', 'fun': lambda p: -constraints_i(p)},
                                options={'maxiter': 80, 'ftol': 1e-11, 'eps': 1e-10})
                
                if result.success and -result.fun > r * 0.9996:
                    circles[i] = result.x
            except:
                pass
    
    # Final aggressive parallel radius maximization (extended to 6 iterations)
    for iteration in range(6):
        for i in range(n):
            max_r = min(circles[i, 0], 1 - circles[i, 0], 
                       circles[i, 1], 1 - circles[i, 1])
            
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(circles[i, :2] - circles[j, :2])
                    max_r = min(max_r, dist - circles[j, 2])
            
            # Ultra-ultra-aggressive: use 0.99999 for maximum radius (push to absolute limit)
            circles[i, 2] = min(circles[i, 2] * 1.002, max_r * 0.99999)
            circles[i, 2] = max(circles[i, 2], 0.001)
    
    return circles


# EVOLVE-BLOCK-END
