# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def halton_sequence(n, base):
    """Generate n points of Halton low-discrepancy sequence."""
    sequence = []
    for i in range(1, n + 1):
        f, r = 1.0, 0.0
        index = i
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index = index // base
        sequence.append(r)
    return np.array(sequence)

def initialize_grid(n, config_id=0):
    """Initialize circles with multiple grid configurations and boundary-aware sizing."""
    circles = np.zeros((n, 3))
    
    if config_id == 0:
        # 4x8 grid with hexagonal offset and boundary-aware radii
        grid_x, grid_y = 4, 8
        spacing_x = 1.0 / grid_x
        spacing_y = 1.0 / grid_y
        idx = 0
        for i in range(grid_y):
            for j in range(grid_x):
                if idx < n:
                    offset = spacing_x * 0.25 if i % 2 == 1 else 0
                    x = (j + 0.5) * spacing_x + offset
                    y = (i + 0.5) * spacing_y
                    
                    # Larger initial radii for boundary circles
                    dist_to_boundary = min(x, 1-x, y, 1-y)
                    base_r = min(spacing_x, spacing_y) * 0.35
                    if dist_to_boundary < 0.15:
                        base_r *= 1.15  # 15% larger for boundary circles
                    
                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    circles[idx, 2] = base_r
                    idx += 1
    
    elif config_id == 1:
        # Corner-priority initialization with explicit corner and edge placement
        positions = []
        
        # 4 corners with larger initial radii
        margin = 0.12
        positions.extend([
            (margin, margin, 0.10),
            (1-margin, margin, 0.10),
            (margin, 1-margin, 0.10),
            (1-margin, 1-margin, 0.10)
        ])
        
        # Edge positions (12 circles, 3 per edge)
        for i in range(3):
            t = 0.25 + i * 0.25
            positions.extend([
                (t, 0.09, 0.08),  # Bottom edge
                (t, 1-0.09, 0.08),  # Top edge
                (0.09, t, 0.08),  # Left edge
                (1-0.09, t, 0.08)  # Right edge
            ])
        
        # Interior grid (16 circles in 4x4)
        for i in range(4):
            for j in range(4):
                x = 0.24 + j * 0.17
                y = 0.24 + i * 0.17
                positions.append((x, y, 0.07))
        
        for idx, (x, y, r) in enumerate(positions[:n]):
            circles[idx] = [x, y, r]
    
    elif config_id == 2:
        # 6x6 grid with boundary-priority optimization
        grid_size = 6
        spacing = 1.0 / grid_size
        positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                dist_to_boundary = min(x, 1-x, y, 1-y)
                
                # Prioritize boundary positions
                priority = dist_to_boundary * 2 + min(x, 1-x) * min(y, 1-y) * 0.5
                positions.append((x, y, priority))
        
        positions.sort(key=lambda p: p[2], reverse=True)
        
        for idx in range(n):
            x, y, _ = positions[idx]
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = spacing * 0.38  # Slightly larger initial radius
    
    elif config_id == 3:
        # Halton sequence initialization (seed 42)
        x_init = halton_sequence(n, 2) * 0.82 + 0.09
        y_init = halton_sequence(n, 3) * 0.82 + 0.09
        circles[:, 0] = x_init
        circles[:, 1] = y_init
        circles[:, 2] = 0.025  # Start with small radii
    
    elif config_id == 4:
        # Halton sequence with offset (seed 123 equivalent)
        x_init = halton_sequence(n, 5) * 0.80 + 0.10
        y_init = halton_sequence(n, 7) * 0.80 + 0.10
        circles[:, 0] = x_init
        circles[:, 1] = y_init
        circles[:, 2] = 0.028
    
    elif config_id == 5:
        # Sunflower spiral pattern (optimal space-filling from INSPIRATIONS)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            theta = i * golden_angle
            r = 0.42 * np.sqrt(i / n)
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            circles[i, 0] = np.clip(x, 0.05, 0.95)
            circles[i, 1] = np.clip(y, 0.05, 0.95)
            circles[i, 2] = 0.035
    
    else:
        # Halton sequence with different scaling
        x_init = halton_sequence(n, 2) * 0.78 + 0.11
        y_init = halton_sequence(n, 3) * 0.78 + 0.11
        circles[:, 0] = x_init
        circles[:, 1] = y_init
        circles[:, 2] = 0.022
    
    # Ensure bounds
    for i in range(n):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2] + 0.001, 1 - circles[i, 2] - 0.001)
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2] + 0.001, 1 - circles[i, 2] - 0.001)
    
    return circles

def compute_max_safe_radius(pos, all_circles, idx, margin=0.0):
    """Compute maximum radius for circle at pos without violating constraints."""
    x, y = pos
    max_r = min(x, y, 1-x, 1-y) - margin
    
    for i in range(len(all_circles)):
        if i == idx:
            continue
        ox, oy, or_ = all_circles[i]
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        max_r = min(max_r, dist - or_ - margin)
    
    return max(0.0, max_r)

def optimize_positions(circles, iterations=50, initial_dt=0.01):
    """Adaptive physics-based position optimization with convergence detection."""
    n = len(circles)
    positions = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    velocities = np.zeros_like(positions)
    
    for iter_num in range(iterations):
        forces = np.zeros_like(positions)
        dt = initial_dt * (1.0 - 0.7 * iter_num / iterations)
        
        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist * 1.3 and dist > 1e-10:
                    force_dir = diff / dist
                    if dist < min_dist:
                        force_mag = (min_dist - dist) * 20.0
                    else:
                        force_mag = (min_dist - dist) * 3.0
                    
                    forces[i] += force_dir * force_mag
                    forces[j] -= force_dir * force_mag
        
        # Boundary repulsion
        for i in range(n):
            r = radii[i]
            margin = 0.03
            
            for dim in range(2):
                if positions[i, dim] < r + margin:
                    forces[i, dim] += (r + margin - positions[i, dim]) * 10.0
                if positions[i, dim] > 1 - r - margin:
                    forces[i, dim] -= (positions[i, dim] - (1 - r - margin)) * 10.0
        
        # Update with damping
        velocities = velocities * 0.7 + forces * dt
        positions += velocities * dt
        
        # Hard boundaries
        for i in range(n):
            for dim in range(2):
                if positions[i, dim] < radii[i] + 0.001:
                    positions[i, dim] = radii[i] + 0.001
                    velocities[i, dim] = 0
                if positions[i, dim] > 1 - radii[i] - 0.001:
                    positions[i, dim] = 1 - radii[i] - 0.001
                    velocities[i, dim] = 0
        
        # Early convergence detection
        if np.max(np.abs(velocities)) < 1e-7:
            break
    
    circles[:, :2] = positions
    return circles

def expand_radii(circles, steps=100, expansion_factor=0.5, use_binary_search=False):
    """Location-aware radius expansion with KDTree optimization and optional binary search."""
    n = len(circles)
    
    for step in range(steps):
        potentials = []
        tree = KDTree(circles[:, :2])
        
        for i in range(n):
            x, y = circles[i, :2]
            max_r_boundary = min(x, 1-x, y, 1-y)
            
            # Use KDTree for efficient neighbor search
            search_radius = min(2 * max_r_boundary, 0.45)
            neighbors = tree.query_ball_point([x, y], r=search_radius)
            neighbors = [j for j in neighbors if j != i]
            
            if use_binary_search and len(neighbors) > 0:
                # Binary search for maximum feasible radius (more precise)
                max_r_neighbors = float('inf')
                for j in neighbors:
                    dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                    max_r_neighbors = min(max_r_neighbors, dist - circles[j, 2])
                
                max_r = min(max_r_boundary, max_r_neighbors) - 0.0001
                
                if max_r > circles[i, 2] * 1.00005:
                    low, high = circles[i, 2], max_r
                    best_r = circles[i, 2]
                    
                    for _ in range(26):  # Binary search iterations (theoretical maximum precision)
                        mid = (low + high) / 2
                        feasible = mid <= max_r_boundary - 0.00001
                        
                        if feasible:
                            for j in neighbors:
                                dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                                if dist < mid + circles[j, 2] - 2e-11:
                                    feasible = False
                                    break
                        
                        if feasible:
                            best_r = mid
                            low = mid
                        else:
                            high = mid
                    
                    potential = best_r - circles[i, 2]
                else:
                    potential = 0.0
            else:
                # Standard method
                max_r = compute_max_safe_radius(circles[i, :2], circles, i, margin=0.0001)
                potential = max_r - circles[i, 2]
            
            # Enhanced boundary bonus (from INSPIRATIONS)
            dist_to_boundary = min(x, 1-x, y, 1-y)
            boundary_bonus = 1.0 if dist_to_boundary < 0.15 else 0.0
            weighted_potential = potential * (1.0 + boundary_bonus * 0.4)
            potentials.append((i, potential, weighted_potential))
        
        potentials.sort(key=lambda x: x[2], reverse=True)
        
        improved = False
        for i, potential, _ in potentials:
            if potential > 0.00003:
                # More aggressive expansion for boundary circles
                x, y = circles[i, :2]
                dist_to_boundary = min(x, 1-x, y, 1-y)
                location_multiplier = 1.2 if dist_to_boundary < 0.15 else 1.0
                
                # Adaptive expansion rate
                if step < steps // 3:
                    factor = expansion_factor * 0.85 * location_multiplier
                elif step < 2 * steps // 3:
                    factor = expansion_factor * 0.45 * location_multiplier
                else:
                    factor = expansion_factor * 0.18
                
                circles[i, 2] = circles[i, 2] + potential * factor
                improved = True
        
        if not improved:
            break
        
        circles[:, 2] = np.maximum(circles[:, 2], 0.001)
    
    return circles

def final_optimization(circles):
    """SLSQP with analytical Jacobians for 10-100× speedup (from INSPIRATIONS)."""
    n = len(circles)
    
    def objective(x):
        c = x.reshape(n, 3)
        return -np.sum(c[:, 2])
    
    def objective_grad(x):
        grad = np.zeros(3*n)
        grad[2::3] = -1.0
        return grad
    
    def constraint_non_overlap(x):
        c = x.reshape(n, 3)
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(c[i, :2] - c[j, :2])
                constraints.append(dist - c[i, 2] - c[j, 2])
        return np.array(constraints)
    
    def constraint_non_overlap_jac(x):
        c = x.reshape(n, 3)
        n_constraints = n * (n - 1) // 2
        jac = np.zeros((n_constraints, 3*n))
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 1e-10:
                    jac[idx, 3*i] = dx / dist
                    jac[idx, 3*i+1] = dy / dist
                    jac[idx, 3*i+2] = -1.0
                    jac[idx, 3*j] = -dx / dist
                    jac[idx, 3*j+1] = -dy / dist
                    jac[idx, 3*j+2] = -1.0
                
                idx += 1
        
        return jac
    
    def constraint_boundary(x):
        c = x.reshape(n, 3)
        constraints = []
        for i in range(n):
            constraints.append(c[i, 0] - c[i, 2])
            constraints.append(1 - c[i, 0] - c[i, 2])
            constraints.append(c[i, 1] - c[i, 2])
            constraints.append(1 - c[i, 1] - c[i, 2])
            constraints.append(c[i, 2])
        return np.array(constraints)
    
    def constraint_boundary_jac(x):
        n_constraints = 5 * n
        jac = np.zeros((n_constraints, 3*n))
        
        for i in range(n):
            jac[5*i, 3*i] = 1.0
            jac[5*i, 3*i+2] = -1.0
            jac[5*i+1, 3*i] = -1.0
            jac[5*i+1, 3*i+2] = -1.0
            jac[5*i+2, 3*i+1] = 1.0
            jac[5*i+2, 3*i+2] = -1.0
            jac[5*i+3, 3*i+1] = -1.0
            jac[5*i+3, 3*i+2] = -1.0
            jac[5*i+4, 3*i+2] = 1.0
        
        return jac
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_non_overlap, 'jac': constraint_non_overlap_jac},
        {'type': 'ineq', 'fun': constraint_boundary, 'jac': constraint_boundary_jac}
    ]
    
    x0 = circles.flatten()
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    try:
        result = minimize(objective, x0, method='SLSQP', 
                        jac=objective_grad,
                        constraints=constraints, bounds=bounds,
                        options={'maxiter': 400, 'ftol': 1e-10})
        if result.success:
            return result.x.reshape(n, 3)
    except:
        pass
    
    return circles

def boundary_refinement(circles, iterations=30):
    """Special optimization for boundary circles with adaptive thresholds (from INSPIRATIONS)."""
    n = len(circles)
    for iter_num in range(iterations):
        improved = False
        
        # Adaptive boundary threshold - expand search area as iterations progress
        boundary_threshold = 0.20 if iter_num < iterations // 2 else 0.25
        
        for i in range(n):
            x, y = circles[i, :2]
            dist_to_boundary = min(x, 1-x, y, 1-y)
            
            if dist_to_boundary < boundary_threshold:
                max_r = compute_max_safe_radius(circles[i, :2], circles, i, margin=0.0)
                
                if max_r > circles[i, 2]:
                    # Ultra-aggressive expansion with adaptive rate
                    expansion_rate = 0.72 if iter_num < iterations // 2 else 0.55
                    circles[i, 2] = circles[i, 2] + (max_r - circles[i, 2]) * expansion_rate
                    improved = True
        
        if not improved:
            break
    
    return circles

def local_search(circles, num_perturbations=5):
    """Enhanced local search with varying perturbation strategies (from INSPIRATIONS)."""
    n = len(circles)
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    for perturbation_idx in range(num_perturbations):
        perturbed = circles.copy()
        
        # Vary perturbation strategy based on iteration
        if perturbation_idx < 2:
            # Small perturbations (fine-tuning)
            indices = np.random.choice(n, size=min(6, n), replace=False)
            max_shift = 0.015
        else:
            # Larger perturbations (escape local optima)
            indices = np.random.choice(n, size=min(10, n), replace=False)
            max_shift = 0.03
        
        for i in indices:
            perturbed[i, 0] += np.random.uniform(-max_shift, max_shift)
            perturbed[i, 1] += np.random.uniform(-max_shift, max_shift)
            
            perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2] + 0.001, 1 - perturbed[i, 2] - 0.001)
            perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2] + 0.001, 1 - perturbed[i, 2] - 0.001)
        
        perturbed = optimize_positions(perturbed, iterations=25, initial_dt=0.006)
        perturbed = expand_radii(perturbed, steps=60, expansion_factor=0.45)
        
        if np.sum(perturbed[:, 2]) > best_sum:
            best_sum = np.sum(perturbed[:, 2])
            best_circles = perturbed.copy()
    
    return best_circles

def try_uniform_scaling(circles, max_scale=1.20, num_points=60, tolerance=-5e-8):
    """Extreme precision uniform scaling with configurable tolerance (from INSPIRATIONS)."""
    n = len(circles)
    
    def constraint_check(x):
        c = x.reshape(n, 3)
        # Boundary constraints with extreme tolerance
        for i in range(n):
            if c[i, 0] - c[i, 2] < tolerance or c[i, 0] + c[i, 2] > 1 - tolerance:
                return False
            if c[i, 1] - c[i, 2] < tolerance or c[i, 1] + c[i, 2] > 1 - tolerance:
                return False
        # Non-overlap constraints with extreme tolerance
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(c[i, :2] - c[j, :2])
                if dist < c[i, 2] + c[j, 2] - tolerance:
                    return False
        return True
    
    best_circles = circles.copy()
    for scale in np.linspace(1.0, max_scale, num_points):
        scaled = circles.copy()
        scaled[:, 2] *= scale
        if constraint_check(scaled.flatten()):
            best_circles = scaled
        else:
            break
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses time-budgeted multi-start optimization with uniform scaling (from INSPIRATION 1).

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    import time
    np.random.seed(42)
    n = 32
    
    best_circles = None
    best_sum = 0.0
    benchmark = 2.937944526205518
    quality_threshold = benchmark * 0.9996  # Early termination at 99.96%
    
    start_time = time.time()
    time_limit = 54.0
    
    # Explore all 6 configurations with Jacobian speedup (from INSPIRATIONS)
    for config_id in range(6):
        if time.time() - start_time > time_limit:
            break
        
        circles = initialize_grid(n, config_id)
        
        # Adaptive strategy with conditional perturbation (from INSPIRATION 1)
        if config_id == 3:  # Halton needs more optimization
            circles = optimize_positions(circles, iterations=95, initial_dt=0.015)
            circles = expand_radii(circles, steps=220, expansion_factor=0.82, use_binary_search=False)
            use_perturbation = True
        else:  # Structured grids converge faster
            circles = optimize_positions(circles, iterations=70, initial_dt=0.012)
            circles = expand_radii(circles, steps=180, expansion_factor=0.75, use_binary_search=False)
            use_perturbation = False
        
        circles = optimize_positions(circles, iterations=60, initial_dt=0.008)
        circles = expand_radii(circles, steps=160, expansion_factor=0.62, use_binary_search=True)
        circles = boundary_refinement(circles, iterations=40)
        
        if use_perturbation:
            circles = local_search(circles, num_perturbations=2)
        
        circles = optimize_positions(circles, iterations=40, initial_dt=0.005)
        circles = expand_radii(circles, steps=130, expansion_factor=0.42, use_binary_search=True)
        circles = boundary_refinement(circles, iterations=30)
        
        # Triple SLSQP with Jacobians (fast and effective from INSPIRATIONS)
        optimized = final_optimization(circles)
        optimized = final_optimization(optimized)
        optimized = final_optimization(optimized)
        
        # Optimized five-stage uniform scaling with reduced iterations for speed (from INSPIRATION 1)
        # Stage 1: Coarse aggressive scaling (reduced from 100 to 80)
        for scale in np.linspace(1.0, 1.20, 80):
            scaled = try_uniform_scaling(optimized, max_scale=scale, num_points=1, tolerance=-3e-8)
            if np.sum(scaled[:, 2]) > np.sum(optimized[:, 2]):
                optimized = scaled
            else:
                break
        
        # Stage 2: Fine-grained refinement (reduced from 200 to 150)
        for delta in np.linspace(0, 0.030, 150):
            scaled = try_uniform_scaling(optimized, max_scale=1.0 + delta, num_points=1, tolerance=-3e-8)
            if np.sum(scaled[:, 2]) > np.sum(optimized[:, 2]):
                optimized = scaled
            else:
                break
        
        # Stage 3: Ultra-fine micro-scaling (reduced from 150 to 100)
        current_scale = np.sum(optimized[:, 2]) / np.sum(circles[:, 2])
        for delta in np.linspace(0, 0.008, 100):
            scaled = try_uniform_scaling(optimized, max_scale=current_scale * (1.0 + delta), num_points=1, tolerance=-3e-8)
            if np.sum(scaled[:, 2]) > np.sum(optimized[:, 2]):
                optimized = scaled
            else:
                break
        
        # Stage 4: Hyper-fine final push (reduced from 120 to 80, tighter tolerance)
        for delta in np.linspace(0, 0.003, 80):
            scaled = try_uniform_scaling(optimized, max_scale=1.0 + delta, num_points=1, tolerance=-5e-8)
            if np.sum(scaled[:, 2]) > np.sum(optimized[:, 2]):
                optimized = scaled
            else:
                break
        
        # Stage 5: Ultimate nano-scaling (reduced from 100 to 60, extreme tolerance)
        for delta in np.linspace(0, 0.001, 60):
            scaled = try_uniform_scaling(optimized, max_scale=1.0 + delta, num_points=1, tolerance=-6e-8)
            if np.sum(scaled[:, 2]) > np.sum(optimized[:, 2]):
                optimized = scaled
            else:
                break
        
        # Streamlined post-scaling refinement (reduced steps for speed)
        optimized = expand_radii(optimized, steps=60, expansion_factor=0.24, use_binary_search=True)
        optimized = boundary_refinement(optimized, iterations=35)
        optimized = expand_radii(optimized, steps=40, expansion_factor=0.14, use_binary_search=True)
        optimized = boundary_refinement(optimized, iterations=20)
        
        # Double final verification SLSQP for ultimate precision
        optimized = final_optimization(optimized)
        optimized = final_optimization(optimized)
        
        # Ultra-precise final adjustment with extreme tolerances
        for i in range(n):
            optimized[i, 0] = np.clip(optimized[i, 0], optimized[i, 2] + 1e-8, 1 - optimized[i, 2] - 1e-8)
            optimized[i, 1] = np.clip(optimized[i, 1], optimized[i, 2] + 1e-8, 1 - optimized[i, 2] - 1e-8)
            max_r = compute_max_safe_radius(optimized[i, :2], optimized, i, margin=0.0)
            optimized[i, 2] = min(optimized[i, 2], max_r * 0.99999999)
            optimized[i, 2] = max(optimized[i, 2], 0.001)
        
        current_sum = np.sum(optimized[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized.copy()
            
            # Early termination if quality threshold reached
            if current_sum > quality_threshold:
                break
    
    return best_circles


# EVOLVE-BLOCK-END
