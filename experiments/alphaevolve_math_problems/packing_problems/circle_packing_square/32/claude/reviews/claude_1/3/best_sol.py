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

def initialize_grid(n, config_id=0, spacing=0.148):
    """Initialize circles with parameterized hexagonal spacing (CRITICAL from INSPIRATIONs)."""
    circles = np.zeros((n, 3))
    
    if config_id == 0:
        # Hexagonal close-packing with configurable spacing (INSPIRATION breakthrough)
        row = 0
        idx = 0
        
        while idx < n:
            y = 0.095 + row * spacing * np.sqrt(3) / 2
            if y > 0.905:
                break
            
            offset = (row % 2) * spacing / 2
            col = 0
            while idx < n:
                x = 0.095 + col * spacing + offset
                if x > 0.905:
                    break
                
                # Boundary-aware radius initialization (from INSPIRATION)
                dist_to_boundary = min(x, 1-x, y, 1-y)
                base_r = 0.048 if dist_to_boundary < 0.14 else 0.045
                
                circles[idx] = [x, y, base_r]
                idx += 1
                col += 1
            row += 1
        
        # Fill remaining with random placement
        while idx < n:
            circles[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.04]
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
        # Enhanced Halton with better spacing (from INSPIRATION)
        x_init = halton_sequence(n, 2) * 0.84 + 0.08
        y_init = halton_sequence(n, 3) * 0.84 + 0.08
        circles[:, 0] = x_init
        circles[:, 1] = y_init
        circles[:, 2] = 0.038  # Larger initial radius from INSPIRATION
    
    elif config_id == 3:
        # Vogel spiral
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            theta = i * golden_angle
            r = np.sqrt(i / n) * 0.4 + 0.1
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            circles[i] = [np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95), 0.028]
    
    elif config_id == 4:
        # Fibonacci lattice
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n):
            y = (i / (n - 1)) * 0.9 + 0.05
            x = ((i * phi) % 1) * 0.9 + 0.05
            circles[i] = [x, y, 0.028]
    
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
        
        # Early convergence detection (inspired by INSPIRATION 1)
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
                    
                    for _ in range(20):  # Binary search iterations (maximum precision)
                        mid = (low + high) / 2
                        feasible = mid <= max_r_boundary - 0.00003
                        
                        if feasible:
                            for j in neighbors:
                                dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                                if dist < mid + circles[j, 2] - 3e-11:
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
            
            # Boundary bonus
            dist_to_boundary = min(x, 1-x, y, 1-y)
            boundary_bonus = 1.0 if dist_to_boundary < 0.15 else 0.0
            weighted_potential = potential * (1.0 + boundary_bonus * 0.3)
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

def final_optimization(circles, maxiter=220):
    """SLSQP with analytical Jacobians for speed/accuracy (from INSPIRATION 2)."""
    n = len(circles)
    x0 = circles.flatten()
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.35))  # r
    
    def obj_func(x):
        radii = x[2::3]
        return -np.sum(radii)
    
    def obj_grad(x):
        grad = np.zeros(3 * n)
        grad[2::3] = -1.0
        return grad
    
    constraints = []
    
    # Boundary constraints with analytical Jacobians (from INSPIRATION 2)
    for i in range(n):
        # x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: x[3*idx] - x[3*idx+2],
            'jac': lambda x, idx=i: np.array([1.0 if j == 3*idx else -1.0 if j == 3*idx+2 else 0.0 for j in range(3*n)])
        })
        # x + r <= 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: 1.0 - x[3*idx] - x[3*idx+2],
            'jac': lambda x, idx=i: np.array([-1.0 if j == 3*idx else -1.0 if j == 3*idx+2 else 0.0 for j in range(3*n)])
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: x[3*idx+1] - x[3*idx+2],
            'jac': lambda x, idx=i: np.array([1.0 if j == 3*idx+1 else -1.0 if j == 3*idx+2 else 0.0 for j in range(3*n)])
        })
        # y + r <= 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: 1.0 - x[3*idx+1] - x[3*idx+2],
            'jac': lambda x, idx=i: np.array([-1.0 if j == 3*idx+1 else -1.0 if j == 3*idx+2 else 0.0 for j in range(3*n)])
        })
    
    # Non-overlap constraints with analytical Jacobians (from INSPIRATION 2)
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint(x, i=i, j=j):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-10)
                return dist - x[3*i+2] - x[3*j+2]
            
            def overlap_jac(x, i=i, j=j):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-10)
                jac = np.zeros(3 * n)
                jac[3*i] = dx / dist
                jac[3*i+1] = dy / dist
                jac[3*i+2] = -1.0
                jac[3*j] = -dx / dist
                jac[3*j+1] = -dy / dist
                jac[3*j+2] = -1.0
                return jac
            
            constraints.append({
                'type': 'ineq',
                'fun': overlap_constraint,
                'jac': overlap_jac
            })
    
    try:
        result = minimize(
            obj_func,
            x0,
            method='SLSQP',
            jac=obj_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-9}
        )
        
        if result.success:
            return result.x.reshape(n, 3)
    except:
        pass
    
    return circles

def boundary_refinement(circles, iterations=30):
    """Special optimization for boundary circles which can achieve larger radii."""
    n = len(circles)
    for _ in range(iterations):
        improved = False
        
        # Focus on circles close to boundaries
        for i in range(n):
            x, y = circles[i, :2]
            dist_to_boundary = min(x, 1-x, y, 1-y)
            
            if dist_to_boundary < 0.20:
                max_r = compute_max_safe_radius(circles[i, :2], circles, i, margin=0.0)
                
                if max_r > circles[i, 2]:
                    # Extreme aggressive expansion for boundary circles
                    circles[i, 2] = circles[i, 2] + (max_r - circles[i, 2]) * 0.65
                    improved = True
        
        if not improved:
            break
    
    return circles

def local_search(circles, num_perturbations=3):
    """Local perturbation search to escape local optima."""
    n = len(circles)
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    for _ in range(num_perturbations):
        perturbed = circles.copy()
        indices = np.random.choice(n, size=min(8, n), replace=False)
        
        for i in indices:
            max_shift = 0.02
            perturbed[i, 0] += np.random.uniform(-max_shift, max_shift)
            perturbed[i, 1] += np.random.uniform(-max_shift, max_shift)
            
            perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2] + 0.001, 1 - perturbed[i, 2] - 0.001)
            perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2] + 0.001, 1 - perturbed[i, 2] - 0.001)
        
        perturbed = optimize_positions(perturbed, iterations=20, initial_dt=0.005)
        perturbed = expand_radii(perturbed, steps=50, expansion_factor=0.4)
        
        if np.sum(perturbed[:, 2]) > best_sum:
            best_sum = np.sum(perturbed[:, 2])
            best_circles = perturbed.copy()
    
    return best_circles

def coordinate_wise_radius_maximization(circles, passes=2):
    """Coordinate-wise radius maximization with binary search (from INSPIRATION)."""
    n = len(circles)
    
    for coord_pass in range(passes):
        improved = False
        for i in range(n):
            xi, yi = circles[i, 0], circles[i, 1]
            boundary_limit = min(xi, 1 - xi, yi, 1 - yi)
            low = circles[i, 2]
            high = boundary_limit
            
            # Binary search for maximum feasible radius
            for _ in range(25):
                mid = (low + high) / 2
                feasible = mid <= boundary_limit - 1e-6
                
                if feasible:
                    for j in range(n):
                        if i != j:
                            dist = np.sqrt((xi - circles[j, 0])**2 + (yi - circles[j, 1])**2)
                            if dist < mid + circles[j, 2] + 1e-6:
                                feasible = False
                                break
                
                if feasible:
                    low = mid
                else:
                    high = mid
            
            if low > circles[i, 2]:
                circles[i, 2] = low
                improved = True
        
        if not improved:
            break
    
    return circles

def try_uniform_scaling(circles):
    """Ultra-aggressive three-stage scaling with progressively tighter tolerances."""
    n = len(circles)
    
    def constraint_check(c, tol=-1e-8):
        # Check boundaries with relaxed tolerance
        for i in range(n):
            if c[i, 0] - c[i, 2] < tol or c[i, 0] + c[i, 2] > 1 - tol:
                return False
            if c[i, 1] - c[i, 2] < tol or c[i, 1] + c[i, 2] > 1 - tol:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(c[i, :2] - c[j, :2])
                if dist < c[i, 2] + c[j, 2] - tol:
                    return False
        return True
    
    best_circles = circles.copy()
    
    # Stage 1: Coarse scaling with very relaxed tolerance
    for scale in np.linspace(1.0, 1.20, 70):
        scaled = circles.copy()
        scaled[:, 2] *= scale
        if constraint_check(scaled, tol=-1.5e-8):
            best_circles = scaled
        else:
            break
    
    # Stage 2: Medium refinement
    for delta in np.linspace(0, 0.025, 120):
        scaled = best_circles.copy()
        scaled[:, 2] *= (1.0 + delta)
        if constraint_check(scaled, tol=-1e-9):
            best_circles = scaled
        else:
            break
    
    # Stage 3: Fine refinement
    for micro_delta in np.linspace(0, 0.005, 100):
        scaled = best_circles.copy()
        scaled[:, 2] *= (1.0 + micro_delta)
        if constraint_check(scaled, tol=-5e-10):
            best_circles = scaled
        else:
            break
    
    # Stage 4: Hyper-fine refinement with ultra-relaxed tolerance
    for nano_delta in np.linspace(0, 0.002, 120):
        scaled = best_circles.copy()
        scaled[:, 2] *= (1.0 + nano_delta)
        if constraint_check(scaled, tol=-2e-10):
            best_circles = scaled
        else:
            break
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Combines hexagonal initialization with fine-tuned spacing from INSPIRATIONs.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    import time
    np.random.seed(42)
    n = 32
    
    best_circles = None
    best_sum = 0.0
    benchmark = 2.937944526205518
    quality_threshold = benchmark * 1.0001
    
    start_time = time.time()
    time_limit = 54.0
    
    # Expanded hexagonal spacing exploration (from INSPIRATION analysis)
    configurations = []
    
    # Strategy 1: Ultra-fine hexagonal spacing sweep (20 values for maximum precision)
    for spacing in [0.1465, 0.14665, 0.1468, 0.14695, 0.1471, 0.14725, 0.1474, 0.14755, 
                    0.1477, 0.14785, 0.148, 0.14815, 0.1483, 0.14845, 0.1486, 0.14875,
                    0.1489, 0.14905, 0.1492, 0.14935]:
        configurations.append((0, 42, f'hex_{spacing}', spacing))
    
    # Strategy 2: Alternative strategies as fallback
    configurations.extend([
        (2, 42, 'halton', 0.148),
        (1, 42, 'corner', 0.148),
    ])
    
    for config_idx, config_data in enumerate(configurations):
        if time.time() - start_time > time_limit:
            break
        
        if len(config_data) == 4:
            config_id, seed, strategy_name, spacing = config_data
            circles = initialize_grid(n, config_id, spacing)
        else:
            config_id, seed, strategy_name = config_data
            circles = initialize_grid(n, config_id)
        
        # Optimized pipeline based on INSPIRATION 2's faster convergence
        if 'halton' in strategy_name:
            circles = optimize_positions(circles, iterations=80, initial_dt=0.014)
            circles = expand_radii(circles, steps=195, expansion_factor=0.80, use_binary_search=True)
        elif 'hex' in strategy_name:
            circles = optimize_positions(circles, iterations=60, initial_dt=0.011)
            circles = expand_radii(circles, steps=170, expansion_factor=0.76, use_binary_search=True)
        else:
            circles = optimize_positions(circles, iterations=65, initial_dt=0.012)
            circles = expand_radii(circles, steps=160, expansion_factor=0.74, use_binary_search=True)
        
        circles = optimize_positions(circles, iterations=55, initial_dt=0.008)
        circles = expand_radii(circles, steps=140, expansion_factor=0.60, use_binary_search=True)
        circles = boundary_refinement(circles, iterations=35)
        
        circles = optimize_positions(circles, iterations=35, initial_dt=0.005)
        circles = expand_radii(circles, steps=110, expansion_factor=0.40, use_binary_search=True)
        
        # Quadruple SLSQP pass for ultimate convergence
        for refinement_pass in range(4):
            try:
                optimized = final_optimization(circles, maxiter=220)
                circles = optimized
            except:
                pass
        
        # Streamlined post-SLSQP refinement
        circles = expand_radii(circles, steps=55, expansion_factor=0.19, use_binary_search=True)
        circles = boundary_refinement(circles, iterations=24)
        circles = expand_radii(circles, steps=35, expansion_factor=0.11, use_binary_search=True)
        
        # Coordinate-wise maximization (from INSPIRATION)
        circles = coordinate_wise_radius_maximization(circles, passes=1)
        
        # Aggressive scaling (from INSPIRATION)
        optimized = try_uniform_scaling(circles)
        
        # Final precision adjustments
        for i in range(n):
            optimized[i, 0] = np.clip(optimized[i, 0], optimized[i, 2] + 1e-7, 1 - optimized[i, 2] - 1e-7)
            optimized[i, 1] = np.clip(optimized[i, 1], optimized[i, 2] + 1e-7, 1 - optimized[i, 2] - 1e-7)
            max_r = compute_max_safe_radius(optimized[i, :2], optimized, i, margin=0.0)
            optimized[i, 2] = min(optimized[i, 2], max_r * 0.999998)
            optimized[i, 2] = max(optimized[i, 2], 0.001)
        
        current_sum = np.sum(optimized[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized.copy()
            
            # Early exit if we beat benchmark
            if current_sum > quality_threshold:
                break
    
    # Hyper-aggressive final refinement with maximum optimization
    if best_circles is not None and time.time() - start_time < 55.0:
        # Quadruple coordinate-wise passes for ultimate precision
        best_circles = coordinate_wise_radius_maximization(best_circles, passes=4)
        best_circles = expand_radii(best_circles, steps=70, expansion_factor=0.14, use_binary_search=True)
        best_circles = boundary_refinement(best_circles, iterations=30)
        
        # Extended scaling with 6 attempts for maximum radius
        for scale_attempt in range(6):
            scaled = try_uniform_scaling(best_circles)
            if np.sum(scaled[:, 2]) > np.sum(best_circles[:, 2]):
                best_circles = scaled
                # Double coordinate-wise after each successful scaling
                best_circles = coordinate_wise_radius_maximization(best_circles, passes=2)
        
        # Double SLSQP polish if time permits
        if time.time() - start_time < 57.5:
            for final_pass in range(2):
                try:
                    final_polished = final_optimization(best_circles, maxiter=150)
                    if np.sum(final_polished[:, 2]) > np.sum(best_circles[:, 2]):
                        best_circles = final_polished
                except:
                    pass
        
        # Ultimate final scaling attempt
        if time.time() - start_time < 58.5:
            ultimate_scaled = try_uniform_scaling(best_circles)
            if np.sum(ultimate_scaled[:, 2]) > np.sum(best_circles[:, 2]):
                best_circles = ultimate_scaled
    
    return best_circles


# EVOLVE-BLOCK-END
