# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

# EXPLORATION: Enhanced physics-based approach with hexagonal initialization
# Combines INSPIRATION 1's efficient hexagonal lattice with improved relaxation and optimization

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize sum of radii.
    
    Optimized hybrid strategy combining INSPIRATION 1's efficiency with streamlined refinement:
    1. Multi-strategy initialization with early stopping for excellent results
    2. Vectorized physics relaxation with momentum
    3. Aggressive initial radii computation
    4. Smart restart logic for promising solutions only
    5. Consolidated micro-adjustment phase (replacing micro/ultra/nano redundancy)
    6. SLSQP optimization with adaptive iterations
    7. Post-optimization incremental inflation (from INSPIRATION 1)
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 32
    
    best_circles = None
    best_sum = 0
    target_threshold = 2.944  # Slightly higher threshold for more thorough exploration
    
    # Ordered strategies: hexagonal first (best performer), then variants
    strategies = [
        ('hexagonal', initialize_hexagonal_grid(n)),
        ('hexagonal_variant', initialize_hexagonal_grid_variant(n, spacing=0.170, offset=0.09)),
        ('4x8_grid', initialize_4x8_grid(n)),
    ]
    
    for strategy_name, initial_positions in strategies:
        # Phase 1: Physics relaxation with momentum (extended for better equilibrium)
        positions = physics_relaxation_with_momentum(initial_positions.copy(), iterations=360)
        
        # Phase 2: Aggressive initial radii
        radii = compute_maximum_radii(positions, safety_factor=0.998)
        circles = np.column_stack([positions, radii])
        
        # Phase 3: SLSQP optimization (adaptive iterations based on strategy)
        slsqp_iters = 180 if strategy_name == 'hexagonal' else 150
        circles = local_refinement(circles, iterations=slsqp_iters)
        
        # Phase 4: Smart restart (only for very promising solutions >2.88)
        # Extended to 4 trials for even better exploration when we have a good solution
        if np.sum(circles[:, 2]) > 2.88:
            for trial in range(4):
                perturbed = circles.copy()
                if trial == 0:
                    scale = 0.007
                elif trial == 1:
                    scale = 0.011
                elif trial == 2:
                    scale = 0.005
                else:  # trial == 3
                    scale = 0.009  # Intermediate scale
                perturbed[:, :2] += np.random.randn(n, 2) * scale
                perturbed[:, :2] = np.clip(perturbed[:, :2], 0.01, 0.99)
                optimized = local_refinement(perturbed, iterations=120)
                if is_valid_configuration(optimized) and np.sum(optimized[:, 2]) > np.sum(circles[:, 2]):
                    circles = optimized
        
        # Phase 5: Ultra-aggressive radius expansion
        circles = expand_radii_ultra_aggressive(circles, iterations=180)
        
        # Phase 6: Consolidated micro-adjustment (replaces micro/ultra/nano redundancy)
        circles = consolidated_micro_adjustment(circles, iterations=80)
        
        # Phase 7: Post-optimization incremental inflation (from INSPIRATION 1)
        circles = inflate_radii(circles)
        
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_circles = circles
            best_sum = current_sum
        
        # Early stopping: skip remaining strategies if we've achieved excellent results
        if best_sum > target_threshold:
            break
    
    return best_circles


def initialize_hexagonal_grid(n):
    """Initialize positions in hexagonal pattern for optimal space utilization."""
    # Hexagonal packing is provably optimal for 2D circle packing
    rows = int(np.ceil(np.sqrt(n / 0.866)))  # 0.866 ≈ sqrt(3)/2
    cols = int(np.ceil(n / rows))
    
    positions = []
    margin = 0.04  # Reduced margin for more space
    
    x_spacing = (1 - 2*margin) / (cols - 1) if cols > 1 else 0.5
    y_spacing = (1 - 2*margin) / (rows - 1) if rows > 1 else 0.5
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            # Hexagonal offset pattern
            x = margin + col * x_spacing + (0.5 * x_spacing if row % 2 == 1 and cols > 1 else 0)
            y = margin + row * y_spacing
            
            x = np.clip(x, margin, 1 - margin)
            y = np.clip(y, margin, 1 - margin)
            
            positions.append([x, y])
            count += 1
    
    return np.array(positions[:n])


def physics_relaxation_with_momentum(positions, iterations=320, dt=0.015):
    """
    Enhanced physics relaxation with momentum term and early convergence detection.
    From INSPIRATION 1 - proven more efficient than basic relaxation.
    """
    pos = positions.copy()
    n = len(pos)
    velocities = np.zeros_like(pos)
    momentum = 0.7
    prev_energy = np.inf
    convergence_threshold = 1e-6
    stagnation_count = 0
    
    for iteration in range(iterations):
        forces = np.zeros_like(pos)
        
        # Vectorized pairwise force computation
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-6
        np.fill_diagonal(dist, np.inf)
        
        # Inverse square repulsion
        force_magnitude = 0.002 / (dist ** 2 + 0.001)
        force_magnitude = force_magnitude[:, :, np.newaxis]
        direction = diff / dist[:, :, np.newaxis]
        forces = np.sum(force_magnitude * direction, axis=1)
        
        # Boundary containment forces
        margin = 0.04
        boundary_strength = 0.015
        forces[:, 0] += boundary_strength / (pos[:, 0] - 0 + 0.01)
        forces[:, 0] -= boundary_strength / (1 - pos[:, 0] + 0.01)
        forces[:, 1] += boundary_strength / (pos[:, 1] - 0 + 0.01)
        forces[:, 1] -= boundary_strength / (1 - pos[:, 1] + 0.01)
        
        # Update with momentum
        damping = 0.85 if iteration < 160 else 0.9
        velocities = momentum * velocities + (1 - momentum) * forces
        pos += dt * velocities
        pos = np.clip(pos, margin, 1 - margin)
        
        # Early convergence detection (check every 20 iterations after initial phase)
        if iteration > 100 and iteration % 20 == 0:
            energy = np.sum(np.linalg.norm(velocities, axis=1))
            if abs(energy - prev_energy) < convergence_threshold:
                stagnation_count += 1
                if stagnation_count >= 2:  # Converged for 40 iterations
                    break
            else:
                stagnation_count = 0
            prev_energy = energy
        
        if iteration % 100 == 0 and iteration > 0:
            dt *= 0.95
    
    return pos


def compute_maximum_radii(positions, safety_factor=0.998):
    """
    Compute maximum radius per circle with configurable aggressive safety factor.
    From INSPIRATION 1 - more aggressive than INSPIRATION 2's 0.995.
    """
    n = len(positions)
    radii = np.zeros(n)
    dist_matrix = squareform(pdist(positions))
    np.fill_diagonal(dist_matrix, np.inf)
    
    for i in range(n):
        x, y = positions[i]
        r_boundary = min(x, y, 1-x, 1-y)
        r_neighbor = np.min(dist_matrix[i]) / 2.0
        radii[i] = min(r_boundary, r_neighbor) * safety_factor
    
    return radii


def compute_max_radius_single(pos, idx, positions, radii):
    """Compute maximum feasible radius for a single circle at given position."""
    max_r = min(pos[0], pos[1], 1.0 - pos[0], 1.0 - pos[1])
    n = len(positions)
    for j in range(n):
        if j != idx:
            dist = np.linalg.norm(pos - positions[j])
            max_r = min(max_r, dist - radii[j])
    return max(0, max_r)


def initialize_hexagonal_grid_variant(n, spacing=0.170, offset=0.09):
    """Alternative hexagonal pattern with adjustable spacing (from INSPIRATION 1)."""
    positions = []
    rows = 7
    cols = 6
    
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= n:
                break
            x = offset + col * spacing
            if row % 2 == 1:
                x += spacing / 2
            y = offset + row * spacing * 0.866
            
            if 0.03 < x < 0.97 and 0.03 < y < 0.97:
                positions.append([x, y])
    
    while len(positions) < n:
        x, y = np.random.uniform(0.12, 0.88, 2)
        positions.append([x, y])
    
    return np.array(positions[:n])


def initialize_4x8_grid(n):
    """4x8 grid pattern specifically optimized for 32 circles."""
    positions = []
    rows, cols = 4, 8
    x_spacing = 0.88 / (cols - 1)
    y_spacing = 0.88 / (rows - 1)
    x_offset = 0.06
    y_offset = 0.06
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = x_offset + j * x_spacing
            y = y_offset + i * y_spacing
            positions.append([x, y])
    
    return np.array(positions[:n])


def expand_radii_ultra_aggressive(circles, iterations=180):
    """
    Optimized priority-based ultra-aggressive radius expansion (from INSPIRATION 1).
    Vectorized potential computation and early termination for efficiency.
    """
    n = len(circles)
    positions = circles[:, :2]
    radii = circles[:, 2].copy()
    no_improvement_count = 0
    
    for iteration in range(iterations):
        # Vectorized expansion potential computation
        boundary_limits = np.minimum(
            np.minimum(positions[:, 0], positions[:, 1]),
            np.minimum(1 - positions[:, 0], 1 - positions[:, 1])
        )
        
        # Vectorized neighbor distance computation
        dist_matrix = squareform(pdist(positions))
        np.fill_diagonal(dist_matrix, np.inf)
        min_neighbor_dists = np.min(dist_matrix - radii, axis=1)
        
        potential = np.minimum(boundary_limits, min_neighbor_dists) - radii
        
        # Process in order of decreasing potential
        order = np.argsort(-potential)
        improved = False
        
        # Progressive safety margin: 0.999 → 0.9998
        safety = 0.999 + 0.0008 * min(iteration / 180, 1.0)
        
        for i in order:
            if potential[i] < 1e-6:  # Skip circles with negligible potential
                continue
                
            existing = np.vstack([circles[:i], circles[i+1:]]) if i < n - 1 else circles[:i]
            max_r = compute_max_radius_for_circle(circles[i, :2], existing)
            new_r = max_r * safety
            
            if new_r > circles[i, 2]:
                circles[i, 2] = new_r
                radii[i] = new_r
                improved = True
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= 3 or iteration > 60:
                break
        else:
            no_improvement_count = 0
    
    return circles


def compute_max_radius_for_circle(pos, existing_circles):
    """Compute maximum feasible radius for a circle at given position."""
    x, y = pos
    max_r = min(x, 1-x, y, 1-y)
    
    if len(existing_circles) > 0:
        for cx, cy, cr in existing_circles:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_r = min(max_r, dist - cr)
    
    return max(0, max_r)


def consolidated_micro_adjustment(circles, iterations=80):
    """
    Consolidated micro-adjustment phase replacing redundant micro/ultra/nano loops.
    Single efficient pass with progressive step size reduction.
    """
    n = len(circles)
    
    # Precompute angular directions
    angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    for iteration in range(iterations):
        order = np.arange(n)
        if iteration % 2 == 0:
            np.random.shuffle(order)
        else:
            order = np.argsort(circles[:, 2])
        
        improved = False
        
        # Progressive step size: 0.004 → 0.0002
        step = 0.004 * (1 - iteration / iterations) ** 0.8
        
        # Progressive safety margin: 0.9998 → 0.99992 (slightly more aggressive)
        safety = 0.9998 + 0.00012 * min(iteration / iterations, 1.0)
        
        for i in order:
            existing = np.vstack([circles[:i], circles[i+1:]]) if i < n - 1 else circles[:i]
            max_r = compute_max_radius_for_circle(circles[i, :2], existing)
            new_r = max_r * safety
            
            if new_r > circles[i, 2] * 1.000015:
                circles[i, 2] = new_r
                improved = True
            
            # Angular position adjustments every 4 iterations
            if iteration % 4 == 0:
                best_pos = circles[i, :2].copy()
                best_r = circles[i, 2]
                
                # Vectorized position testing
                test_positions = circles[i, :2] + step * np.column_stack([cos_angles, sin_angles])
                
                for idx, test_pos in enumerate(test_positions):
                    if 0.001 < test_pos[0] < 0.999 and 0.001 < test_pos[1] < 0.999:
                        test_max_r = compute_max_radius_for_circle(test_pos, existing)
                        if test_max_r > best_r:
                            best_pos = test_pos
                            best_r = test_max_r * safety
                            improved = True
                
                if improved:
                    circles[i, :2] = best_pos
                    circles[i, 2] = best_r
        
        if not improved:
            break
    
    return circles


def inflate_radii(circles):
    """
    Ultra-fine post-optimization incremental radius inflation (from INSPIRATION 1).
    Proven most effective technique for extracting final gains.
    """
    n = len(circles)
    best_circles = circles.copy()
    
    # Four passes with progressively finer increments (added one more pass)
    for pass_num in range(4):
        improved_pass = False
        
        # Order by current radius (smaller circles first - more room to grow)
        order = np.argsort(best_circles[:, 2])
        
        for i in order:
            improved_circle = False
            
            # Fine increment schedule (added ultra-fine pass_num==3)
            if pass_num == 0:
                increments = [0.0005, 0.00025, 0.00012, 0.00006, 0.00003]
            elif pass_num == 1:
                increments = [0.000015, 0.000008, 0.000004, 0.000002]
            elif pass_num == 2:
                increments = [0.000001, 0.0000005, 0.0000002]
            else:  # pass_num == 3: ultra-fine (extended with more granular steps)
                increments = [0.00000020, 0.00000015, 0.00000010, 0.00000006, 0.00000003]
            
            for increment in increments:
                test_circles = best_circles.copy()
                test_circles[i, 2] += increment
                
                if is_valid_configuration(test_circles):
                    best_circles = test_circles
                    improved_circle = True
                    improved_pass = True
                else:
                    break
            
            if not improved_circle and pass_num > 1:
                continue
        
        if not improved_pass:
            break
    
    # Final greedy maximization pass (extended to 15 iterations with adaptive safety)
    for iter_num in range(15):
        improved = False
        # Progressively more aggressive safety factor
        safety_final = 0.999999 + 0.0000005 * min(iter_num / 15, 1.0)
        
        for i in range(n):
            existing = np.vstack([best_circles[:i], best_circles[i+1:]]) if i < n - 1 else best_circles[:i]
            max_r = compute_max_radius_for_circle(best_circles[i, :2], existing)
            new_r = max_r * safety_final
            if new_r > best_circles[i, 2]:
                best_circles[i, 2] = new_r
                improved = True
        if not improved:
            break
    
    return best_circles


def is_valid_configuration(circles):
    """Fast feasibility check for circle configuration (from INSPIRATION 1)."""
    n = len(circles)
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Check boundary constraints
    if np.any(positions[:, 0] - radii < -1e-8) or np.any(positions[:, 0] + radii > 1 + 1e-8):
        return False
    if np.any(positions[:, 1] - radii < -1e-8) or np.any(positions[:, 1] + radii > 1 + 1e-8):
        return False
    
    # Check overlap constraints using vectorized computation
    dist_matrix = squareform(pdist(positions))
    sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
    np.fill_diagonal(sum_radii, 0)
    np.fill_diagonal(dist_matrix, np.inf)
    
    if np.any(dist_matrix < sum_radii - 1e-8):
        return False
    
    return True


def local_refinement(circles, iterations=180):
    """
    SLSQP optimization with ultra-tight tolerances (from INSPIRATION 1).
    """
    n = len(circles)
    
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    def constraints_func(params):
        pos = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        violations = []
        
        violations.extend(pos[:, 0] - radii)
        violations.extend(1 - pos[:, 0] - radii)
        violations.extend(pos[:, 1] - radii)
        violations.extend(1 - pos[:, 1] - radii)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                violations.append(dist - radii[i] - radii[j])
        
        return np.array(violations)
    
    x0 = np.concatenate([circles[:, :2].flatten(), circles[:, 2]])
    bounds = [(0.002, 0.998)] * (2*n) + [(0.0003, 0.5)] * n
    constraint = {'type': 'ineq', 'fun': constraints_func}
    
    try:
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraint,
            options={'maxiter': iterations, 'ftol': 1e-10, 'eps': 1e-9, 'disp': False}
        )
        
        if result.success or result.fun < objective(x0):
            optimized = result.x
            positions = optimized[:2*n].reshape(n, 2)
            radii = optimized[2*n:]
            if np.all(constraints_func(optimized) >= -1e-8):
                return np.column_stack([positions, radii])
    except Exception:
        pass
    
    return circles


# EVOLVE-BLOCK-END
