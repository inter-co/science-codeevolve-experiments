# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

# EXPLORATION: Enhanced physics-based approach with hexagonal initialization
# Combines INSPIRATION 1's efficient hexagonal lattice with improved relaxation and optimization

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize sum of radii.
    
    Optimized multi-strategy approach with early stopping:
    1. Try up to 3 initialization patterns with early exit for excellent results
    2. Vectorized physics-based relaxation with momentum (reduced iterations)
    3. Aggressive initial radii (0.998 safety factor)
    4. Constraint-based SLSQP optimization with smart restarts
    5. Ultra-aggressive radius expansion (0.999 safety factor)
    6. Final polishing with micro-position adjustments
    7. Post-optimization incremental inflation (from INSPIRATION 1)
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 32
    
    # Multi-strategy: ordered by expected performance (hexagonal first)
    strategies = [
        initialize_hexagonal_grid(n),
        initialize_hexagonal_grid_variant(n, spacing=0.170, offset=0.09),
        initialize_4x8_grid(n),
    ]
    
    best_circles = None
    best_sum = 0
    target_threshold = 2.935  # 99.9% of benchmark - early stopping threshold
    
    for strategy_idx, initial_positions in enumerate(strategies):
        # Phase 1: Reduced physics relaxation iterations (300 vs 350)
        positions = physics_relaxation_with_momentum(initial_positions.copy(), iterations=300)
        
        # Phase 2: Aggressive initial radii
        radii = compute_maximum_radii(positions, safety_factor=0.998)
        circles = np.column_stack([positions, radii])
        
        # Phase 3: SLSQP optimization (reduced iterations for speed)
        circles = local_refinement(circles, iterations=180)
        
        # Phase 4: Smart restart (only for very promising solutions)
        if np.sum(circles[:, 2]) > 2.88:
            for trial in range(2):
                perturbed = circles.copy()
                scale = 0.007 if trial == 0 else 0.011
                perturbed[:, :2] += np.random.randn(n, 2) * scale
                perturbed[:, :2] = np.clip(perturbed[:, :2], 0.01, 0.99)
                optimized = local_refinement(perturbed, iterations=120)
                if is_valid_configuration(optimized) and np.sum(optimized[:, 2]) > np.sum(circles[:, 2]):
                    circles = optimized
        
        # Phase 5: Ultra-aggressive radius expansion
        circles = expand_radii_ultra_aggressive(circles, iterations=180)
        
        # Phase 6: Final polishing
        circles = final_polish(circles, iterations=150)
        
        # Phase 7: Post-optimization incremental inflation (from INSPIRATION 1)
        circles = inflate_radii(circles)
        
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_circles = circles
            best_sum = current_sum
        
        # Early stopping: if we've achieved excellent results, skip remaining strategies
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


def physics_relaxation_with_momentum(positions, iterations=300, dt=0.015):
    """
    Enhanced physics relaxation with momentum term.
    Optimized for faster convergence with fewer iterations.
    """
    pos = positions.copy()
    n = len(pos)
    velocities = np.zeros_like(pos)
    momentum = 0.7
    
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
        damping = 0.85 if iteration < 150 else 0.9
        velocities = momentum * velocities + (1 - momentum) * forces
        pos += dt * velocities
        pos = np.clip(pos, margin, 1 - margin)
        
        if iteration % 100 == 0 and iteration > 0:
            dt *= 0.95
    
    return pos


def compute_maximum_radii(positions, safety_factor=0.998):
    """
    Compute maximum radius per circle with aggressive safety factor.
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
    """4x8 grid pattern specifically optimized for 32 circles (from INSPIRATION 2)."""
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


def compute_max_radius_for_circle(pos, existing_circles):
    """Compute maximum feasible radius for a circle at given position."""
    x, y = pos
    max_r = min(x, 1-x, y, 1-y)
    
    if len(existing_circles) > 0:
        for cx, cy, cr in existing_circles:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_r = min(max_r, dist - cr)
    
    return max(0, max_r)


def expand_radii_ultra_aggressive(circles, iterations=180):
    """
    Ultra-aggressive radius expansion with optimized iterations.
    Uses 99.9% safety factor for maximum radii.
    """
    n = len(circles)
    
    for iteration in range(iterations):
        order = np.random.permutation(n)
        improved = False
        
        for i in order:
            existing = np.vstack([circles[:i], circles[i+1:]]) if i < n - 1 else circles[:i]
            max_r = compute_max_radius_for_circle(circles[i, :2], existing)
            new_r = max_r * 0.999
            
            if new_r > circles[i, 2]:
                circles[i, 2] = new_r
                improved = True
        
        if not improved:
            break
    
    return circles


def final_polish(circles, iterations=150):
    """
    Three-phase polish adapted from INSPIRATION 2's winning approach.
    Phase 1: Radius expansion with progressive safety
    Phase 2: Circular micro-adjustments (KEY INNOVATION - 13 angles)
    Phase 3: Final ultra-tight pass
    """
    n = len(circles)
    
    # Phase 1: Radius expansion with progressive safety (0.9985→0.9999)
    no_improvement_count = 0
    for iteration in range(100):
        order = np.random.permutation(n)
        improved = False
        
        # Progressive safety: 0.9985→0.9999
        safety = 0.9985 + 0.0014 * min(iteration / 100, 1.0)
        
        for i in order:
            existing = np.vstack([circles[:i], circles[i+1:]]) if i < n - 1 else circles[:i]
            max_r = compute_max_radius_for_circle(circles[i, :2], existing)
            new_r = max_r * safety
            
            if new_r > circles[i, 2] * 1.000015:
                circles[i, 2] = new_r
                improved = True
        
        # Enhanced early stopping
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= 2 and iteration > 30:
                break
        else:
            no_improvement_count = 0
    
    # Phase 2: Circular micro-adjustments (KEY INNOVATION from INSPIRATION 2)
    for micro_iter in range(40):
        improved = False
        micro_step = 0.004 * (1 - micro_iter / 40) ** 0.8
        
        for i in range(n):
            best_r = circles[i, 2]
            best_pos = circles[i, :2].copy()
            existing = np.vstack([circles[:i], circles[i+1:]]) if i < n - 1 else circles[:i]
            
            # Circular sampling with 13 angles (instead of 4 directions)
            for angle in np.linspace(0, 2*np.pi, 13)[:-1]:
                test_pos = circles[i, :2] + micro_step * np.array([np.cos(angle), np.sin(angle)])
                if 0.002 <= test_pos[0] <= 0.998 and 0.002 <= test_pos[1] <= 0.998:
                    max_r = compute_max_radius_for_circle(test_pos, existing)
                    
                    if max_r > best_r:
                        best_r = max_r * 0.9999
                        best_pos = test_pos
                        improved = True
            
            if improved:
                circles[i, :2] = best_pos
                circles[i, 2] = best_r
        
        if not improved and micro_iter > 18:
            break
    
    # Phase 3: Final ultra-tight pass
    for _ in range(20):
        for i in range(n):
            existing = np.vstack([circles[:i], circles[i+1:]]) if i < n - 1 else circles[:i]
            max_r = compute_max_radius_for_circle(circles[i, :2], existing)
            circles[i, 2] = max(circles[i, 2], max_r * 0.999995)
    
    return circles


def inflate_radii(circles):
    """
    Post-optimization incremental radius inflation (from INSPIRATION 1).
    Try to squeeze out extra radius after all optimization converges.
    """
    n = len(circles)
    best_circles = circles.copy()
    
    # Try inflating each radius with progressively smaller increments
    for i in range(n):
        for increment in [0.0008, 0.0004, 0.0002, 0.0001, 0.00005]:
            test_circles = best_circles.copy()
            test_circles[i, 2] += increment
            
            if is_valid_configuration(test_circles):
                best_circles = test_circles
            else:
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
    SLSQP optimization with ultra-tight tolerances.
    Optimized iteration count for speed while maintaining quality.
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
