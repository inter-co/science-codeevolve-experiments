# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import distance_matrix
from numba import jit

# EXPLORATION: Multi-scale SQP optimization with analytical gradients
# This approach formulates circle packing as a continuous constrained optimization
# problem, using Sequential Quadratic Programming with exact gradient computation
# and adaptive penalty methods for constraint handling.

@jit(nopython=True)
def compute_overlap_violations(positions, radii):
    """Compute all pairwise overlap violations using numba for speed."""
    n = len(positions)
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = (radii[i] + radii[j]) - dist
            if violation > 0:
                violations.append(violation)
    return violations

def check_feasibility(positions, radii, tol=1e-6):
    """Check if configuration is feasible (no overlaps, within bounds)."""
    n = len(positions)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    
    # Check non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True

def objective_and_gradient(x, n):
    """
    Objective: maximize sum of radii = minimize -sum(radii)
    x = [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    radii = x[2::3]
    obj = -np.sum(radii)  # Negative because we minimize
    
    # Gradient: d(-sum(r))/dx = [0, 0, -1, 0, 0, -1, ...]
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0
    
    return obj, grad

def constraint_violations_and_jacobian(x, n):
    """
    Compute all constraint violations and their Jacobian.
    Returns violations as a vector and Jacobian matrix.
    """
    positions = x.reshape(n, 3)[:, :2]
    radii = x[2::3]
    
    violations = []
    jacobian_rows = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        
        # x - r >= 0  =>  violation = max(0, r - x)
        v1 = ri - xi
        violations.append(v1)
        jac1 = np.zeros(3 * n)
        jac1[3*i] = -1.0  # d/dx
        jac1[3*i+2] = 1.0  # d/dr
        jacobian_rows.append(jac1)
        
        # x + r <= 1  =>  violation = max(0, x + r - 1)
        v2 = xi + ri - 1.0
        violations.append(v2)
        jac2 = np.zeros(3 * n)
        jac2[3*i] = 1.0
        jac2[3*i+2] = 1.0
        jacobian_rows.append(jac2)
        
        # y - r >= 0
        v3 = ri - yi
        violations.append(v3)
        jac3 = np.zeros(3 * n)
        jac3[3*i+1] = -1.0
        jac3[3*i+2] = 1.0
        jacobian_rows.append(jac3)
        
        # y + r <= 1
        v4 = yi + ri - 1.0
        violations.append(v4)
        jac4 = np.zeros(3 * n)
        jac4[3*i+1] = 1.0
        jac4[3*i+2] = 1.0
        jacobian_rows.append(jac4)
    
    # Non-overlap constraints: distance >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy + 1e-10)  # Add small epsilon for stability
            
            # violation = (ri + rj) - dist
            v = ri + rj - dist
            violations.append(v)
            
            # Jacobian: d(violation)/d(xi, yi, ri, xj, yj, rj)
            jac = np.zeros(3 * n)
            jac[3*i] = dx / dist      # d/dxi
            jac[3*i+1] = dy / dist    # d/dyi
            jac[3*i+2] = 1.0          # d/dri
            jac[3*j] = -dx / dist     # d/dxj
            jac[3*j+1] = -dy / dist   # d/dyj
            jac[3*j+2] = 1.0          # d/drj
            jacobian_rows.append(jac)
    
    return np.array(violations), np.array(jacobian_rows)

def penalty_objective(x, n, penalty_coeff):
    """Objective with penalty for constraint violations."""
    obj, grad_obj = objective_and_gradient(x, n)
    violations, jac_violations = constraint_violations_and_jacobian(x, n)
    
    # Penalty: sum of squared positive violations
    positive_violations = np.maximum(0, violations)
    penalty = penalty_coeff * np.sum(positive_violations ** 2)
    
    # Gradient of penalty
    grad_penalty = np.zeros(3 * n)
    for i, v in enumerate(violations):
        if v > 0:
            grad_penalty += 2 * penalty_coeff * v * jac_violations[i]
    
    return obj + penalty, grad_obj + grad_penalty

def optimize_positions(circles, iterations=50, initial_dt=0.01):
    """Physics-based position optimization with adaptive damping (from INSPIRATION)."""
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
                    force_mag = (min_dist - dist) * (20.0 if dist < min_dist else 3.0)
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
                positions[i, dim] = np.clip(positions[i, dim], radii[i] + 0.001, 1 - radii[i] - 0.001)
                if positions[i, dim] == radii[i] + 0.001 or positions[i, dim] == 1 - radii[i] - 0.001:
                    velocities[i, dim] = 0
        
        if np.max(np.abs(velocities)) < 1e-7:
            break
    
    circles[:, :2] = positions
    return circles

def compute_max_safe_radius(pos, all_circles, idx, margin=0.0):
    """Compute maximum feasible radius without constraint violations (from INSPIRATION)."""
    x, y = pos
    max_r = min(x, y, 1-x, 1-y) - margin
    
    for i in range(len(all_circles)):
        if i == idx:
            continue
        ox, oy, or_ = all_circles[i]
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        max_r = min(max_r, dist - or_ - margin)
    
    return max(0.0, max_r)

def expand_radii_binary_search(circles, steps=150, expansion_factor=0.6):
    """Greedy radius expansion with binary search for precision (from INSPIRATION)."""
    from scipy.spatial import KDTree
    n = len(circles)
    
    for step in range(steps):
        potentials = []
        tree = KDTree(circles[:, :2])
        
        for i in range(n):
            x, y = circles[i, :2]
            max_r_boundary = min(x, 1-x, y, 1-y)
            
            search_radius = min(2 * max_r_boundary, 0.45)
            neighbors = tree.query_ball_point([x, y], r=search_radius)
            neighbors = [j for j in neighbors if j != i]
            
            if len(neighbors) > 0:
                max_r_neighbors = float('inf')
                for j in neighbors:
                    dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                    max_r_neighbors = min(max_r_neighbors, dist - circles[j, 2])
                
                max_r = min(max_r_boundary, max_r_neighbors) - 0.0001
                
                if max_r > circles[i, 2] * 1.00005:
                    low, high = circles[i, 2], max_r
                    best_r = circles[i, 2]
                    
                    for _ in range(20):
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
                max_r = compute_max_safe_radius(circles[i, :2], circles, i, margin=0.0001)
                potential = max_r - circles[i, 2]
            
            dist_to_boundary = min(x, 1-x, y, 1-y)
            boundary_bonus = 1.3 if dist_to_boundary < 0.15 else 1.0
            weighted_potential = potential * boundary_bonus
            potentials.append((i, potential, weighted_potential))
        
        potentials.sort(key=lambda x: x[2], reverse=True)
        
        improved = False
        for i, potential, _ in potentials:
            if potential > 0.00003:
                x, y = circles[i, :2]
                dist_to_boundary = min(x, 1-x, y, 1-y)
                location_multiplier = 1.2 if dist_to_boundary < 0.15 else 1.0
                
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

def boundary_refinement(circles, iterations=30):
    """Aggressive expansion for boundary circles (from INSPIRATION)."""
    n = len(circles)
    for _ in range(iterations):
        improved = False
        
        for i in range(n):
            x, y = circles[i, :2]
            dist_to_boundary = min(x, 1-x, y, 1-y)
            
            if dist_to_boundary < 0.20:
                max_r = compute_max_safe_radius(circles[i, :2], circles, i, margin=0.0)
                
                if max_r > circles[i, 2]:
                    circles[i, 2] = circles[i, 2] + (max_r - circles[i, 2]) * 0.65
                    improved = True
        
        if not improved:
            break
    
    return circles

def final_optimization_slsqp(circles, maxiter=200):
    """Enhanced SLSQP with analytical gradients (inspired by INSPIRATION programs)."""
    n = len(circles)
    x0 = circles.flatten()
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.35))  # r - increased upper bound for better solutions
    
    def obj_func(x):
        radii = x[2::3]
        return -np.sum(radii)
    
    def obj_grad(x):
        grad = np.zeros(3 * n)
        grad[2::3] = -1.0
        return grad
    
    constraints = []
    
    # Boundary constraints with analytical Jacobians
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
    
    # Non-overlap constraints with analytical Jacobians
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

def local_perturbation_search(circles, num_perturbations=2):
    """Local perturbation to escape local optima (from INSPIRATION 1)."""
    n = len(circles)
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    for _ in range(num_perturbations):
        perturbed = circles.copy()
        indices = np.random.choice(n, size=min(6, n), replace=False)
        
        for i in indices:
            max_shift = 0.015
            perturbed[i, 0] += np.random.uniform(-max_shift, max_shift)
            perturbed[i, 1] += np.random.uniform(-max_shift, max_shift)
            
            perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2] + 0.001, 1 - perturbed[i, 2] - 0.001)
            perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2] + 0.001, 1 - perturbed[i, 2] - 0.001)
        
        perturbed = optimize_positions(perturbed, iterations=25, initial_dt=0.006)
        perturbed = expand_radii_binary_search(perturbed, steps=60, expansion_factor=0.35)
        
        if np.sum(perturbed[:, 2]) > best_sum:
            best_sum = np.sum(perturbed[:, 2])
            best_circles = perturbed.copy()
    
    return best_circles

def coordinate_wise_radius_maximization(circles, passes=2):
    """Coordinate-wise radius maximization with binary search (from INSPIRATION 2)."""
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
    """Aggressive two-stage scaling with relaxed tolerance (from INSPIRATION 2)."""
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
    
    # Stage 1: Coarse scaling with relaxed tolerance
    for scale in np.linspace(1.0, 1.20, 60):
        scaled = circles.copy()
        scaled[:, 2] *= scale
        if constraint_check(scaled, tol=-1e-8):
            best_circles = scaled
        else:
            break
    
    # Stage 2: Fine-grained refinement
    for delta in np.linspace(0, 0.025, 100):
        scaled = best_circles.copy()
        scaled[:, 2] *= (1.0 + delta)
        if constraint_check(scaled, tol=-1e-9):
            best_circles = scaled
        else:
            break
    
    return best_circles

def halton_sequence(n, base):
    """Generate n points of Halton sequence with given base (from INSPIRATION 1)."""
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

def initialize_configuration(n, config_id=0, seed=42, spacing=0.148):
    """Enhanced initialization with parameterized hexagonal spacing (from INSPIRATION 2)."""
    circles = np.zeros((n, 3))
    
    if config_id == 0:
        # Hexagonal close-packing with configurable spacing (CRITICAL from INSPIRATION 2)
        np.random.seed(seed)
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
                
                # Boundary-aware radius initialization
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
        # Corner-priority initialization (kept from target)
        positions = []
        margin = 0.12
        positions.extend([
            (margin, margin, 0.10),
            (1-margin, margin, 0.10),
            (margin, 1-margin, 0.10),
            (1-margin, 1-margin, 0.10)
        ])
        
        for i in range(3):
            t = 0.25 + i * 0.25
            positions.extend([
                (t, 0.09, 0.08), (t, 1-0.09, 0.08),
                (0.09, t, 0.08), (1-0.09, t, 0.08)
            ])
        
        for i in range(4):
            for j in range(4):
                x = 0.24 + j * 0.17
                y = 0.24 + i * 0.17
                positions.append((x, y, 0.07))
        
        for idx, (x, y, r) in enumerate(positions[:n]):
            circles[idx] = [x, y, r]
    
    elif config_id == 2:
        # Enhanced Halton with better spacing
        np.random.seed(seed)
        x_init = halton_sequence(n, 2) * 0.84 + 0.08
        y_init = halton_sequence(n, 3) * 0.84 + 0.08
        circles[:, 0] = x_init
        circles[:, 1] = y_init
        circles[:, 2] = 0.038  # Slightly larger initial radius
    
    elif config_id == 3:
        # Vogel spiral (kept from target)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            theta = i * golden_angle
            r = np.sqrt(i / n) * 0.4 + 0.1
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            circles[i] = [np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95), 0.028]
    
    elif config_id == 4:
        # Fibonacci lattice (kept from target)
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n):
            y = (i / (n - 1)) * 0.9 + 0.05
            x = ((i * phi) % 1) * 0.9 + 0.05
            circles[i] = [x, y, 0.028]
    
    # Ensure feasibility
    for i in range(n):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2] + 0.001, 1 - circles[i, 2] - 0.001)
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2] + 0.001, 1 - circles[i, 2] - 0.001)
    
    return circles

def multi_start_optimization(n):
    """Enhanced multi-start with fine-grained hexagonal spacing exploration (from INSPIRATION 2)."""
    import time
    best_circles = None
    best_sum = 0
    benchmark = 2.937944526205518
    quality_threshold = benchmark * 1.0001  # Target: beat benchmark
    
    np.random.seed(42)
    start_time = time.time()
    time_limit = 54.0  # Increased from 52s since target finishes in 4.3s
    
    # Enhanced configuration portfolio - 5 hexagonal spacings (inspired by INSPIRATION's 9)
    configurations = []
    
    # Strategy 1: Five carefully selected hexagonal spacings
    # Expanded from 3 to 5 based on INSPIRATION's success with diverse spacings
    for spacing in [0.145, 0.147, 0.148, 0.150, 0.152]:
        configurations.append((0, 42, f'hex_{spacing}', spacing))
    
    # Strategy 2: Alternative high-performing strategies
    configurations.extend([
        (2, 42, 'halton', 0.148),
        (1, 42, 'corner', 0.148),
    ])
    
    for config_idx, config_data in enumerate(configurations):
        if time.time() - start_time > time_limit:
            break
        
        if len(config_data) == 4:
            config_id, seed, strategy_name, spacing = config_data
            circles = initialize_configuration(n, config_id, seed, spacing)
        else:
            config_id, seed, strategy_name = config_data
            circles = initialize_configuration(n, config_id, seed)
        
        # Optimized pipeline with increased SLSQP for hexagonal (inspired by INSPIRATION's 300)
        if 'halton' in strategy_name:
            circles = optimize_positions(circles, iterations=85, initial_dt=0.015)
            circles = expand_radii_binary_search(circles, steps=210, expansion_factor=0.82)
            slsqp_maxiter = 220
        elif 'hex' in strategy_name:
            # Hexagonal: more SLSQP iterations for better convergence (inspired by INSPIRATION)
            circles = optimize_positions(circles, iterations=65, initial_dt=0.012)
            circles = expand_radii_binary_search(circles, steps=190, expansion_factor=0.79)
            slsqp_maxiter = 260  # Increased from 220, compromise between target's 220 and INSPIRATION's 300
        else:
            circles = optimize_positions(circles, iterations=70, initial_dt=0.013)
            circles = expand_radii_binary_search(circles, steps=175, expansion_factor=0.76)
            slsqp_maxiter = 200
        
        circles = optimize_positions(circles, iterations=60, initial_dt=0.009)
        circles = expand_radii_binary_search(circles, steps=155, expansion_factor=0.63)
        circles = boundary_refinement(circles, iterations=40)
        
        circles = optimize_positions(circles, iterations=40, initial_dt=0.006)
        circles = expand_radii_binary_search(circles, steps=125, expansion_factor=0.43)
        
        # Three SLSQP passes for hexagonal (inspired by INSPIRATION's multi-pass approach)
        num_slsqp_passes = 3 if 'hex' in strategy_name else 2
        for refinement_pass in range(num_slsqp_passes):
            try:
                optimized = final_optimization_slsqp(circles, maxiter=slsqp_maxiter)
                if check_feasibility(optimized[:, :2], optimized[:, 2]):
                    circles = optimized
            except:
                pass
        
        # Streamlined post-SLSQP refinement
        circles = expand_radii_binary_search(circles, steps=65, expansion_factor=0.21)
        circles = boundary_refinement(circles, iterations=28)
        circles = expand_radii_binary_search(circles, steps=40, expansion_factor=0.13)
        
        # Single pass coordinate-wise maximization
        circles = coordinate_wise_radius_maximization(circles, passes=1)
        
        # Aggressive scaling
        optimized = try_uniform_scaling(circles)
        
        # Final precision adjustments with minimal safety margin
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
    
    # Enhanced final refinement with additional SLSQP pass (inspired by INSPIRATION)
    if best_circles is not None and time.time() - start_time < 57.0:
        # Additional SLSQP refinement on best solution (from INSPIRATION's multi-pass approach)
        try:
            optimized = final_optimization_slsqp(best_circles, maxiter=200)
            if check_feasibility(optimized[:, :2], optimized[:, 2]):
                if np.sum(optimized[:, 2]) > np.sum(best_circles[:, 2]):
                    best_circles = optimized
        except:
            pass
        
        best_circles = coordinate_wise_radius_maximization(best_circles, passes=1)
        best_circles = expand_radii_binary_search(best_circles, steps=50, expansion_factor=0.10)
        best_circles = boundary_refinement(best_circles, iterations=20)
        best_circles = try_uniform_scaling(best_circles)
        
        # Final coordinate-wise pass (from INSPIRATION 2)
        best_circles = coordinate_wise_radius_maximization(best_circles, passes=1)
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses hybrid optimization combining physics-based relaxation, greedy expansion,
    and gradient-based refinement (synthesized from INSPIRATION programs).

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    try:
        circles = multi_start_optimization(n)
        
        if circles is None or not check_feasibility(circles[:, :2], circles[:, 2]):
            circles = initialize_configuration(n, config_id=0)
        
        return circles
    
    except Exception as e:
        circles = initialize_configuration(n, config_id=0)
        return circles


# EVOLVE-BLOCK-END
