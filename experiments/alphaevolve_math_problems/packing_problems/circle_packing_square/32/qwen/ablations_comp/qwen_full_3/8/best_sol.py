# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import math
from typing import Tuple

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _validate_circles(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False
    
    # Check overlap constraints
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Compute pairwise distances efficiently
    distances = cdist(positions, positions)
    
    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < radii[i] + radii[j]:
                return False
    
    return True

def _compute_radius_sum(circles: np.ndarray) -> float:
    """Compute sum of all radii"""
    return np.sum(circles[:, 2])

def _initialize_hexagonal(n: int) -> np.ndarray:
    """Create a good initial configuration using hexagonal packing"""
    circles = []
    # Hexagonal packing parameters
    rows = 6
    cols = 6
    spacing_x = 1.0 / cols
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            # Ensure we're within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Initial radius based on spacing
                r = min(spacing_x, spacing_y) / 4
                circles.append([x, y, r])
    
    # Fill remaining positions with random valid circles
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = np.random.uniform(0.01, 0.1)
        circles.append([x, y, r])
        
    return np.array(circles[:n])

def _initialize_focused_placement(n: int) -> np.ndarray:
    """Initialize circles with focused placement near center"""
    circles = []
    
    # Place some circles near the center with larger radii
    center_count = min(8, n)
    for i in range(center_count):
        angle = 2 * np.pi * i / center_count
        radius = 0.15 * random.uniform(0.7, 1.0)
        x = 0.5 + radius * np.cos(angle) * random.uniform(0.5, 1.0)
        y = 0.5 + radius * np.sin(angle) * random.uniform(0.5, 1.0)
        r = radius * random.uniform(0.8, 1.0)
        
        # Ensure within bounds
        r = min(r, x, 1-x, y, 1-y)
        circles.append([x, y, r])
    
    # Fill remaining with hexagonal approach for better distribution
    remaining = n - len(circles)
    if remaining > 0:
        hex_circles = _initialize_hexagonal(remaining)
        circles.extend(hex_circles.tolist())
    
    return np.array(circles)

def _compute_constraint_violations(circles: np.ndarray) -> Tuple[float, float]:
    """Compute total violation amounts for containment and overlap"""
    containment_violation = 0.0
    overlap_violation = 0.0
    
    n = len(circles)
    
    # Check containment violations
    for i in range(n):
        x, y, r = circles[i]
        if x < r:
            containment_violation += (r - x)
        if x > 1 - r:
            containment_violation += (x - (1 - r))
        if y < r:
            containment_violation += (r - y)
        if y > 1 - r:
            containment_violation += (y - (1 - r))
    
    # Check overlap violations
    positions = circles[:, :2]
    radii = circles[:, 2]
    distances = cdist(positions, positions)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < (radii[i] + radii[j]):
                overlap_violation += (radii[i] + radii[j] - dist)
    
    return containment_violation, overlap_violation

def _constraint_functions(params: np.ndarray, n: int):
    """Return constraint values for scipy optimization"""
    # Reconstruct circles from flattened parameters
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = params[3*i]
        circles[i, 1] = params[3*i+1]
        circles[i, 2] = params[3*i+2]
    
    # Containment constraints (should be >= 0)
    containment_constraints = []
    for i in range(n):
        x, y, r = circles[i]
        containment_constraints.extend([
            x - r,           # Left boundary
            1 - x - r,       # Right boundary  
            y - r,           # Bottom boundary
            1 - y - r        # Top boundary
        ])
    
    # Overlap constraints (should be >= 0)
    overlap_constraints = []
    positions = circles[:, :2]
    radii = circles[:, 2]
    distances = cdist(positions, positions)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            overlap_constraints.append(dist - (radii[i] + radii[j]))
    
    return np.array(containment_constraints + overlap_constraints)

def _objective_with_constraints(params: np.ndarray, n: int) -> float:
    """Objective function with explicit constraints like INSPIRATION 1"""
    # Reconstruct circles from flattened parameters
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = params[3*i]
        circles[i, 1] = params[3*i+1]
        circles[i, 2] = params[3*i+2]
    
    # Objective: maximize sum of radii (minimize negative sum)
    radius_sum = np.sum(circles[:, 2])
    
    return -radius_sum  # Negative because we minimize

def _optimize_with_scipy(circles: np.ndarray) -> np.ndarray:
    """Use scipy optimization for local refinement with explicit constraints"""
    n = len(circles)
    
    # Flatten parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = circles.flatten()
    
    # Define bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate
        bounds.append((0.001, 0.499))  # radius (max radius is 0.5 when placed at corner)
    
    # Define constraints for scipy optimization
    constraints = {
        'type': 'ineq',
        'fun': lambda x: _constraint_functions(x, n)
    }
    
    # Optimization using SLSQP which handles both bounds and constraints well
    try:
        result = minimize(
            _objective_with_constraints,
            initial_params,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-8, 'eps': 1e-8}  # More precise
        )
        
        if result.success:
            # Reconstruct circles from optimized parameters
            optimized_circles = circles.copy()
            for i in range(n):
                optimized_circles[i, 0] = result.x[3*i]
                optimized_circles[i, 1] = result.x[3*i+1]
                optimized_circles[i, 2] = result.x[3*i+2]
            return optimized_circles
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    return circles

def _local_optimization_step(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Improve configuration using local search similar to INSPIRATION 2"""
    current = circles.copy()
    best = circles.copy()
    best_radius = _compute_radius_sum(best)
    
    for iteration in range(max_iterations):
        # Try small perturbations to improve radii
        improved = False
        for i in range(len(current)):
            # Try to increase radius while maintaining constraints
            x, y, r = current[i]
            old_r = r
            
            # Try increasing radius by small amounts
            for delta in [0.001, 0.005, 0.01]:
                new_r = min(r + delta, 0.4)
                
                # Check if this change maintains validity
                temp_circles = current.copy()
                temp_circles[i, 2] = new_r
                
                # Check containment
                valid = True
                if (new_r > temp_circles[i, 0] or new_r > temp_circles[i, 1] or 
                    new_r > (1 - temp_circles[i, 0]) or new_r > (1 - temp_circles[i, 1])):
                    valid = False
                
                # Check overlaps with all other circles
                if valid:
                    for j in range(len(temp_circles)):
                        if i != j:
                            dx = temp_circles[i, 0] - temp_circles[j, 0]
                            dy = temp_circles[i, 1] - temp_circles[j, 1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            min_dist = new_r + temp_circles[j, 2]
                            if dist < min_dist:
                                valid = False
                                break
                
                if valid:
                    current[i, 2] = new_r
                    improved = True
        
        # If we made improvements, update best
        current_radius = _compute_radius_sum(current)
        if current_radius > best_radius:
            best = current.copy()
            best_radius = current_radius
        
        if not improved:
            break
                
    return best

def _physics_refinement(circles: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Refine using a simple physics-inspired approach like INSPIRATION 2"""
    current = circles.copy()
    
    for _ in range(iterations):
        # Apply forces to push circles apart and keep them in bounds
        forces = np.zeros_like(current[:, :2])
        
        # Compute repulsion forces between overlapping circles
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                x1, y1, r1 = current[i]
                x2, y2, r2 = current[j]
                
                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0 and dist < (r1 + r2):
                    # Repulsion force
                    force_magnitude = (r1 + r2 - dist) / (dist + 1e-8)
                    forces[i, 0] += force_magnitude * dx / dist
                    forces[i, 1] += force_magnitude * dy / dist
                    forces[j, 0] -= force_magnitude * dx / dist
                    forces[j, 1] -= force_magnitude * dy / dist
        
        # Apply boundary forces
        for i in range(len(current)):
            x, y, r = current[i]
            # Push away from boundaries
            forces[i, 0] += max(0, r - x) * 10  # Left boundary
            forces[i, 0] += max(0, 1 - r - x) * 10  # Right boundary
            forces[i, 1] += max(0, r - y) * 10  # Bottom boundary
            forces[i, 1] += max(0, 1 - r - y) * 10  # Top boundary
        
        # Update positions with forces
        step_size = 0.001
        for i in range(len(current)):
            current[i, 0] += forces[i, 0] * step_size
            current[i, 1] += forces[i, 1] * step_size
            
            # Keep within bounds
            current[i, 0] = max(r, min(1-r, current[i, 0]))
            current[i, 1] = max(r, min(1-r, current[i, 1]))
    
    return current

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    best_circles = None
    best_sum = 0
    
    # Multi-start approach with different initialization strategies
    initializers = [
        _initialize_hexagonal,
        _initialize_focused_placement
    ]
    
    # Use more iterations to explore better solutions
    for start_iter in range(15):  # Slightly more iterations for even better exploration
        # Choose random initializer
        initializer = random.choice(initializers)
        circles = initializer(n)
        
        # Apply physics-based refinement
        circles = _physics_refinement(circles, 35)  # Slightly increased iterations
        
        # Apply local optimization
        circles = _local_optimization_step(circles, 60)  # Slightly increased iterations
        
        # Then optimize with scipy for final refinement
        circles = _optimize_with_scipy(circles)
        
        # Validate and check quality
        if _validate_circles(circles):
            current_sum = _compute_radius_sum(circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
    
    # Final refinement passes with more aggressive optimization
    if best_circles is not None:
        # Try several more optimization passes with physics refinement
        for _ in range(4):  # One more pass for final tuning
            refined = _physics_refinement(best_circles, 25)
            refined = _local_optimization_step(refined, 35)
            refined = _optimize_with_scipy(refined)
            if _validate_circles(refined):
                refined_sum = _compute_radius_sum(refined)
                if refined_sum > best_sum:
                    best_sum = refined_sum
                    best_circles = refined.copy()
    
    # Ensure final validation
    if best_circles is None:
        # Fallback to hexagonal initialization with refinement
        best_circles = _initialize_hexagonal(n)
        best_circles = _physics_refinement(best_circles, 40)
        best_circles = _local_optimization_step(best_circles, 40)
        best_circles = _optimize_with_scipy(best_circles)
    
    return best_circles


# EVOLVE-BLOCK-END
