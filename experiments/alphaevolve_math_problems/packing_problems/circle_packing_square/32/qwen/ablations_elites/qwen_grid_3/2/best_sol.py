# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time

# Global constants for optimization
CIRCLE_COUNT = 32

def initialize_hexagonal_placement(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal grid pattern for better initial placement"""
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern with better spacing
    rows = int(np.ceil(np.sqrt(n * 1.2)))  # Slightly denser than pure sqrt
    cols = int(np.ceil(n / rows))
    
    # Adjust grid size to fit exactly n circles
    spacing_x = 0.9 / cols
    spacing_y = 0.9 / rows
    offset_x = 0.05
    offset_y = 0.05
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset for even rows
            x_offset = 0.5 * (i % 2)
            x = offset_x + (j + x_offset) * spacing_x
            y = offset_y + i * spacing_y
            
            # Add more randomness to avoid perfect symmetry
            x += np.random.uniform(-spacing_x/8, spacing_x/8)
            y += np.random.uniform(-spacing_y/8, spacing_y/8)
            
            # Ensure circles stay within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius - start with a more informed value
            circles[idx] = [x, y, 0.025]
            idx += 1
        if idx >= n:
            break
            
    return circles

def penalty_method_objective(circles: np.ndarray, penalty_weight: float = 1000.0) -> float:
    """Objective function with penalty terms for constraint violations"""
    # Objective: maximize sum of radii
    objective_value = -np.sum(circles[:, 2])
    
    # Penalty for containment violations
    penalty = 0
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0:
            penalty += penalty_weight * (x - r)**2
        if x + r > 1:
            penalty += penalty_weight * (x + r - 1)**2
        if y - r < 0:
            penalty += penalty_weight * (y - r)**2
        if y + r > 1:
            penalty += penalty_weight * (y + r - 1)**2
    
    # Penalty for overlap violations
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                penalty += penalty_weight * (r1 + r2 - distance)**2
    
    return objective_value + penalty

def constraint_containment(circles: np.ndarray) -> np.ndarray:
    """Ensure all circles are contained within the unit square"""
    n = len(circles)
    constraints = []
    
    for i in range(n):
        x, y, r = circles[i]
        # r <= x <= 1-r and r <= y <= 1-r
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # x <= 1-r
            y - r,           # y >= r
            1 - y - r        # y <= 1-r
        ])
    
    return np.array(constraints)

def constraint_nonoverlap(circles: np.ndarray) -> np.ndarray:
    """Ensure no two circles overlap"""
    n = len(circles)
    constraints = []
    
    # Calculate pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # For each pair of circles, ensure distance >= sum of radii
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            min_distance = radii[i] + radii[j]
            
            # Constraint: distance >= min_distance (so we want distance - min_distance >= 0)
            constraints.append(distance - min_distance)
    
    return np.array(constraints)

def optimize_with_scipy(circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Use scipy optimization with proper constraints"""
    n = len(circles)
    
    # Flatten circles array for optimization
    initial_flat = circles.flatten()
    
    def objective(flat_params):
        # Reshape back to circles
        temp_circles = flat_params.reshape(-1, 3)
        return penalty_method_objective(temp_circles)
    
    # Define bounds for optimization (radius must be positive, positions bounded)
    bounds = []
    for i in range(n):
        # x coordinate bounds (r <= x <= 1-r)
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.499))  # r (max radius is 0.5)
    
    # Set up constraints properly
    def containment_constraint(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return constraint_containment(temp_circles)
    
    def nonoverlap_constraint(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return constraint_nonoverlap(temp_circles)
    
    cons = [
        {'type': 'ineq', 'fun': containment_constraint},
        {'type': 'ineq', 'fun': nonoverlap_constraint}
    ]
    
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        # Fallback to simpler approach if optimization fails
        pass
    
    # If optimization fails, return original
    return circles

def apply_repulsion_force(circles: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Apply physics-based repulsion to resolve overlaps with increased intensity"""
    new_circles = circles.copy()
    
    for _ in range(iterations):
        # Calculate forces between overlapping circles
        forces = np.zeros((len(new_circles), 2))  # Forces on centers
        
        # Check all pairs
        for i in range(len(new_circles)):
            for j in range(i+1, len(new_circles)):
                x1, y1, r1 = new_circles[i]
                x2, y2, r2 = new_circles[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If circles overlap, apply repulsive force
                if distance < (r1 + r2) and distance > 1e-10:
                    overlap = (r1 + r2) - distance
                    # Force magnitude proportional to overlap and inverse of distance - more aggressive
                    force_magnitude = overlap * 1.0
                    
                    # Normalize direction and apply force
                    if distance > 1e-8:  # Avoid division by zero
                        force = force_magnitude / distance
                        forces[i, 0] += force * dx
                        forces[i, 1] += force * dy
                        forces[j, 0] -= force * dx
                        forces[j, 1] -= force * dy
        
        # Apply forces to move circles with more aggressive update
        for i in range(len(new_circles)):
            # Update positions with forces
            new_circles[i, 0] += forces[i, 0] * 0.002
            new_circles[i, 1] += forces[i, 1] * 0.002
            
            # Ensure circles don't go out of bounds
            x, y, r = new_circles[i]
            new_circles[i, 0] = np.clip(x, r, 1 - r)
            new_circles[i, 1] = np.clip(y, r, 1 - r)
            
    return new_circles

def increase_radii_aggressive(circles: np.ndarray) -> np.ndarray:
    """Aggressively try to increase radii where beneficial"""
    new_circles = circles.copy()
    
    # Try to increase each radius more aggressively
    for i in range(len(new_circles)):
        x, y, r = new_circles[i]
        
        # Calculate maximum possible radius without violating constraints
        max_radius = min(x, 1-x, y, 1-y)
        
        # Check overlap with other circles
        for j in range(len(new_circles)):
            if i != j:
                x2, y2, r2 = new_circles[j]
                distance = np.sqrt((x-x2)**2 + (y-y2)**2)
                max_radius = min(max_radius, distance - r2 - 1e-8)
        
        # Increase radius more aggressively - try to push closer to maximum
        if max_radius > r and max_radius > 0.001:
            # Be more aggressive with radius increases
            new_radius = min(max_radius, r + (max_radius - r) * 0.3)
            new_circles[i, 2] = new_radius
            
    return new_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Strategy: Multiple aggressive optimization attempts
    best_circles = None
    best_sum = 0
    
    # Try multiple initialization strategies with more aggressive refinement
    for attempt in range(8):  # Increased attempts for better chance of finding better solution
        # Initialize with hexagonal pattern
        circles = initialize_hexagonal_placement(CIRCLE_COUNT)
        
        # Apply physics-based repulsion to resolve initial overlaps
        circles = apply_repulsion_force(circles, 50)  # More iterations for better cleanup
        
        # Aggressive radius increase
        circles = increase_radii_aggressive(circles)
        
        # Apply scipy optimization for fine-tuning with more iterations
        optimized_circles = optimize_with_scipy(circles, max_iter=1000)
        
        # Final aggressive refinement
        final_circles = apply_repulsion_force(optimized_circles, 30)
        
        # Even more aggressive radius increase after optimization
        final_circles = increase_radii_aggressive(final_circles)
        
        # Check if this is better
        current_sum = np.sum(final_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = final_circles.copy()
    
    # If no good solution found, fallback to a robust approach
    if best_circles is None:
        circles = initialize_hexagonal_placement(CIRCLE_COUNT)
        circles = apply_repulsion_force(circles, 100)
        circles = increase_radii_aggressive(circles)
        best_circles = optimize_with_scipy(circles, max_iter=1000)
    
    return best_circles


# EVOLVE-BLOCK-END
