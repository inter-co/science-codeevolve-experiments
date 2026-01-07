# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

# Global constants
N_CIRCLES = 32
MAX_ITER = 1000

def initialize_circles() -> np.ndarray:
    """Initialize circles using a hexagonal lattice pattern for good starting configuration."""
    # Use a hexagonal lattice approach similar to INSPIRATION 1
    # For 32 circles, create a roughly 6x6 grid with adjustments
    rows = 6
    cols = 6
    
    # Hexagonal spacing factor
    spacing = 0.866
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N_CIRCLES:
                break
            # Hexagonal offset
            x = (j + (i % 2) * 0.5) * spacing * 0.15
            y = i * spacing * 0.15
            positions.append([x, y])
    
    # Normalize and fit within unit square
    if positions:
        positions = np.array(positions)
        max_x = np.max(positions[:, 0])
        max_y = np.max(positions[:, 1])
        
        if max_x > 0 and max_y > 0:
            scale_factor = min(0.9/max_x, 0.9/max_y)
            positions *= scale_factor
            
        # Center in unit square
        center_offset_x = 0.5 - np.mean(positions[:, 0])
        center_offset_y = 0.5 - np.mean(positions[:, 1])
        positions[:, 0] += center_offset_x
        positions[:, 1] += center_offset_y
        
        # Ensure all positions are within bounds
        positions[:, 0] = np.clip(positions[:, 0], 0, 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, 1)
    
    # Initialize with appropriate radii
    circles = np.zeros((N_CIRCLES, 3))
    for i in range(N_CIRCLES):
        x, y = positions[i]
        # Set initial radius based on proximity to boundaries
        r = min(x, 1-x, y, 1-y) * 0.8  # Leave some margin
        circles[i] = [x, y, max(0.01, r)]
    
    return circles

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration satisfies all constraints."""
    # Check containment
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps using efficient pairwise distance calculation
    centers = circles[:, :2]
    radii = circles[:, 2]
    
    # Compute all pairwise distances
    distances = cdist(centers, centers)
    
    # Check all pairs for overlap
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            dist = distances[i, j]
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                return False
    
    return True

def compute_total_radius(circles: np.ndarray) -> float:
    """Compute total sum of radii."""
    return np.sum(circles[:, 2])

def optimize_with_constraints(circles: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Use constrained optimization approach similar to INSPIRATION 2 but with better implementation."""
    n = len(circles)
    
    # Flatten for optimization
    initial_params = circles.flatten()
    
    # Objective function to minimize (negative of sum of radii)
    def objective(params):
        temp_circles = params.reshape((n, 3))
        return -np.sum(temp_circles[:, 2])
    
    # Constraints for boundary and overlap
    def constraint_func(params):
        temp_circles = params.reshape((n, 3))
        constraints = []
        
        # Boundary constraints: each circle must be fully within unit square
        for i in range(n):
            x, y, r = temp_circles[i]
            # Distance to boundaries (should be >= r)
            constraints.extend([
                x - r,           # left boundary
                1 - x - r,       # right boundary
                y - r,           # bottom boundary
                1 - y - r        # top boundary
            ])
        
        # Overlap constraints: distance between centers >= sum of radii
        centers = temp_circles[:, :2]
        radii = temp_circles[:, 2]
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                # Constraint should be positive when satisfied: dist - min_dist >= 0
                constraints.append(dist - min_dist)
        
        return np.array(constraints)
    
    # Bounds for parameters: x, y in [0,1], r in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])
    
    # Create constraint dictionary
    cons = {
        'type': 'ineq',
        'fun': lambda x: constraint_func(x)
    }
    
    # Optimization options
    options = {'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
    
    try:
        # Run optimization with SLSQP
        result = minimize(objective, initial_params, method='SLSQP', 
                         bounds=bounds, constraints=cons, options=options)
        
        if result.success:
            final_circles = result.x.reshape((n, 3))
            return final_circles
    except Exception:
        pass
    
    return circles

def local_improvement(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Perform local improvement to refine the solution."""
    current_circles = circles.copy()
    
    for iteration in range(max_iter):
        improved = False
        
        # Try to increase radii for each circle
        for i in range(len(current_circles)):
            original = current_circles[i].copy()
            old_radius = original[2]
            
            # Find maximum possible radius for this circle
            max_radius = min(
                original[0], 1 - original[0],
                original[1], 1 - original[1]
            )
            
            # Find minimum distance to other circles
            min_dist = float('inf')
            for j in range(len(current_circles)):
                if i != j:
                    dx = original[0] - current_circles[j, 0]
                    dy = original[1] - current_circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, distance)
            
            # Maximum possible radius considering overlap constraints
            if min_dist > 0:
                max_possible_radius = min_dist - 0.0001
                max_possible_radius = min(max_possible_radius, max_radius)
            else:
                max_possible_radius = max_radius
                
            if max_possible_radius > old_radius and max_possible_radius > 0:
                # Increase radius up to maximum allowed
                new_radius = min(max_possible_radius, old_radius + 0.01)
                current_circles[i, 2] = new_radius
                
                # Check validity
                if is_valid_configuration(current_circles):
                    improved = True
                else:
                    # Revert
                    current_circles[i] = original
            else:
                # Try moving circle to improve packing
                current_circles[i, 0] += np.random.normal(0, 0.001)
                current_circles[i, 1] += np.random.normal(0, 0.001)
                # Keep within bounds
                current_circles[i, 0] = np.clip(current_circles[i, 0], 0.001, 0.999)
                current_circles[i, 1] = np.clip(current_circles[i, 1], 0.001, 0.999)
                
                if is_valid_configuration(current_circles):
                    improved = True
                else:
                    # Revert
                    current_circles[i] = original
        
        if not improved:
            break
    
    return current_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Phase 1: Initialize with good starting configuration
    circles = initialize_circles()
    
    # Phase 2: Constrained optimization
    circles = optimize_with_constraints(circles, max_iter=300)
    
    # Phase 3: Local improvement
    circles = local_improvement(circles, max_iter=100)
    
    # Phase 4: Final refinement with more aggressive local search
    circles = local_improvement(circles, max_iter=50)
    
    # Final validation
    if not is_valid_configuration(circles):
        # Fallback to simpler approach if needed
        circles = initialize_circles()
    
    return circles


# EVOLVE-BLOCK-END
