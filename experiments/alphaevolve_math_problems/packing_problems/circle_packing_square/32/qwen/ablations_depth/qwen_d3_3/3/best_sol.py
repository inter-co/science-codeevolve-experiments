# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial geometric configuration followed by optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initial configuration: arrange circles in a hexagonal pattern
    def generate_hexagonal_layout():
        # Try to fit circles in a hexagonal grid pattern
        # Estimate how many rows/columns we need
        rows = int(math.sqrt(n))
        cols = int(math.ceil(n / rows))
        
        # Create a more efficient hexagonal layout
        circles = []
        radius_estimate = 0.1  # Start with reasonable estimate
        
        # Generate initial positions in a hexagonal pattern
        y_positions = []
        x_positions = []
        
        # Create staggered rows for hexagonal packing
        for i in range(rows):
            y = radius_estimate + i * 2 * radius_estimate * math.sqrt(3)/2
            if y > 1 - radius_estimate:
                break
            for j in range(cols):
                if i % 2 == 0:
                    x = radius_estimate + j * 2 * radius_estimate
                else:
                    x = radius_estimate + (j + 0.5) * 2 * radius_estimate
                if x <= 1 - radius_estimate:
                    x_positions.append(x)
                    y_positions.append(y)
        
        # If we don't have enough points, fill with additional positions
        while len(x_positions) < n:
            x_positions.append(0.5)
            y_positions.append(0.5)
            
        # Take first n positions
        x_positions = x_positions[:n]
        y_positions = y_positions[:n]
        
        # Initialize with equal small radii
        radii = [radius_estimate] * n
        
        return np.column_stack([x_positions[:n], y_positions[:n], radii])
    
    # Generate initial configuration
    circles = generate_hexagonal_layout()
    
    # Define constraint functions
    def get_distances(circles_array):
        """Calculate pairwise distances between circle centers"""
        centers = circles_array[:, :2]
        return cdist(centers, centers)
    
    def constraint_overlap(i, j, circles_array):
        """Constraint that circles i and j don't overlap"""
        ci, ri = circles_array[i, :2], circles_array[i, 2]
        cj, rj = circles_array[j, :2], circles_array[j, 2]
        dist = np.sqrt(np.sum((ci - cj)**2))
        return dist - (ri + rj)
    
    def constraint_bounds(circles_array):
        """Check if all circles are within bounds"""
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    # Objective function to maximize (negative because minimize)
    def objective(circles_flat):
        total_radius = np.sum(circles_flat[2::3])  # Sum of all radii
        return -total_radius  # Negative because we want to maximize
    
    # Constraints
    def constraint_function(circles_flat):
        # Convert flat array back to 2D array
        circles_array = circles_flat.reshape(-1, 3)
        
        # Check bounds
        bounds_violation = []
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            bounds_violation.extend([
                x - r,  # x - r >= 0
                1 - x - r,  # 1 - x - r >= 0
                y - r,  # y - r >= 0
                1 - y - r   # 1 - y - r >= 0
            ])
        
        # Check overlaps
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                ci, ri = circles_array[i, :2], circles_array[i, 2]
                cj, rj = circles_array[j, :2], circles_array[j, 2]
                dist = np.sqrt(np.sum((ci - cj)**2))
                overlap = dist - (ri + rj)
                bounds_violation.append(overlap)  # Should be >= 0
                
        return np.array(bounds_violation)
    
    # Flatten initial circles for optimization
    initial_flat = circles.flatten()
    
    # Set up bounds for optimization: x,y in [r, 1-r], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])  # x, y, r bounds
    
    # Use scipy's minimize with SLSQP method which handles constraints well
    try:
        # First try a simpler approach with bounds only
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_function},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
        else:
            # Fall back to simple greedy improvement
            optimized_circles = circles.copy()
            # Simple iterative improvement
            for _ in range(100):
                improved = False
                for i in range(n):
                    best_r = optimized_circles[i, 2]
                    best_x = optimized_circles[i, 0]
                    best_y = optimized_circles[i, 1]
                    
                    # Try small adjustments
                    for dx in [-0.01, 0, 0.01]:
                        for dy in [-0.01, 0, 0.01]:
                            for dr in [-0.005, 0, 0.005]:
                                new_x = max(0.001, min(0.999, optimized_circles[i, 0] + dx))
                                new_y = max(0.001, min(0.999, optimized_circles[i, 1] + dy))
                                new_r = max(0.001, min(0.5, optimized_circles[i, 2] + dr))
                                
                                # Check constraints
                                valid = True
                                if new_x - new_r < 0 or new_x + new_r > 1 or \
                                   new_y - new_r < 0 or new_y + new_r > 1:
                                    valid = False
                                
                                if valid:
                                    # Check overlap with others
                                    for j in range(n):
                                        if i != j:
                                            dist = np.sqrt((new_x - optimized_circles[j, 0])**2 + 
                                                          (new_y - optimized_circles[j, 1])**2)
                                            if dist < new_r + optimized_circles[j, 2]:
                                                valid = False
                                                break
                                
                                if valid:
                                    test_circles = optimized_circles.copy()
                                    test_circles[i] = [new_x, new_y, new_r]
                                    new_sum = np.sum(test_circles[:, 2])
                                    old_sum = np.sum(optimized_circles[:, 2])
                                    if new_sum > old_sum:
                                        optimized_circles[i] = [new_x, new_y, new_r]
                                        improved = True
                                        
                if not improved:
                    break
                    
    except Exception as e:
        # Fallback to simple hexagonal configuration
        optimized_circles = generate_hexagonal_layout()
    
    return optimized_circles


# EVOLVE-BLOCK-END
