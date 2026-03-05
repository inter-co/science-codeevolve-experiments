# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric intuition with mathematical optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Try multiple rectangle dimensions to find optimal configuration
    # Different aspect ratios might yield better results
    configs = [
        (1.3, 0.7),  # Original ratio
        (1.2, 0.8),  # More square-like
        (1.4, 0.6),  # More elongated
        (1.0, 1.0),  # Square
        (1.5, 0.5),  # Very elongated
        (0.8, 1.2),  # Reverse ratio
        (0.6, 1.4),  # More elongated vertically
        (0.9, 1.1),  # Nearly square
        (1.1, 0.9),  # Nearly square reverse
    ]
    
    best_result = None
    best_sum = -float('inf')
    
    for width, height in configs:
        # Validate container dimensions
        if width + height != 2:
            height = 2 - width
        
        # Generate initial configuration
        circles = generate_initial_configuration(width, height, 21)
        
        # Optimize using mathematical programming approach
        optimized_circles = optimize_with_mathematical_programming(circles, width, height)
        
        # Check if this is better
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized_circles
    
    return best_result

def generate_initial_configuration(width: float, height: float, n: int) -> np.ndarray:
    """Generate initial circle configuration using improved hexagonal packing"""
    circles = np.zeros((n, 3))
    
    # Estimate initial radius based on area
    container_area = width * height
    # Use 0.85 packing efficiency for hexagonal close packing
    total_circle_area = container_area * 0.85
    avg_radius = np.sqrt(total_circle_area / (np.pi * n))
    
    # Create hexagonal grid pattern
    spacing_x = 2 * avg_radius
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    # Determine grid dimensions
    cols = max(1, int(width / spacing_x) + 1)
    rows = max(1, int(height / spacing_y) + 1)
    
    # Generate positions in hexagonal pattern
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            
            # Offset odd rows for hexagonal packing
            x_offset = spacing_x / 2 if row % 2 == 1 else 0
            x = x_offset + col * spacing_x + avg_radius
            y = row * spacing_y + avg_radius
            
            # Ensure circle stays within bounds
            if (x - avg_radius >= 0 and x + avg_radius <= width and 
                y - avg_radius >= 0 and y + avg_radius <= height):
                circles[idx] = [x, y, avg_radius]
                idx += 1
                
        if idx >= n:
            break
    
    # Fill remaining positions with circles placed near edges
    while idx < n:
        # Try placing near corners or edges
        if idx < n:
            # Place near bottom-left corner
            x = avg_radius * 1.5
            y = avg_radius * 1.5
            circles[idx] = [x, y, avg_radius * 0.7]
            idx += 1
        if idx < n:
            # Place near top-right corner  
            x = width - avg_radius * 1.5
            y = height - avg_radius * 1.5
            circles[idx] = [x, y, avg_radius * 0.7]
            idx += 1
        if idx < n:
            # Place near center
            x = width / 2
            y = height / 2
            circles[idx] = [x, y, avg_radius * 0.6]
            idx += 1
        if idx < n:
            # Place randomly near edges
            x = np.random.uniform(avg_radius, width - avg_radius)
            y = np.random.uniform(avg_radius, height - avg_radius)
            circles[idx] = [x, y, avg_radius * 0.5]
            idx += 1
    
    return circles

def optimize_with_mathematical_programming(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize using scipy's minimize with proper constraints"""
    
    # Flatten initial circles for optimization
    x0 = initial_circles.flatten()
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (every third element starting from index 2)
    
    # Constraint functions for boundary conditions
    def boundary_constraint(x):
        """Ensure all circles stay within rectangle"""
        constraints = []
        for i in range(21):
            x_pos = x[3*i]
            y_pos = x[3*i+1]
            radius = x[3*i+2]
            
            # Left boundary
            constraints.append(x_pos - radius)
            # Right boundary  
            constraints.append(width - x_pos - radius)
            # Bottom boundary
            constraints.append(y_pos - radius)
            # Top boundary
            constraints.append(height - y_pos - radius)
        return np.array(constraints)
    
    # Constraint function for circle-to-circle distance
    def distance_constraint(x):
        """Ensure minimum distance between circle centers"""
        constraints = []
        for i in range(21):
            for j in range(i+1, 21):
                # Extract positions and radii
                pos_i = x[3*i:3*i+2]
                pos_j = x[3*j:3*j+2]
                r_i = x[3*i+2]
                r_j = x[3*j+2]
                
                dist = np.linalg.norm(pos_i - pos_j)
                constraints.append(dist - (r_i + r_j))
        return np.array(constraints)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': boundary_constraint},
        {'type': 'ineq', 'fun': distance_constraint}
    ]
    
    # Set bounds for optimization - more reasonable bounds (as used in successful programs)
    bounds = []
    for i in range(21):
        bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, 0.5)])
    
    # Perform optimization with multiple attempts to improve results
    best_result = None
    best_sum = -float('inf')
    
    # Try multiple optimization runs with different initializations
    # Use exactly 3 attempts like the best performers
    for attempt in range(3):
        try:
            # Small random perturbation to initial guess - even more careful
            perturbed_x0 = x0.copy()
            for i in range(len(perturbed_x0)):
                if i % 3 == 2:  # Radius component
                    # Perturb radius with a smaller range
                    perturbed_x0[i] *= np.random.uniform(0.995, 1.005)
                else:  # Position components
                    # Perturb position with even smaller range
                    perturbed_x0[i] += np.random.uniform(-0.002, 0.002)
            
            # Use only SLSQP method for consistency with successful approaches
            # Match parameters from top performer (Inspiration Program 3) more closely
            result = minimize(objective, perturbed_x0, method='SLSQP', bounds=bounds, constraints=cons, 
                            options={'maxiter': 2000, 'ftol': 1e-7, 'gtol': 1e-7})
            
            if result.success:
                current_sum = -result.fun  # Convert back to positive sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception:
            continue
    
    # Return the best result found or initial configuration if optimization failed
    if best_result is not None and best_result.success:
        optimized_circles = best_result.x.reshape(-1, 3)
        return optimized_circles
    else:
        return initial_circles.reshape(-1, 3)


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
