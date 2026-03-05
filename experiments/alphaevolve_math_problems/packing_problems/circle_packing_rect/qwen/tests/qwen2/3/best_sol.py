# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining hexagonal grid initialization and robust optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Test multiple rectangle aspect ratios to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Try different width/height ratios - using values that typically work well
    ratios = [0.8, 1.0, 1.2, 1.5, 2.0]
    
    # Multi-start approach with different initial configurations (like INSPIRATION 1)
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)  # width + height = 2
        height = 2 / (1 + ratio)
        
        # Try multiple random starts for each configuration
        for start_run in range(10):  # Increased from 8 to 10 for better exploration
            try:
                # Create initial configuration using hexagonal grid approach (like INSPIRATION 1)
                circles = initialize_hexagonal_packing(width, height, 21, start_run)
                
                # Refine using optimization (like INSPIRATION 1)
                optimized_circles = optimize_positions(circles, width, height)
                
                # Evaluate and keep the best result
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles.copy()
                    
            except Exception:
                continue
    
    # Fallback to a default configuration if nothing worked
    if best_result is None:
        width, height = 1.0, 1.0
        best_result = initialize_hexagonal_packing(width, height, 21, 0)
        best_result = optimize_positions(best_result, width, height)
    
    return best_result


def initialize_hexagonal_packing(width: float, height: float, n: int, seed: int) -> np.ndarray:
    """Initialize circle positions using hexagonal packing pattern for better initial distribution."""
    np.random.seed(seed)
    
    # Create hexagonal grid pattern
    circles = []
    
    # Estimate appropriate hexagon radius based on available area
    area_available = width * height
    # Target filling about 65% of the area (more efficient packing)
    target_area = 0.65 * area_available
    avg_radius_squared = target_area / (n * np.pi)
    avg_radius = np.sqrt(avg_radius_squared)
    
    # Determine grid size for hexagonal packing
    rows = max(3, int(np.ceil(np.sqrt(n * 1.2))))
    cols = max(3, int(np.ceil(n / rows)))
    
    # Hexagonal spacing (sqrt(3) ~ 1.732)
    hex_width = avg_radius * 2
    hex_height = avg_radius * 1.732
    
    # Generate hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            
            # Hexagonal offset pattern
            x = (j + (i % 2) * 0.5) * hex_width + avg_radius
            y = i * hex_height + avg_radius
            
            # Add some randomness to avoid perfect symmetry
            x += np.random.uniform(-avg_radius/4, avg_radius/4)
            y += np.random.uniform(-avg_radius/4, avg_radius/4)
            
            # Keep within bounds
            if (avg_radius <= x <= width - avg_radius and 
                avg_radius <= y <= height - avg_radius):
                circles.append([x, y, avg_radius])
    
    # Fill remaining slots if needed
    while len(circles) < n:
        x = np.random.uniform(avg_radius, width - avg_radius)
        y = np.random.uniform(avg_radius, height - avg_radius)
        circles.append([x, y, avg_radius])
    
    return np.array(circles[:n])


def optimize_positions(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circle positions using robust scipy optimization."""
    n = initial_circles.shape[0]
    
    # Flatten initial configuration
    initial_flat = initial_circles.flatten()
    
    # Objective function: negative sum of radii (we want to maximize sum)
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Constraint functions with better handling
    def constraint_distance(params):
        """Ensure no two circles overlap"""
        circles = params.reshape(-1, 3)
        distances = cdist(circles[:, :2], circles[:, :2])
        constraints = []
        
        # Only check pairs where i < j to avoid double counting
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radii_sum = circles[i, 2] + circles[j, 2]
                # Distance should be >= sum of radii (with small tolerance for numerical issues)
                constraints.append(dist - radii_sum + 1e-8)
        return np.array(constraints)
    
    def constraint_bounds(params):
        """Ensure all circles stay within rectangle bounds"""
        circles = params.reshape(-1, 3)
        constraints = []
        
        for i in range(n):
            x, y, r = circles[i]
            # Circle center must be at least radius away from edges
            constraints.extend([
                x - r,           # left bound
                width - x - r,   # right bound
                y - r,           # bottom bound
                height - y - r   # top bound
            ])
        return np.array(constraints)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_distance},
        {'type': 'ineq', 'fun': constraint_bounds}
    ]
    
    # Set bounds for each parameter (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0, width), (0, height), (1e-6, min(width, height)/2)])
    
    # Try multiple optimization methods with different settings (like INSPIRATION 1)
    methods = ['SLSQP', 'trust-constr']
    best_result = None
    best_sum = -float('inf')
    
    # Try with stricter tolerances first
    for method in methods:
        try:
            result = minimize(objective, initial_flat, method=method, 
                             bounds=bounds, constraints=cons, 
                             options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles
                    
        except Exception:
            continue
    
    # If strict optimization failed, try with looser tolerances
    if best_result is None:
        try:
            result = minimize(objective, initial_flat, method='SLSQP', 
                             bounds=bounds, constraints=cons, 
                             options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles
        except Exception:
            pass
    
    # If still no good result, return initial configuration
    return best_result if best_result is not None else initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
