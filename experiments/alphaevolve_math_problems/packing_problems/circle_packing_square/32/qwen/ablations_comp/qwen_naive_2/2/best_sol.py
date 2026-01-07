# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using a more systematic approach
    # First create a denser grid and then use clustering to find good placement
    circles = np.zeros((n, 3))
    
    # Create a refined grid pattern for initial placement
    # Using a more sophisticated approach based on hexagonal packing principles
    rows = 6
    cols = 6
    grid_size = 1.0 / max(rows, cols)
    
    # Generate points in a grid with some randomness
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = (j + 0.5) * grid_size
            y = (i + 0.5) * grid_size
            # Add more significant random perturbation
            x += (np.random.random() - 0.5) * 0.1 * grid_size
            y += (np.random.random() - 0.5) * 0.1 * grid_size
            # Keep within bounds
            x = max(grid_size/2, min(1-grid_size/2, x))
            y = max(grid_size/2, min(1-grid_size/2, y))
            points.append([x, y])
        if len(points) >= n:
            break
    
    # Fill remaining positions if needed
    while len(points) < n:
        x = np.random.random() * 0.8 + 0.1  # Keep away from edges
        y = np.random.random() * 0.8 + 0.1
        points.append([x, y])
    
    # Initialize with larger initial radii
    for i in range(min(len(points), n)):
        circles[i] = [points[i][0], points[i][1], 0.05]
    
    # Use a more sophisticated optimization approach
    # Try to improve initial configuration first with a simpler method
    
    # Define constraint functions
    def radius_constraint(i):
        """Ensure circle i stays within bounds"""
        def constraint(x):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            # Circle must be fully inside unit square
            return min(xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri)
        return constraint
    
    def overlap_constraint(i, j):
        """Ensure circle i doesn't overlap with circle j"""
        def constraint(x):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            # Distance between centers minus radii should be >= 0
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            return dist - ri - rj
        return constraint
    
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    def objective(x):
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Create constraints
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': radius_constraint(i)})
    
    # Overlap constraints between all pairs
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
    
    # Bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        # x, y: [0, 1]
        # r: [0, 0.5] but we'll be more flexible for better results
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
    
    # Initial guess
    x0 = circles.flatten()
    
    # Try multiple optimization approaches
    best_result = None
    best_sum = -np.inf
    
    # Try with different optimization methods
    methods = ['SLSQP', 'trust-constr']
    
    for method in methods:
        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
                tol=1e-6
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception:
            continue
    
    # If no optimization worked, return initial configuration but with better radii
    if best_result is None:
        # Try a simple local optimization approach
        try:
            # Start with a better initialization - use the best known values
            circles = np.zeros((n, 3))
            # Place in a pattern that's known to work well
            positions = []
            # Generate positions in a way that maximizes space usage
            for i in range(n):
                # Use golden ratio spiral for even distribution
                phi = (3 - np.sqrt(5)) * i  # Golden angle
                r = np.sqrt(i) / np.sqrt(n-1)  # Radial distribution
                x = 0.5 + r * np.cos(phi) * 0.8
                y = 0.5 + r * np.sin(phi) * 0.8
                positions.append([x, y])
            
            # Initialize with appropriate radii
            for i in range(n):
                circles[i] = [positions[i][0], positions[i][1], 0.03]
                
            # Simple refinement - just try to increase radii while maintaining constraints
            x0 = circles.flatten()
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
                tol=1e-6
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                return optimized_circles
            else:
                return circles
                
        except Exception:
            return circles
    
    # Return best result found
    if best_result is not None:
        optimized_circles = best_result.x.reshape(-1, 3)
        return optimized_circles
    else:
        return circles


# EVOLVE-BLOCK-END
