# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and optimization.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a grid-based approach
    # Create a coarse grid to distribute initial circle positions
    grid_size = int(math.ceil(math.sqrt(n)))
    positions = []
    
    # Generate positions on a grid
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) >= n:
                break
            x = (i + 0.5) / grid_size
            y = (j + 0.5) / grid_size
            positions.append([x, y])
    
    # Adjust to exactly n positions
    positions = positions[:n]
    
    # Start with equal small radii
    radii = [0.02] * n
    
    # Combine positions and radii into one array for optimization
    initial_guess = []
    for i in range(n):
        initial_guess.extend([positions[i][0], positions[i][1], radii[i]])
    
    # Define constraint functions
    def contain_constraint(x, i):
        """Ensure circle i stays within the unit square"""
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        return min(xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri)
    
    def overlap_constraint(x, i, j):
        """Ensure circles i and j don't overlap"""
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
        distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
        return distance - ri - rj
    
    # Optimization objective (negative because we want to maximize sum of radii)
    def objective(x):
        return -sum(x[3*i+2] for i in range(n))
    
    # Constraints
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: contain_constraint(x, i)
        })
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: overlap_constraint(x, i, j)
            })
    
    # Bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # r <= 0.5 to prevent overly large circles
    
    # Perform optimization
    try:
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            final_circles = []
            for i in range(n):
                x, y, r = result.x[3*i], result.x[3*i+1], result.x[3*i+2]
                final_circles.append([x, y, r])
            return np.array(final_circles)
    except:
        pass
    
    # Fallback to a simpler heuristic approach if optimization fails
    # Use a more systematic approach: start with dense packing and relax
    circles = np.zeros((n, 3))
    
    # Place circles in a hexagonal pattern with varying radii
    # This is inspired by optimal packing principles
    max_radius = 0.15
    total_radius = 0
    
    # Try to place circles systematically
    for i in range(n):
        # Calculate ideal spacing based on number of circles
        rows = int(math.sqrt(n)) + 1
        cols = n // rows + 1
        
        row = i // cols
        col = i % cols
        
        # Distribute in a grid pattern with some randomness
        x = 0.1 + 0.8 * col / (cols - 1) if cols > 1 else 0.5
        y = 0.1 + 0.8 * row / (rows - 1) if rows > 1 else 0.5
        
        # Vary radii to maximize total sum
        r = max_radius * (1 - 0.8 * (i / n))  # Decreasing radii to allow more circles
        
        # Ensure it fits in the square
        r = min(r, x, 1-x, y, 1-y)
        
        circles[i] = [x, y, r]
        total_radius += r
    
    # Final refinement using local optimization for each circle
    for attempt in range(50):  # Multiple refinement attempts
        improved = False
        for i in range(n):
            best_x, best_y, best_r = circles[i]
            
            # Try small perturbations
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dr in [-0.005, -0.002, 0, 0.002, 0.005]:
                        new_x = best_x + dx
                        new_y = best_y + dy
                        new_r = best_r + dr
                        
                        # Check constraints
                        if (new_r > 0 and new_r <= new_x and new_r <= 1-new_x and 
                            new_r <= new_y and new_r <= 1-new_y):
                            
                            # Check overlap with others
                            valid = True
                            for j in range(n):
                                if i != j:
                                    dist = math.sqrt((new_x - circles[j][0])**2 + (new_y - circles[j][1])**2)
                                    if dist < new_r + circles[j][2]:
                                        valid = False
                                        break
                            
                            if valid:
                                if new_r > best_r:
                                    best_x, best_y, best_r = new_x, new_y, new_r
                                    improved = True
            
            circles[i] = [best_x, best_y, best_r]
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
