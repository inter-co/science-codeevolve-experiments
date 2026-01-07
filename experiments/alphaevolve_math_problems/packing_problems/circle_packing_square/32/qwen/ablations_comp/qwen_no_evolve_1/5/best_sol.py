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
    Uses a hybrid approach combining grid-based initialization and optimization.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Stage 1: Grid-based initialization for good starting configuration
    # Create a coarse grid to distribute initial circle centers
    grid_size = int(math.ceil(math.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    # Initialize positions on a grid with some randomness
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < n:
                x = (i + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                y = (j + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                positions.append([x, y])
    
    # Ensure we have exactly n positions
    positions = positions[:n]
    
    # Stage 2: Initialize with small radii and optimize
    # Start with small uniform radii
    radii = np.full(n, 0.02)
    
    # Define constraint functions
    def constraint_radius(i, pos, radii):
        """Ensure circle stays within bounds"""
        x, y = pos
        r = radii[i]
        return min(r, x - r, 1 - x - r, y - r, 1 - y - r)
    
    def constraint_overlap(i, j, pos_i, pos_j, r_i, r_j):
        """Ensure circles don't overlap"""
        dist = np.sqrt(np.sum((pos_i - pos_j)**2))
        return dist - r_i - r_j
    
    # Stage 3: Optimization using scipy minimize
    # Combine all variables into one array: [x1,y1,r1,x2,y2,r2,...]
    def objective(vars):
        # Return negative because we want to maximize sum of radii
        return -np.sum(vars[2::3])  # Sum of all radii
    
    def constraint_func(vars):
        # Check all constraints
        constraints = []
        
        # Position constraints (circle within unit square)
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            
            # Circle must fit within bounds
            constraints.append(x - r)  # x >= r
            constraints.append(y - r)  # y >= r
            constraints.append(1 - x - r)  # 1-x >= r
            constraints.append(1 - y - r)  # 1-y >= r
            
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                
                # Distance between centers minus sum of radii
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - r1 - r2)  # Should be >= 0
                
        return np.array(constraints)
    
    # Initial guess
    initial_vars = []
    for i in range(n):
        x, y = positions[i]
        # Place circles with some initial radius
        initial_vars.extend([x, y, 0.02])
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # Bounds for x and y: [r, 1-r]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Use SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            # Extract final solution
            for i in range(n):
                circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        else:
            # Fallback to grid-based solution if optimization fails
            # Use a more robust approach with better initial conditions
            circles = grid_based_initialization()
    except Exception:
        # If optimization fails completely, use fallback
        circles = grid_based_initialization()
    
    return circles

def grid_based_initialization():
    """Fallback method using a more structured grid approach"""
    n = 32
    circles = np.zeros((n, 3))
    
    # Create a refined grid pattern
    rows = 6
    cols = 6
    padding = 0.05
    
    # Distribute points in a grid pattern
    for i in range(n):
        row = i // cols
        col = i % cols
        
        if row >= rows or col >= cols:
            # Fill remaining positions randomly
            circles[i] = [
                np.random.uniform(padding, 1-padding),
                np.random.uniform(padding, 1-padding),
                0.02
            ]
        else:
            x = padding + (col + 0.5) * (1 - 2*padding) / cols
            y = padding + (row + 0.5) * (1 - 2*padding) / rows
            circles[i] = [x, y, 0.02]
    
    # Refine using local optimization for each circle
    for _ in range(100):
        improved = False
        for i in range(n):
            # Try to increase radius without violating constraints
            current_radius = circles[i, 2]
            best_radius = current_radius
            
            # Simple local search for radius improvement
            step = 0.001
            for _ in range(100):
                test_radius = best_radius + step
                if test_radius > 0.49:
                    break
                    
                # Check if this radius works
                valid = True
                for j in range(n):
                    if i != j:
                        dist = np.sqrt(
                            (circles[i, 0] - circles[j, 0])**2 +
                            (circles[i, 1] - circles[j, 1])**2
                        )
                        if dist < test_radius + circles[j, 2]:
                            valid = False
                            break
                
                if valid and test_radius > best_radius:
                    best_radius = test_radius
                else:
                    break
            
            if best_radius > current_radius:
                circles[i, 2] = best_radius
                improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
