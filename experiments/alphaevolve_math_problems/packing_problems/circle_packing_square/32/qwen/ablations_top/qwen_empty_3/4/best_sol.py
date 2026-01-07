# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal grid initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal grid pattern for good starting configuration
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern that fits within the unit square
        # We'll use a grid of approximately sqrt(n) rows and columns
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        
        # Calculate spacing to fit within unit square
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Generate initial positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Ensure positions are within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                positions.append([x, y])
        
        # If we don't have enough points, fill with random points
        while len(positions) < n:
            positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
            
        return np.array(positions[:n])
    
    # Initialize positions
    initial_positions = initialize_hexagonal_grid()
    
    # Define constraint functions
    def radius_constraint(i, circles_flat, n):
        """Ensure each circle has valid radius"""
        x, y, r = circles_flat[3*i:3*i+3]
        return min(r, x - r, y - r, 1 - x - r, 1 - y - r)
    
    def overlap_constraint(i, j, circles_flat, n):
        """Ensure no two circles overlap"""
        x1, y1, r1 = circles_flat[3*i:3*i+3]
        x2, y2, r2 = circles_flat[3*j:3*j+3]
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance - (r1 + r2)
    
    # Objective function to maximize (negative because minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of radii (negative for minimization)
    
    # Constraints for optimization
    constraints = []
    
    # Add radius constraints (each circle must fit in square)
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: radius_constraint(i, x, n)
        })
    
    # Add overlap constraints (no two circles may overlap)
    # Use a more efficient approach by limiting constraint pairs to reduce computation
    # We'll still generate all constraints but make sure they're properly bounded
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: overlap_constraint(i, j, x, n)
            })
    
    # Bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate  
        bounds.append((0.001, 0.499))  # radius (max radius is 0.5)
    
    # Initial guess: positions with small radii
    initial_guess = np.zeros(3*n)
    for i in range(n):
        initial_guess[3*i] = initial_positions[i][0]   # x
        initial_guess[3*i+1] = initial_positions[i][1] # y
        initial_guess[3*i+2] = 0.05                    # r (small initial radius)
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # Fallback: use initial positions with computed radii
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [initial_positions[i][0], initial_positions[i][1], 0.05]
    except Exception as e:
        # Fallback in case of optimization failure
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [initial_positions[i][0], initial_positions[i][1], 0.05]
    
    # Final refinement using a more targeted greedy approach
    # This improves upon the optimization result without being too aggressive
    circles = refine_solution(circles)
    
    return circles

def refine_solution(circles):
    """Refine the solution using a more careful greedy improvement approach"""
    n = len(circles)
    
    # Try to increase radii while maintaining constraints
    improved = True
    max_iterations = 50  # More iterations to allow better convergence
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Create KDTree for efficient neighbor search (only once per iteration)
        tree = cKDTree(circles[:, :2])
        
        # Try to increase each circle's radius
        for i in range(n):
            # Get neighbors within reasonable distance
            neighbors = tree.query_ball_point(circles[i, :2], 0.5)
            
            # Find minimum allowed radius based on boundaries
            min_radius = min(
                circles[i, 0],      # Distance to left boundary
                1 - circles[i, 0],  # Distance to right boundary
                circles[i, 1],      # Distance to bottom boundary
                1 - circles[i, 1]   # Distance to top boundary
            )
            
            # Check constraints with neighbors
            for j in neighbors:
                if i != j:
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + 
                                 (circles[i, 1] - circles[j, 1])**2)
                    max_radius_from_neighbor = dist - circles[j, 2]
                    min_radius = min(min_radius, max_radius_from_neighbor)
            
            # Increase radius if beneficial and safe
            if min_radius > circles[i, 2] + 1e-6 and min_radius > 0.001:
                # Slightly larger increments to push towards better solutions
                old_radius = circles[i, 2]
                new_radius = min(old_radius + 0.005, min_radius)
                if new_radius > old_radius:
                    circles[i, 2] = new_radius
                    improved = True
    
    # Final cleanup: enforce all constraints one more time
    for i in range(n):
        # Ensure containment
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
    
    return circles


# EVOLVE-BLOCK-END
