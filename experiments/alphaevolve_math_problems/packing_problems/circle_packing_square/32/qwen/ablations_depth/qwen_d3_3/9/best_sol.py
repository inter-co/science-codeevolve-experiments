# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.spatial import KDTree
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with advanced optimization techniques.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using a more systematic approach
    # Start with a dense packing approximation
    circles = np.zeros((n, 3))
    
    # Use a more strategic initial placement - place in a grid-like pattern with some randomness
    # This helps avoid poor local optima early on
    
    # Grid dimensions for initial placement
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            # Add slight randomness to avoid perfect grid patterns
            x = (j + 1) * spacing_x + random.uniform(-spacing_x*0.1, spacing_x*0.1)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y*0.1, spacing_y*0.1)
            circles[idx] = [x, y, 0.0]  # Initialize with zero radius
            idx += 1
        if idx >= n:
            break
    
    # Set initial radii to be small but feasible
    min_radius = 0.01
    for i in range(n):
        circles[i][2] = min_radius
    
    # Improved optimization approach using a two-phase strategy:
    # Phase 1: Optimize radii while keeping positions fixed
    # Phase 2: Optimize both positions and radii
    
    # Phase 1: Radii optimization with proper constraints
    def optimize_radii(circles_positions_radii):
        # Extract positions and radii
        positions = circles_positions_radii[:, :2]
        radii = circles_positions_radii[:, 2]
        
        def objective(radii):
            return -np.sum(radii)  # Negative because we want to maximize
        
        def constraint_containment(i, radii):
            x, y = positions[i, 0], positions[i, 1]
            r = radii[i]
            # Check containment constraints
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        
        def constraint_overlap(i, j, radii):
            x1, y1 = positions[i, 0], positions[i, 1]
            x2, y2 = positions[j, 0], positions[j, 1]
            r1, r2 = radii[i], radii[j]
            # Distance between centers minus sum of radii
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return dist - (r1 + r2)
        
        # Initial guess for radii
        initial_radii = radii.copy()
        
        # Set up constraints
        constraints = []
        
        # Containment constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda radii, i=i: constraint_containment(i, radii)})
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': lambda radii, i=i, j=j: constraint_overlap(i, j, radii)})
        
        # Bounds for radii (positive, reasonable upper bound)
        bounds = [(0.001, 0.5) for _ in range(n)]
        
        # Perform optimization
        try:
            result = minimize(objective, initial_radii, method='SLSQP', bounds=bounds, constraints=constraints, 
                             options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6})
            
            if result.success:
                return result.x
            else:
                return initial_radii
        except:
            return initial_radii
    
    # Phase 2: Combined optimization of positions and radii
    def optimize_positions_and_radii(circles):
        # Convert to flat array for optimization
        initial_flat = circles.flatten()
        
        def objective(flat_params):
            # Reconstruct circles from flat parameters
            circles_reconstructed = flat_params.reshape((n, 3))
            return -np.sum(circles_reconstructed[:, 2])  # Maximize sum of radii
        
        def constraint_containment(i, flat_params):
            circles_reconstructed = flat_params.reshape((n, 3))
            x, y = circles_reconstructed[i, 0], circles_reconstructed[i, 1]
            r = circles_reconstructed[i, 2]
            # Check containment constraints
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        
        def constraint_overlap(i, j, flat_params):
            circles_reconstructed = flat_params.reshape((n, 3))
            x1, y1 = circles_reconstructed[i, 0], circles_reconstructed[i, 1]
            x2, y2 = circles_reconstructed[j, 0], circles_reconstructed[j, 1]
            r1, r2 = circles_reconstructed[i, 2], circles_reconstructed[j, 2]
            # Distance between centers minus sum of radii
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return dist - (r1 + r2)
        
        # Set up constraints
        constraints = []
        
        # Containment constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda flat_params, i=i: constraint_containment(i, flat_params)})
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': lambda flat_params, i=i, j=j: constraint_overlap(i, j, flat_params)})
        
        # Bounds for positions (0,1) and radii (0.001, 0.5)
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 1.0), (0.001, 1.0), (0.001, 0.5)])  # x, y, r bounds
        
        # Perform optimization
        try:
            result = minimize(objective, initial_flat, method='SLSQP', bounds=bounds, constraints=constraints, 
                             options={'maxiter': 300, 'ftol': 1e-5})
            
            if result.success:
                return result.x.reshape((n, 3))
            else:
                return circles
        except:
            return circles
    
    # Run the optimization phases
    # Phase 1: Optimize radii first
    for _ in range(3):
        radii_optimized = optimize_radii(circles)
        circles[:, 2] = radii_optimized
    
    # Phase 2: Optimize both positions and radii
    circles = optimize_positions_and_radii(circles)
    
    # Final refinement using a more sophisticated approach
    # Use a greedy algorithm with spatial indexing for better neighbor detection
    tree = KDTree(circles[:, :2])
    
    # Refinement loop
    max_refinements = 10
    for _ in range(max_refinements):
        improved = False
        # Try to increase each circle's radius
        for i in range(n):
            # Get neighbors within a reasonable distance
            neighbors = tree.query_ball_point(circles[i, :2], 0.5)
            neighbors = [idx for idx in neighbors if idx != i]
            
            # Compute maximum possible radius for this circle
            max_radius = float('inf')
            
            # Check containment
            x, y = circles[i, 0], circles[i, 1]
            containment_radius = min(x, 1-x, y, 1-y)
            max_radius = min(max_radius, containment_radius)
            
            # Check overlap with neighbors
            for j in neighbors:
                x2, y2 = circles[j, 0], circles[j, 1]
                r2 = circles[j, 2]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist > 0:  # Avoid division by zero
                    max_radius = min(max_radius, dist - r2)
            
            # Increase radius if beneficial
            if max_radius > circles[i, 2] and max_radius > 0:
                circles[i, 2] = min(max_radius, circles[i, 2] * 1.1)  # Small increment
                improved = True
        
        if not improved:
            break
    
    # Final cleanup to ensure all constraints are satisfied
    for i in range(n):
        x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
        # Ensure containment
        r = min(r, x, 1-x, y, 1-y)
        # Ensure non-negative
        r = max(r, 0.001)
        circles[i, 2] = r
    
    return circles


# EVOLVE-BLOCK-END
