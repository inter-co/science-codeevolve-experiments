# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from scipy.optimize import differential_evolution
import time

@jit(nopython=True)
def compute_pairwise_distances_numba(positions):
    """Compute pairwise distances efficiently using numba"""
    n = positions.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining global optimization with local refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initialization using hexagonal packing approximation
    np.random.seed(42)  # For reproducibility
    
    # Create a hexagonal grid pattern which typically gives better initial configurations
    # Hexagonal packing density is ~0.9069, so we'll use a systematic approach
    grid_size = int(np.ceil(np.sqrt(n)))
    grid_points = []
    
    # Generate hexagonal grid points
    spacing = 1.0 / (grid_size + 1)
    hex_spacing = spacing * 0.866  # sqrt(3)/2 for hexagonal packing
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(grid_points) < n:
                # Offset every other row
                x_offset = (j % 2) * hex_spacing
                x = (i + 1) * spacing + x_offset + np.random.normal(0, 0.005 * spacing)
                y = (j + 1) * hex_spacing + np.random.normal(0, 0.005 * hex_spacing)
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                grid_points.append([x, y])
    
    # If we don't have enough points, add random ones
    if len(grid_points) < n:
        extra_points = np.random.rand(n - len(grid_points), 2)
        extra_points[:, 0] = extra_points[:, 0] * 0.98 + 0.01
        extra_points[:, 1] = extra_points[:, 1] * 0.98 + 0.01
        grid_points.extend(extra_points.tolist())
    
    centers = np.array(grid_points[:n])
    
    # Phase 2: Initialize radii based on local density estimation
    # Start with a reasonable estimate for dense packing
    radii = np.full(n, 0.03)  # Smaller initial radii to allow room for growth
    
    # Phase 3: Efficient constraint handling with proper optimization approach
    # Variables: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    initial_vars = np.column_stack([centers, radii]).flatten()
    
    def objective(vars):
        """Minimize negative sum of radii (equivalent to maximizing sum of radii)"""
        radii = vars[2::3]  # Every third element starting from index 2
        return -np.sum(radii)
    
    def constraint_func(vars):
        """Constraint function for non-overlapping condition"""
        positions = vars.reshape(-1, 3)[:, :2]  # Extract (x,y) coordinates
        radii = vars.reshape(-1, 3)[:, 2]       # Extract radii
        
        # Use optimized distance computation
        distances = compute_pairwise_distances_numba(positions)
        
        # Constraint: distance between centers >= sum of radii
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                # Add constraint: distance - radii >= 0
                constraint_val = distances[i, j] - radii[i] - radii[j]
                constraints.append(constraint_val)
        
        return np.array(constraints)
    
    def containment_constraints(vars):
        """Ensure all circles are within the unit square"""
        positions = vars.reshape(-1, 3)[:, :2]
        radii = vars.reshape(-1, 3)[:, 2]
        
        constraints = []
        for i in range(n):
            # x - r >= 0
            constraints.append(positions[i, 0] - radii[i])
            # 1 - x - r >= 0  
            constraints.append(1 - positions[i, 0] - radii[i])
            # y - r >= 0
            constraints.append(positions[i, 1] - radii[i])
            # 1 - y - r >= 0
            constraints.append(1 - positions[i, 1] - radii[i])
        
        return np.array(constraints)
    
    # Set up bounds for variables
    bounds = []
    for i in range(n):
        # Bounds for x coordinate
        bounds.append((0.001, 0.999))  # Slightly away from boundaries to prevent issues
        # Bounds for y coordinate  
        bounds.append((0.001, 0.999))
        # Bounds for radius
        bounds.append((0.001, 0.3))  # Reduced upper bound to avoid numerical issues
    
    # Define constraints - use multiple constraint types for better handling
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_func(x)},
        {'type': 'ineq', 'fun': lambda x: containment_constraints(x)}
    ]
    
    # Phase 4: Use differential evolution for global search first, then local optimization
    print("Starting global optimization with differential evolution...")
    
    # Global optimization with differential evolution
    try:
        # Use a simpler approach first with DE for global search
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            # Use DE result as starting point for local optimization
            initial_vars = de_result.x
        else:
            print("Differential evolution failed, using grid initialization")
    except Exception as e:
        print(f"Differential evolution error: {e}")
        pass
    
    # Local optimization with SLSQP
    print("Starting local optimization with SLSQP...")
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    
    try:
        # Run optimization with a better method for constrained problems
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options=options,
            tol=1e-8,
            callback=lambda x: print(f"Current sum of radii: {-objective(x)}") if len(x) > 0 else None
        )
        
        if result.success:
            final_vars = result.x
            circles = final_vars.reshape(-1, 3)
        else:
            # If optimization fails, return the initial configuration
            circles = np.column_stack([centers, radii])
            
    except Exception as e:
        print(f"Optimization error: {e}")
        # If anything goes wrong, return the initial configuration
        circles = np.column_stack([centers, radii])
    
    # Ensure we have exactly 32 circles
    if circles.shape[0] < n:
        # Fill with zeros
        padding = np.zeros((n - circles.shape[0], 3))
        circles = np.vstack([circles, padding])
    
    return circles


# EVOLVE-BLOCK-END
