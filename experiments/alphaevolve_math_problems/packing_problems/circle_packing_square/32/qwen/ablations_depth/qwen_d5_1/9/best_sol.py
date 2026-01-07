# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from itertools import combinations
from scipy.optimize import differential_evolution

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
    Uses a multi-phase approach with multi-start optimization and advanced techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    best_sum = 0
    best_circles = None
    
    # Multi-start approach with different initialization strategies
    for attempt in range(8):  # Increased attempts for better chance
        np.random.seed(42 + attempt)  # Different seed for each attempt
        
        # Strategy 1: Hexagonal grid initialization for better packing density
        if attempt % 3 == 0:
            # Hexagonal grid pattern
            grid_size = int(np.ceil(np.sqrt(n)))
            hex_points = []
            
            # Create hexagonal lattice
            spacing = 1.0 / (grid_size + 2)
            sqrt3 = np.sqrt(3)
            
            for i in range(grid_size + 1):
                for j in range(grid_size + 1):
                    if len(hex_points) < n:
                        x = (i + 0.5 * (j % 2)) * spacing + np.random.normal(0, 0.003 * spacing)
                        y = j * spacing * sqrt3 / 2 + np.random.normal(0, 0.003 * spacing)
                        # Keep within bounds
                        x = max(0.01, min(0.99, x))
                        y = max(0.01, min(0.99, y))
                        hex_points.append([x, y])
            
            # If we don't have enough points, add random ones
            if len(hex_points) < n:
                extra_points = np.random.rand(n - len(hex_points), 2)
                extra_points[:, 0] = extra_points[:, 0] * 0.98 + 0.01
                extra_points[:, 1] = extra_points[:, 1] * 0.98 + 0.01
                hex_points.extend(extra_points.tolist())
            
            centers = np.array(hex_points[:n])
            
        else:
            # Grid-based initialization (as before)
            grid_size = int(np.ceil(np.sqrt(n)))
            grid_points = []
            
            # Generate grid points
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(grid_points) < n:
                        # Add some randomness to avoid symmetric solutions
                        x = (i + 1) * spacing_x + np.random.normal(0, 0.005 * spacing_x)
                        y = (j + 1) * spacing_y + np.random.normal(0, 0.005 * spacing_y)
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
        
        # Phase 2: Initialize radii with a more intelligent approach
        # Start with smaller radii and let optimization increase them
        radii = np.full(n, 0.02)
        
        # Phase 3: More efficient constraint handling
        # Variables: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        initial_vars = np.column_stack([centers, radii]).flatten()
        
        def objective(vars):
            """Minimize negative sum of radii (equivalent to maximizing sum of radii)"""
            radii = vars[2::3]  # Every third element starting from index 2
            return -np.sum(radii)
        
        def constraint_func(vars):
            """Constraint function for non-overlapping condition - optimized version"""
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
            bounds.append((0.001, 0.5))  # Radius bounded to avoid extreme values
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            {'type': 'ineq', 'fun': lambda x: containment_constraints(x)}
        ]
        
        # Optimization parameters - try different methods
        options = {'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
        
        try:
            # Try both optimization methods for better results
            # First try SLSQP
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options=options,
                tol=1e-6
            )
            
            if result.success:
                final_vars = result.x
                circles = final_vars.reshape(-1, 3)
                current_sum = np.sum(circles[:, 2])
                
                # Update best solution if this one is better
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
            else:
                # Even if optimization fails, keep the initial solution for this attempt
                circles = np.column_stack([centers, radii])
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
                    
        except Exception as e:
            # If anything goes wrong, just keep the initial configuration
            circles = np.column_stack([centers, radii])
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
    
    # If we never found a good solution, return the best we have
    if best_circles is None:
        # Fallback to a simple approach
        centers = np.random.rand(n, 2)
        centers[:, 0] = centers[:, 0] * 0.98 + 0.01
        centers[:, 1] = centers[:, 1] * 0.98 + 0.01
        radii = np.full(n, 0.03)
        best_circles = np.column_stack([centers, radii])
    
    return best_circles


# EVOLVE-BLOCK-END
