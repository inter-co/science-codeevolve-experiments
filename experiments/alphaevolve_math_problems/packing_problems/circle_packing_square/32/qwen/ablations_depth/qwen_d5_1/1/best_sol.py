# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')
from numba import jit
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
    Uses a hybrid approach: enhanced geometric construction + simulated annealing-inspired 
    optimization + spatial indexing for collision detection.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Enhanced Geometric Construction with Better Initial Placement
    # Use a more refined approach based on known optimal packings for small numbers
    np.random.seed(42)  # For reproducibility
    
    # Create a better initial configuration using a combination of regular and irregular placements
    centers = []
    
    # First, create a structured layout (like a 6x6 grid with slight perturbations)
    grid_size = 6
    for i in range(grid_size):
        for j in range(grid_size):
            if len(centers) < n:
                # Create grid with slight perturbation to avoid symmetry
                x = (j + 0.5 + np.random.normal(0, 0.05)) / grid_size
                y = (i + 0.5 + np.random.normal(0, 0.05)) / grid_size
                
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                centers.append([x, y])
    
    # If we don't have enough points, add random ones
    if len(centers) < n:
        extra_points = np.random.rand(n - len(centers), 2)
        extra_points[:, 0] = extra_points[:, 0] * 0.98 + 0.01
        extra_points[:, 1] = extra_points[:, 1] * 0.98 + 0.01
        centers.extend(extra_points.tolist())
    
    centers = np.array(centers[:n])
    
    # Phase 2: Smart Radius Initialization Based on Local Density
    # Initialize radii with a more sophisticated approach
    initial_radii = np.full(n, 0.05)
    
    # Use a density-based approach for initial radii estimation
    tree = cKDTree(centers)
    
    # For each point, find neighbors and adjust radius based on local crowding
    for i in range(n):
        # Find nearby neighbors
        neighbors = tree.query_ball_point(centers[i], 0.2)
        if len(neighbors) > 1:
            # More neighbors = more crowded area = smaller radius
            # Base radius inversely proportional to neighbor count
            density_factor = min(1.0, 5.0 / len(neighbors))
            initial_radii[i] = max(0.005, 0.08 * density_factor)
    
    # Phase 3: Advanced Multi-stage Optimization
    # Stage 1: Pre-optimization with a simplified model to get a good baseline
    positions = centers.copy()
    radii = initial_radii.copy()
    
    # Simulated Annealing inspired approach with adaptive cooling
    def optimize_positions_and_radii():
        # Create a working copy
        pos = positions.copy()
        rad = radii.copy()
        
        # Simple energy minimization approach
        for iteration in range(500):
            # Perturb positions slightly
            delta_pos = np.random.normal(0, 0.001, pos.shape)
            new_pos = pos + delta_pos
            
            # Keep within bounds
            new_pos[:, 0] = np.clip(new_pos[:, 0], rad, 1 - rad)
            new_pos[:, 1] = np.clip(new_pos[:, 1], rad, 1 - rad)
            
            # Calculate objective improvement
            old_obj = -np.sum(rad)  # Negative because we want to maximize sum of radii
            
            # Test new configuration
            new_rad = rad.copy()
            # Simple greedy adjustment of radii
            for i in range(n):
                # Try to increase radius while maintaining constraints
                test_rad = min(0.5, rad[i] + np.random.uniform(-0.005, 0.005))
                if test_rad > 0.001:
                    # Check if this change maintains feasibility
                    valid = True
                    for j in range(n):
                        if i != j:
                            dist = np.sqrt((new_pos[i,0]-new_pos[j,0])**2 + (new_pos[i,1]-new_pos[j,1])**2)
                            if dist < test_rad + rad[j]:
                                valid = False
                                break
                    if valid:
                        new_rad[i] = test_rad
            
            new_obj = -np.sum(new_rad)
            
            # Accept improvement or accept with probability based on temperature
            if new_obj < old_obj:  # Better solution
                pos = new_pos.copy()
                rad = new_rad.copy()
            elif np.random.rand() < np.exp(-(new_obj - old_obj) * 100):  # Accept worse solution
                pos = new_pos.copy()
                rad = new_rad.copy()
        
        return pos, rad
    
    # Run the pre-optimization
    try:
        positions, radii = optimize_positions_and_radii()
    except Exception:
        pass  # Fall back to initial configuration if something fails
    
    # Stage 2: Fine-grained optimization using scipy with better parameter tuning
    # Variables: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    initial_vars = np.column_stack([positions, radii]).flatten()
    
    def objective(vars):
        """Minimize negative sum of radii (equivalent to maximizing sum of radii)"""
        radii = vars[2::3]  # Every third element starting from index 2
        return -np.sum(radii)
    
    def constraint_func(vars):
        """Constraint function for non-overlapping condition"""
        positions = vars.reshape(-1, 3)[:, :2]  # Extract (x,y) coordinates
        radii = vars.reshape(-1, 3)[:, 2]       # Extract radii
        
        # Use efficient distance computation
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
    
    # Phase 4: Use a more robust optimization approach with multiple restarts
    best_result = None
    best_sum = float('-inf')
    
    # Try multiple optimization runs with different starting points
    for run in range(3):
        try:
            # Start with different perturbations
            np.random.seed(42 + run)  # Different seeds for different runs
            perturbed_vars = initial_vars + np.random.normal(0, 0.01, initial_vars.shape)
            
            # Run optimization with SLSQP
            result = minimize(
                objective,
                perturbed_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-8
            )
            
            if result.success:
                current_sum = -result.fun  # Convert back to positive sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception:
            continue  # Skip failed runs
    
    # If we found a good result, use it; otherwise fall back to initial
    if best_result is not None and best_result.success:
        final_vars = best_result.x
        circles = final_vars.reshape(-1, 3)
    else:
        circles = initial_vars.reshape(-1, 3)
    
    # Ensure we have exactly 32 circles
    if circles.shape[0] < n:
        # Fill with zeros
        padding = np.zeros((n - circles.shape[0], 3))
        circles = np.vstack([circles, padding])
    
    return circles


# EVOLVE-BLOCK-END
