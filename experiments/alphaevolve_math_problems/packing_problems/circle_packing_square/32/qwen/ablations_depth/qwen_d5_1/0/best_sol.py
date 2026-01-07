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
    Uses a hybrid approach: discrete geometric construction + physics-inspired optimization + 
    spatial indexing for collision detection.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Discrete Geometric Construction
    # Generate a lattice-based arrangement that approximates optimal packing
    # Start with a hexagonal close packing pattern
    
    # Create a hexagonal grid with some randomness to avoid regular patterns
    np.random.seed(42)  # For reproducibility
    
    # Calculate hexagonal grid parameters
    side_length = np.sqrt(2/n)  # Approximate side length for hexagonal packing
    
    # Generate hexagonal grid points
    hex_points = []
    rows = int(np.ceil(np.sqrt(n))) + 2
    cols = int(np.ceil(n / rows)) + 2
    
    for i in range(rows):
        for j in range(cols):
            # Offset every other row
            x_offset = 0.5 if i % 2 == 0 else 0.75
            x = (j + x_offset) * side_length * 0.8
            y = i * side_length * np.sqrt(3)/2 * 0.8
            
            # Keep within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                hex_points.append([x, y])
    
    # If we don't have enough points, add random ones
    if len(hex_points) < n:
        extra_points = np.random.rand(n - len(hex_points), 2)
        extra_points[:, 0] = extra_points[:, 0] * 0.98 + 0.01
        extra_points[:, 1] = extra_points[:, 1] * 0.98 + 0.01
        hex_points.extend(extra_points.tolist())
    
    # Select n points
    centers = np.array(hex_points[:n])
    
    # Phase 2: Physics-based Initialization with Adaptive Radii
    # Use a force-based approach to distribute initial radii
    initial_radii = np.full(n, 0.02)
    
    # Estimate initial radii based on local density
    tree = cKDTree(centers)
    
    # For each point, find neighbors within a certain radius
    neighbor_radius = 0.2
    for i in range(n):
        neighbors = tree.query_ball_point(centers[i], neighbor_radius)
        if len(neighbors) > 1:
            # More neighbors = more constrained space = smaller radius
            initial_radii[i] = max(0.005, 0.05 * (1.0 / len(neighbors)))
    
    # Phase 3: Multi-stage Optimization
    # Stage 1: Coarse-grained optimization using a simpler approach
    def compute_violations(positions, radii):
        """Compute total overlap violations"""
        total_violation = 0
        tree = cKDTree(positions)
        
        # Check all pairs for overlaps
        for i in range(len(positions)):
            # Find neighbors within 2*(r_i + r_j) distance
            neighbors = tree.query_ball_point(positions[i], 2*(radii[i] + 0.1))
            for j in neighbors:
                if i != j:
                    dist = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    overlap = radii[i] + radii[j] - dist
                    if overlap > 0:
                        total_violation += overlap
        
        return total_violation
    
    # Stage 2: Improved initialization using the coarse approach
    positions = centers.copy()
    radii = initial_radii.copy()
    
    # Refine with a simple physics simulation
    for _ in range(100):
        # Compute forces between circles
        forces = np.zeros_like(positions)
        
        tree = cKDTree(positions)
        for i in range(n):
            neighbors = tree.query_ball_point(positions[i], 2*(radii[i] + 0.05))
            for j in neighbors:
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # Repulsion force (inverse square law)
                        force_magnitude = 1.0 / (dist * dist + 0.01)
                        forces[i, 0] += force_magnitude * dx / dist
                        forces[i, 1] += force_magnitude * dy / dist
                        
                        # Attraction to center for boundary avoidance
                        if positions[i, 0] < 0.1 or positions[i, 0] > 0.9 or \
                           positions[i, 1] < 0.1 or positions[i, 1] > 0.9:
                            center_force = 0.01
                            forces[i, 0] -= center_force * (positions[i, 0] - 0.5)
                            forces[i, 1] -= center_force * (positions[i, 1] - 0.5)
        
        # Update positions
        step_size = 0.001
        positions += step_size * forces
        
        # Enforce boundary constraints
        positions[:, 0] = np.clip(positions[:, 0], radii, 1 - radii)
        positions[:, 1] = np.clip(positions[:, 1], radii, 1 - radii)
    
    # Stage 3: Fine-grained optimization using scipy
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
    
    # Phase 4: Use differential evolution for global search first
    # This gives us a much better starting point than simple local optimization
    try:
        # Differential evolution for global search (faster than full optimization)
        de_result = differential_evolution(
            objective,
            bounds,
            constraints=cons,
            maxiter=100,
            popsize=15,
            seed=42,
            polish=False  # Skip polishing to save time
        )
        
        if de_result.success:
            # Use DE result as starting point for local optimization
            initial_vars = de_result.x
        else:
            # Fallback to initial guess if DE fails
            pass
            
    except Exception as e:
        # If DE fails, continue with initial guess
        pass
    
    # Phase 5: Local optimization with SLSQP for fine-tuning
    try:
        # Run optimization with SLSQP for better local refinement
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        
        if result.success:
            final_vars = result.x
            circles = final_vars.reshape(-1, 3)
        else:
            # If optimization fails, return the best we have
            circles = initial_vars.reshape(-1, 3)
            
    except Exception as e:
        # If anything goes wrong, return the initial configuration
        circles = initial_vars.reshape(-1, 3)
    
    # Ensure we have exactly 32 circles
    if circles.shape[0] < n:
        # Fill with zeros
        padding = np.zeros((n - circles.shape[0], 3))
        circles = np.vstack([circles, padding])
    
    return circles


# EVOLVE-BLOCK-END
