# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Phase 1: Initialize with a good starting configuration
    # Arrange circles in a grid-like pattern with some randomness
    circles = np.zeros((n, 3))
    
    # Create a roughly grid-based initialization
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Distribute circles in a grid pattern
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Position in grid with padding
        x = (col + 0.5) / cols * 0.8 + 0.1
        y = (row + 0.5) / rows * 0.8 + 0.1
        
        # Initial radius - start with small values to ensure feasibility
        r = min(0.05, 0.5 / (cols + rows))
        
        circles[i] = [x, y, r]
    
    # Phase 2: Optimization using constrained minimization
    # We'll minimize negative of sum of radii (equivalent to maximizing sum)
    
    def objective(params):
        # Reshape parameters into positions and radii
        positions_and_radii = params.reshape(-1, 3)
        radii = positions_and_radii[:, 2]
        return -np.sum(radii)
    
    def constraint_containment(params):
        """Ensure all circles are contained within the unit square"""
        positions_and_radii = params.reshape(-1, 3)
        x = positions_and_radii[:, 0]
        y = positions_and_radii[:, 1]
        r = positions_and_radii[:, 2]
        
        # Each circle must satisfy: r <= x <= 1-r and r <= y <= 1-r
        # This gives us constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        return np.concatenate([
            x - r,           # x >= r
            1 - x - r,       # x <= 1-r
            y - r,           # y >= r
            1 - y - r        # y <= 1-r
        ])
    
    def constraint_nonoverlap(params):
        """Ensure no two circles overlap"""
        positions_and_radii = params.reshape(-1, 3)
        positions = positions_and_radii[:, :2]
        radii = positions_and_radii[:, 2]
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        # For each pair, we want distance >= sum of radii
        # So we want: distances[i,j] >= radii[i] + radii[j]
        # Which means: distances[i,j] - radii[i] - radii[j] >= 0
        # But we only consider i < j to avoid double counting
        
        constraints = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = distances[i, j]
                rad_sum = radii[i] + radii[j]
                constraints.append(dist - rad_sum)
        
        return np.array(constraints)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]
    
    # Set bounds: x,y in [r, 1-r], r in [0, 0.5]
    bounds = []
    for i in range(n):
        # x coordinate: [r, 1-r] => we need r <= x <= 1-r
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    # Initial guess
    initial_params = circles.flatten()
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure final constraints are satisfied
            final_positions_and_radii = optimized_circles.copy()
            
            # Re-optimize with tighter constraints if needed
            # For better results, let's also try a local refinement approach
            
            # Refinement: Apply a few iterations of local optimization
            for _ in range(3):
                refined_result = minimize(
                    objective,
                    final_positions_and_radii.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if refined_result.success:
                    final_positions_and_radii = refined_result.x.reshape(-1, 3)
                else:
                    break
                    
            return final_positions_and_radii
        else:
            # Return initial configuration if optimization fails
            return circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
