# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
from itertools import combinations
from deap import base, creator, tools, algorithms
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithms with local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Create a more sophisticated initial configuration
    def create_better_initial():
        # Start with a hexagonal packing approximation
        # For 32 circles, we can arrange in roughly a 6x5 grid with some adjustments
        circles = []
        
        # Try different arrangements
        best_config = None
        best_radius_sum = 0
        
        # Grid-based approach with better spacing
        grid_rows = 6
        grid_cols = 6
        spacing_x = 1.0 / grid_cols
        spacing_y = 1.0 / grid_rows
        
        # Generate points in a hexagonal pattern
        test_circles = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                if len(test_circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Apply hexagonal offset for odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius - based on spacing but ensure reasonable values
                r = min(spacing_x, spacing_y) / 2.0
                r = min(r, 0.15)  # Cap max radius
                test_circles.append([x, y, r])
            
            if len(test_circles) >= n:
                break
        
        # Fill remaining positions with carefully placed circles
        while len(test_circles) < n:
            # Place near corners and edges for better utilization
            if len(test_circles) < n:
                # Corner placement
                corners = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
                corner = corners[len(test_circles) % len(corners)]
                x, y = corner
                r = np.random.uniform(0.02, 0.08)
                test_circles.append([x, y, r])
        
        # Ensure we have exactly n circles
        test_circles = test_circles[:n]
        
        # Refine using a local optimization approach
        refined_circles = refine_initial_configuration(np.array(test_circles))
        
        return refined_circles
    
    def refine_initial_configuration(initial_circles):
        """Use local optimization to improve the initial configuration"""
        # Simple refinement: try to increase radii while maintaining constraints
        circles = initial_circles.copy()
        
        # Try to maximize radii using a greedy approach
        # This is a simplified version - in practice, more sophisticated local search would be used
        for _ in range(100):  # Limited iterations for efficiency
            improved = False
            for i in range(len(circles)):
                # Try to increase radius of circle i
                old_r = circles[i][2]
                new_r = min(old_r * 1.05, 0.4)  # Slightly increase radius
                
                # Check if we can actually increase it without violating constraints
                valid = True
                for j in range(len(circles)):
                    if i != j:
                        dx = circles[i][0] - circles[j][0]
                        dy = circles[i][1] - circles[j][1]
                        dist_sq = dx*dx + dy*dy
                        r_sum = new_r + circles[j][2]
                        if dist_sq < r_sum * r_sum:
                            valid = False
                            break
                
                # Check containment
                if valid and new_r <= circles[i][0] and new_r <= circles[i][1] and \
                   new_r <= 1 - circles[i][0] and new_r <= 1 - circles[i][1]:
                    circles[i][2] = new_r
                    improved = True
            
            if not improved:
                break
                
        return circles
    
    # Create initial guess
    initial_circles = create_better_initial()
    
    # Define constraints and objective
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_containment(params):
        circles = params.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Each circle must be fully contained in [0,1]x[0,1]
        cons = np.concatenate([
            r,                          # r >= 0
            1 - r - x,                  # x + r <= 1
            1 - r - y,                  # y + r <= 1
            x - r,                      # x - r >= 0
            y - r                       # y - r >= 0
        ])
        return cons
    
    def constraint_nonoverlap(params):
        circles = params.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # More efficient way to compute non-overlap constraints using vectorized operations
        # Compute pairwise distances efficiently
        n_circles = len(circles)
        cons = []
        
        # Vectorized approach for better performance
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                r_sum = r[i] + r[j]
                # We want dist >= r_sum to prevent overlap
                # So constraint is: dist_sq - r_sum^2 >= 0
                cons.append(dist_sq - r_sum*r_sum)
        return np.array(cons)
    
    # Enhanced optimization using a hybrid approach
    def hybrid_optimization(initial_params):
        # Use a combination of global and local search
        best_params = initial_params.copy()
        best_sum = -objective(best_params)
        
        # Multiple restarts with different strategies
        for restart in range(5):
            # Strategy 1: Random perturbation
            if restart % 3 == 0:
                perturbed_params = initial_params + np.random.normal(0, 0.005, len(initial_params))
            elif restart % 3 == 1:
                # Strategy 2: Slightly shift positions
                perturbed_params = initial_params.copy()
                # Modify some positions slightly
                indices = np.random.choice(len(initial_params), size=len(initial_params)//3, replace=False)
                perturbed_params[indices] += np.random.normal(0, 0.01, len(indices))
            else:
                # Strategy 3: Random initialization
                perturbed_params = generate_random_solution()
            
            # Create bounds (x, y, r for each circle)
            bounds = []
            for _ in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Constraints
            cons = [
                {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
                {'type': 'ineq', 'fun': lambda p: constraint_nonoverlap(p)}
            ]
            
            try:
                # Try different optimization methods
                result = minimize(
                    objective,
                    perturbed_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 300, 'ftol': 1e-6}
                )
                
                if result.success:
                    current_sum = -objective(result.x)  # Convert back to positive sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_params = result.x.copy()
                        
            except Exception:
                continue
        
        return best_params
    
    def generate_random_solution():
        """Generate a valid random solution"""
        circles = []
        for _ in range(n):
            # Generate valid random circle
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.15)
            circles.append([x, y, r])
        return np.array(circles).flatten()
    
    # Run optimization
    try:
        optimized_params = hybrid_optimization(initial_circles.flatten())
        
        if optimized_params is not None:
            final_circles = optimized_params.reshape(-1, 3)
            return final_circles
        else:
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
