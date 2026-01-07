# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
import itertools
from numba import jit
import time

@jit(nopython=True)
def compute_distance_squared(x1, y1, x2, y2):
    """Fast computation of squared distance"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def compute_min_distance_to_points_fast(x, y, points, n):
    """Compute minimum distance to all other points - optimized version"""
    min_dist_sq = 1e20
    for i in range(n):
        dist_sq = compute_distance_squared(x, y, points[i, 0], points[i, 1])
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return min_dist_sq

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    start_time = time.time()
    
    # Stage 1: Better initial placement using hexagonal packing inspiration
    np.random.seed(42)  # For reproducibility
    
    # Create a more systematic initial placement
    # Try to place points in a pattern that's closer to optimal
    circles = []
    
    # Create a refined grid pattern
    grid_size = 6  # More refined grid for better coverage
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    # Create a hexagonal-like pattern for better initial spread
    for i in range(grid_size):
        for j in range(grid_size):
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            if 0 <= x <= 1 and 0 <= y <= 1:
                circles.append([x, y])
    
    # Add some randomness to avoid perfect grid patterns
    if len(circles) < n:
        for _ in range(n - len(circles)):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y])
    
    # Ensure we have exactly n circles
    circles = np.array(circles[:n])
    
    # Stage 2: Estimate initial radii with better approach
    try:
        radii = np.zeros(n)
        
        for i in range(n):
            # Find minimum distance to any other circle center
            min_dist_sq = 1e20
            for j in range(n):
                if i != j:
                    dist_sq = compute_distance_squared(circles[i,0], circles[i,1], 
                                                      circles[j,0], circles[j,1])
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
            
            min_dist = np.sqrt(min_dist_sq)
            
            # Boundary constraints
            boundary_dist = min(
                circles[i,0], 
                1 - circles[i,0],
                circles[i,1],
                1 - circles[i,1]
            )
            
            # Use minimum of boundary and neighbor distances, divided by 2
            max_radius = min(boundary_dist, min_dist / 2.0)
            radii[i] = max(0.001, max_radius)
                
    except Exception as e:
        # Fallback to uniform distribution if Voronoi fails
        radii = np.full(n, 0.05)
    
    # Stage 3: Advanced optimization with better constraint handling
    # Flatten the parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = np.zeros(3 * n)
    for i in range(n):
        initial_params[3*i] = circles[i][0]
        initial_params[3*i+1] = circles[i][1]
        initial_params[3*i+2] = radii[i]
    
    # Define objective function - minimize negative sum of radii
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)
    
    # Constraint functions with better numerical stability
    def containment_constraints(params):
        """Ensure all circles are within the unit square"""
        res = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # x - r >= 0 and y - r >= 0 and 1 - x - r >= 0 and 1 - y - r >= 0
            res.extend([x - r, y - r, 1 - x - r, 1 - y - r])
        return np.array(res)
    
    def nonoverlap_constraints(params):
        """Ensure no overlap between circles - more numerically stable version"""
        res = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                
                # Distance squared between centers
                dist_sq = compute_distance_squared(x1, y1, x2, y2)
                # Required distance squared to avoid overlap
                req_dist_sq = (r1 + r2)**2
                
                # Constraint: distance^2 >= (radius1 + radius2)^2
                # So: distance^2 - (radius1 + radius2)^2 >= 0
                # Add small epsilon to avoid numerical issues
                res.append(dist_sq - req_dist_sq + 1e-10)
        return np.array(res)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x bounds: [0, 1]
        bounds.append((0, 1))
        # y bounds: [0, 1]  
        bounds.append((0, 1))
        # r bounds: [0.001, 0.5] (small minimum to avoid degenerate cases)
        bounds.append((0.001, 0.5))
    
    # Create constraint objects
    cons = [
        {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
        {'type': 'ineq', 'fun': lambda p: nonoverlap_constraints(p)}
    ]
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = 0
    
    # Try with trust-constr which often works better for constrained problems
    try:
        result_trust = minimize(objective, initial_params, method='trust-constr', 
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if result_trust.success:
            current_sum = -result_trust.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result_trust
    except Exception as e:
        pass
    
    # Try with SLSQP as backup
    if best_result is None:
        try:
            result_slsqp = minimize(objective, initial_params, method='SLSQP', 
                                   bounds=bounds, constraints=cons, 
                                   options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8})
            
            if result_slsqp.success:
                current_sum = -result_slsqp.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_slsqp
        except Exception as e:
            pass
    
    # If still no good result, try a hybrid approach with manual refinement
    if best_result is None:
        # Use a simpler optimization approach first
        try:
            result_simple = minimize(objective, initial_params, method='L-BFGS-B',
                                    bounds=bounds, options={'maxiter': 300, 'ftol': 1e-6})
            
            if result_simple.success:
                current_sum = -result_simple.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_simple
        except Exception as e:
            pass
    
    # If still no good result, use the initial parameters with some refinement
    if best_result is None:
        final_params = initial_params
    else:
        final_params = best_result.x
    
    # Final refinement step with simple gradient descent approach
    try:
        # Simple local search around the solution
        refined_params = final_params.copy()
        step_size = 0.001
        
        # Run a few iterations of simple local optimization
        for _ in range(20):
            old_sum = -objective(refined_params)
            # Try small perturbations
            for i in range(n):
                # Perturb x coordinate
                test_params = refined_params.copy()
                test_params[3*i] += step_size
                if test_params[3*i] <= 1 and test_params[3*i] >= 0:
                    if all(containment_constraints(test_params) >= 0) and all(nonoverlap_constraints(test_params) >= 0):
                        new_sum = -objective(test_params)
                        if new_sum > old_sum:
                            refined_params = test_params
                            old_sum = new_sum
                
                # Perturb y coordinate
                test_params = refined_params.copy()
                test_params[3*i+1] += step_size
                if test_params[3*i+1] <= 1 and test_params[3*i+1] >= 0:
                    if all(containment_constraints(test_params) >= 0) and all(nonoverlap_constraints(test_params) >= 0):
                        new_sum = -objective(test_params)
                        if new_sum > old_sum:
                            refined_params = test_params
                            old_sum = new_sum
                
                # Perturb radius
                test_params = refined_params.copy()
                test_params[3*i+2] += step_size
                if test_params[3*i+2] <= 0.5 and test_params[3*i+2] >= 0.001:
                    if all(containment_constraints(test_params) >= 0) and all(nonoverlap_constraints(test_params) >= 0):
                        new_sum = -objective(test_params)
                        if new_sum > old_sum:
                            refined_params = test_params
                            old_sum = new_sum
        
        final_params = refined_params
    except Exception as e:
        pass
    
    # Construct the final result
    circles_final = np.zeros((n, 3))
    for i in range(n):
        circles_final[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
    
    return circles_final


# EVOLVE-BLOCK-END
