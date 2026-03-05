# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses improved initialization and enhanced optimization strategy based on inspiration programs.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    def compute_distances(points):
        """Compute pairwise distances and return min/max ratio"""
        if len(points) < 2:
            return 0, 0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0, 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min, d_max
    
    def get_better_initial_configuration():
        """Better initialization based on cube vertices and strategic points"""
        # Start with vertices of a cube (8 points) 
        points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    points.append([i, j, k])
        
        # Add points along axes and diagonals for better distribution
        points.extend([
            [1.5, 0, 0], [-1.5, 0, 0],  # x-axis
            [0, 1.5, 0], [0, -1.5, 0],  # y-axis  
            [0, 0, 1.5], [0, 0, -1.5],  # z-axis
            [0.707, 0.707, 0.707], [-0.707, -0.707, -0.707],  # diagonals
            [0.707, -0.707, 0.707], [-0.707, 0.707, -0.707]   # more diagonals
        ])
        
        # Keep only first 14 points and normalize
        points = np.array(points[:14], dtype=float)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points
    
    def objective_function(x):
        """Objective function to minimize (negative of ratio)"""
        points = x.reshape(-1, 3)
        d_min, d_max = compute_distances(points)
        
        # Avoid division by zero or invalid cases
        if d_max <= 1e-12:
            return 1e10  # Large penalty for invalid configurations
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
    def sphere_constraint(x):
        """Constraint function for unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should be zero for points on unit sphere
    
    # Generate better initial configuration
    initial_points = get_better_initial_configuration()
    
    best_ratio = -np.inf
    best_points = None
    
    # Use more sophisticated restart strategy with better parameter tuning
    # Based on inspiration programs, using fewer but more effective restarts
    restart_configs = [
        (0.01, 500),  # Small perturbation, moderate iterations
        (0.02, 750),  # Medium perturbation, more iterations  
        (0.03, 1000), # Larger perturbation, many iterations
        (0.02, 1000), # Standard config for final check
    ]
    
    # Use 4 restarts instead of 9 for better time management
    for restart_idx, (perturbation, max_iter) in enumerate(restart_configs):
        np.random.seed(42 + restart_idx)
        
        # Slightly perturb initial solution for each restart
        x_start = initial_points.flatten() + np.random.normal(0, perturbation, len(initial_points.flatten()))
        # Ensure points stay on unit sphere after perturbation
        x_start = x_start.reshape(14, 3)
        norms = np.linalg.norm(x_start, axis=1, keepdims=True)
        x_start = (x_start / norms).flatten()
        
        try:
            # Try multiple optimization methods for better convergence
            # Use both L-BFGS-B and SLSQP for better robustness
            methods = ['L-BFGS-B', 'SLSQP']
            for method in methods:
                if best_ratio > 0.48:  # Early exit if we're close to benchmark
                    break
                    
                result = minimize(
                    objective_function,
                    x_start,
                    method=method,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result.success:
                    optimized_points = result.x.reshape(14, 3)
                    # Evaluate the result
                    d_min, d_max = compute_distances(optimized_points)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
        except Exception:
            continue
    
    # Final refinement with highest precision using L-BFGS-B
    if best_points is not None:
        try:
            # One final high-precision optimization with L-BFGS-B
            final_result = minimize(
                objective_function,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
                tol=1e-14
            )
            
            if final_result.success:
                final_points = final_result.x.reshape(-1, 3)
                d_min, d_max = compute_distances(final_points)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = final_points.copy()
        except Exception:
            pass
    
    # Return best solution found or fallback to initial configuration
    if best_points is None:
        return initial_points
    
    return best_points


# EVOLVE-BLOCK-END
