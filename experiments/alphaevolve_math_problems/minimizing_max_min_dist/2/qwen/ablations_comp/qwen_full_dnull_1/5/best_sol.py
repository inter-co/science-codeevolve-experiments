# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach: geometric initialization + global optimization + local refinement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    def objective(params):
        """Objective function to maximize min/max distance ratio"""
        # Reshape parameters back into points
        points_current = params.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points_current)
        
        if len(distances) == 0:
            return -np.inf
            
        # Avoid division by zero
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return -np.inf
            
        # Return negative because we want to maximize
        return -min_dist / max_dist
    
    def constraint_func(params):
        """Ensure points stay within unit square"""
        points_current = params.reshape(-1, 2)
        # Return positive values where constraints are satisfied
        return np.concatenate([
            points_current[:, 0],      # x >= 0
            1 - points_current[:, 0],  # x <= 1
            points_current[:, 1],      # y >= 0
            1 - points_current[:, 1]   # y <= 1
        ])
    
    # Better geometric initialization using a more systematic approach
    # Create points on a circle with some randomness to break symmetry
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    radius = 0.4  # Adjust to fit within [0.1, 0.9] range
    points = []
    
    # Distribute points around a circle, then add small random perturbations
    for i, angle in enumerate(angles):
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        
        # Add small random perturbation to break symmetry
        x += (np.random.random() - 0.5) * 0.05
        y += (np.random.random() - 0.5) * 0.05
        
        points.append([x, y])
    
    points = np.array(points)
    
    # Ensure points are within bounds
    points = np.clip(points, 0, 1)
    
    # Try multiple optimization approaches for better results
    best_ratio = -np.inf
    best_points = points.copy()
    
    # Approach 1: Differential Evolution (global optimization) - Enhanced with more iterations
    try:
        bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=120,    # Increased iterations
            popsize=25,     # Larger population
            seed=42,
            disp=False,
            tol=1e-10       # Even tighter tolerance
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            # Check if this is better
            dists = pdist(de_points)
            if len(dists) > 0:
                min_d = np.min(dists)
                max_d = np.max(dists)
                if max_d > 0:
                    ratio = min_d / max_d
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = de_points.copy()
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
    
    # Approach 2: Local optimization with multiple restarts - Enhanced
    for restart in range(8):  # Increased from 5 to 8 restarts for better exploration
        # Start with a perturbed version of the best configuration so far
        if restart == 0:
            perturbed_points = points.copy()
        else:
            # Add more significant perturbation for later restarts
            perturbed_points = best_points + np.random.normal(0, 0.06, (16, 2))  # Slightly larger perturbation
        
        # Keep within bounds
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        try:
            result = minimize(
                objective,
                perturbed_points.flatten(),
                method='SLSQP',
                bounds=[(0, 1) for _ in range(32)],
                constraints={
                    'type': 'ineq',
                    'fun': constraint_func
                },
                options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerances
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                # Check if this is better
                dists = pdist(final_points)
                if len(dists) > 0:
                    min_d = np.min(dists)
                    max_d = np.max(dists)
                    if max_d > 0:
                        ratio = min_d / max_d
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
    
    # Approach 3: Additional refinement with L-BFGS-B for high precision
    try:
        # Try one more optimization with potentially better starting point
        refined_points = best_points.copy()
        result = minimize(
            objective,
            refined_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(32)],
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            # Check if this is better
            dists = pdist(final_points)
            if len(dists) > 0:
                min_d = np.min(dists)
                max_d = np.max(dists)
                if max_d > 0:
                    ratio = min_d / max_d
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
    except Exception as e:
        warnings.warn(f"L-BFGS-B optimization failed: {e}")
    
    # Final validation and return
    final_dists = pdist(best_points)
    if len(final_dists) > 0:
        min_d = np.min(final_dists)
        max_d = np.max(final_dists)
        if max_d <= 0:
            # If we still have issues, return the initial configuration
            return points
    else:
        # If no distances computed, return initial
        return points
    
    return best_points


# EVOLVE-BLOCK-END
