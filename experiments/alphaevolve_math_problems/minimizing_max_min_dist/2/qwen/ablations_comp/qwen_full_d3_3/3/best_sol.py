# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction and global optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
        
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        if dmax == 0:
            return 0
        
        return dmin / dmax

    def objective_function(points_flat):
        """Objective function to maximize (negative because scipy minimizes)."""
        # Reshape flat array back to 2D points
        points = points_flat.reshape(-1, 2)
        
        # Compute distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
        
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax == 0:
            return 0
        
        # Return negative ratio (since we want to maximize)
        return -dmin / dmax

    def generate_better_initialization():
        """Generate initial point configuration using proven good patterns."""
        # Method 1: Start with a square grid pattern with better spacing
        grid_size = 4
        points = []
        
        # Create a 4x4 grid with proper spacing
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * (1.0 / (grid_size - 1)) if grid_size > 1 else 0.5
                y = j * (1.0 / (grid_size - 1)) if grid_size > 1 else 0.5
                
                # Add small random perturbation to avoid symmetric traps
                x += np.random.uniform(-0.03, 0.03)
                y += np.random.uniform(-0.03, 0.03)
                
                points.append([x, y])
        
        # Ensure all points are within bounds [0,1] x [0,1]
        points = np.array(points)
        points = np.clip(points, 0, 1)
        
        return points

    # Strategy 1: Better initial configuration using proven patterns
    points = generate_better_initialization()
    
    # Strategy 2: Multi-stage optimization with enhanced robustness
    # Define bounds for each coordinate (must stay within [0,1])
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates
    
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Run 1: Enhanced Differential Evolution with more aggressive parameters
    try:
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=150,      # Even more iterations for better exploration
            popsize=25,       # Larger population for better diversity
            mutation=(0.9, 1), # Different mutation parameters for more exploration
            recombination=0.95, # Higher recombination for more mixing
            seed=42,
            tol=1e-14,        # Even tighter tolerance
            disp=False
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_ratio = compute_min_max_ratio(de_points)
            
            if de_ratio > best_ratio:
                best_ratio = de_ratio
                best_points = de_points.copy()
    except Exception:
        pass
    
    # Run 2: Multiple local optimizations with different starting points and methods
    # First, try a more thorough refinement with SLSQP
    try:
        if best_ratio > 0:
            refined_result = minimize(
                objective_function,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if refined_result.success:
                refined_points = refined_result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)
                
                if refined_ratio > best_ratio:
                    best_points = refined_points
                    best_ratio = refined_ratio
    except Exception:
        pass
    
    # Run 3: Multiple random restarts with different methods and perturbations
    restart_configs = [
        {'seed': 123, 'perturb': 0.05},
        {'seed': 456, 'perturb': 0.07}, 
        {'seed': 789, 'perturb': 0.03},
        {'seed': 999, 'perturb': 0.08},
        {'seed': 111, 'perturb': 0.04}
    ]
    
    for config in restart_configs:
        try:
            np.random.seed(config['seed'])
            # Random restart with different perturbation
            restart_points = best_points + np.random.normal(0, config['perturb'], best_points.shape)
            restart_points = np.clip(restart_points, 0, 1)
            
            # Try both SLSQP and L-BFGS-B for better results
            for method in ['SLSQP', 'L-BFGS-B']:
                try:
                    restart_result = minimize(
                        objective_function,
                        restart_points.flatten(),
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if restart_result.success:
                        restart_points = restart_result.x.reshape(-1, 2)
                        restart_points = np.clip(restart_points, 0, 1)
                        restart_ratio = compute_min_max_ratio(restart_points)
                        
                        if restart_ratio > best_ratio:
                            best_points = restart_points
                            best_ratio = restart_ratio
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # Run 4: Second DE attempt with different parameters for robustness
    try:
        de_result2 = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=20,
            mutation=(0.6, 1),
            recombination=0.8,
            seed=43,
            tol=1e-13,
            disp=False
        )
        
        if de_result2.success:
            de_points2 = de_result2.x.reshape(-1, 2)
            de_points2 = np.clip(de_points2, 0, 1)
            de_ratio2 = compute_min_max_ratio(de_points2)
            
            if de_ratio2 > best_ratio:
                best_ratio = de_ratio2
                best_points = de_points2.copy()
    except Exception:
        pass
    
    # Run 5: Final aggressive optimization with very tight tolerances
    try:
        if best_ratio > 0:
            final_result = minimize(
                objective_function,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if final_result.success:
                final_points = final_result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                final_ratio = compute_min_max_ratio(final_points)
                
                if final_ratio > best_ratio:
                    best_points = final_points
                    best_ratio = final_ratio
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
