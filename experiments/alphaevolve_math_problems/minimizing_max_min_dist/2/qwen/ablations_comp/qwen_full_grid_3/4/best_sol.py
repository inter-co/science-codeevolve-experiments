# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric insights with multiple SLSQP optimizations.
    Inspired by the most successful implementations in the inspirations.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return 0
            
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to minimize (negative of ratio to maximize ratio)."""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize ratio
        return -ratio
    
    def generate_regular_polygon_initial_guess():
        """Generate initial configuration using regular 16-gon pattern."""
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 0.4  # Scaled appropriately
        
        # Regular polygon with perturbations (as in successful inspirations)
        points = np.zeros((n, 2))
        points[:, 0] = 0.5 + radius * np.cos(angles)
        points[:, 1] = 0.5 + radius * np.sin(angles)
        
        # Add moderate perturbation to break symmetry (as in Inspiration 3)
        np.random.seed(42)
        perturbation = 0.07 * np.random.randn(n, 2)
        points += perturbation
        points = np.clip(points, 0, 1)
        
        return points
    
    def generate_perturbed_hexagonal_initial_guess():
        """Generate initial configuration in hexagonal grid pattern with perturbations."""
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])
        
        points = np.array(points[:16])  # Ensure exactly 16 points
        
        # Add moderate perturbation to break symmetry
        np.random.seed(42)
        perturbations = 0.07 * np.random.randn(16, 2)
        points += perturbations
        
        # Ensure all points are within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    def optimize_with_slsqp(initial_points):
        """Use SLSQP optimization with tight tolerances."""
        # Flatten initial points
        x0 = initial_points.flatten()
        
        # Define bounds for all coordinates [0,1]
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Use SLSQP optimization (more reliable than other methods according to inspirations)
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                # Ensure all points are within bounds
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        
        # Return initial points if optimization fails
        return initial_points
    
    def multi_start_optimization():
        """Try multiple initial configurations and pick the best result."""
        best_points = None
        best_ratio = -float('inf')
        
        # Generate multiple initial guesses with different seeds for better exploration
        initial_guesses = []
        
        # Regular polygon with different seeds
        for seed in [42, 142, 242, 342]:
            np.random.seed(seed)
            initial_guesses.append(generate_regular_polygon_initial_guess())
        
        # Perturbed hexagonal grid with different seeds
        for seed in [42, 142, 242, 342]:
            np.random.seed(seed)
            initial_guesses.append(generate_perturbed_hexagonal_initial_guess())
        
        # Try multiple random restarts with different seeds (as in Inspiration 3)
        for restart in range(5):
            np.random.seed(42 + restart * 100)
            # Create random initial configuration
            points = np.random.uniform(0.05, 0.95, (16, 2))
            initial_guesses.append(points)
        
        # Optimize each initial guess
        for i, initial_points in enumerate(initial_guesses):
            try:
                # Local optimization using SLSQP
                optimized_points = optimize_with_slsqp(initial_points)
                
                # Evaluate quality
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception:
                continue
        
        # If no success, return the first guess
        if best_points is not None:
            return best_points
        else:
            # Fallback to regular polygon with seed 42
            return generate_regular_polygon_initial_guess()
    
    try:
        # Use multi-start approach for better results
        final_points = multi_start_optimization()
        return final_points
        
    except Exception as e:
        # Fallback to simple regular polygon if anything goes wrong
        return generate_regular_polygon_initial_guess()


# EVOLVE-BLOCK-END
