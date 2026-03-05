# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import random
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multiple optimization strategies,
    and simulated annealing for better global optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the min/max distance ratio for given points."""
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def constraint_func(x):
        """Constraint to keep points within unit square"""
        points = x.reshape(-1, 2)
        # Keep all points in [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])
    
    def generate_better_initial_config():
        """Generate a better initial configuration using golden ratio-inspired arrangement"""
        # Create points arranged in a pattern inspired by optimal point distributions
        points = []
        # Arrange in a more uniform way that avoids clustering
        for i in range(4):
            for j in range(4):
                # Use golden ratio spacing for better distribution
                x = (i + 0.5 + 0.3 * np.sin(i * 0.5)) / 4.0
                y = (j + 0.5 + 0.3 * np.cos(j * 0.5)) / 4.0
                points.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
        return np.array(points)
    
    def perturb_points(points: np.ndarray, temperature: float) -> np.ndarray:
        """Create a perturbed version of the point configuration using simulated annealing approach"""
        new_points = points.copy()
        # Randomly select one point to move
        idx = random.randint(0, len(points) - 1)
        # Add small random displacement
        displacement = np.random.normal(0, temperature * 0.1, 2)
        new_points[idx] += displacement
        # Keep points within [0,1] bounds
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    # Initialize with a better starting configuration
    np.random.seed(42)
    
    # Start with a better initial configuration
    points = generate_better_initial_config()
    
    # Add some randomness to avoid degenerate cases
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, 0, 1)
    
    # Multi-start optimization with different strategies
    best_ratio = -float('inf')
    best_points = points.copy()
    
    # Try multiple optimization strategies with more iterations
    for strategy in ['L-BFGS-B', 'SLSQP']:
        # Multiple random restarts for this strategy
        for _ in range(5):  # Increased number of restarts
            # Randomize initial points slightly
            init_points = points + np.random.normal(0, 0.02, points.shape)
            init_points = np.clip(init_points, 0, 1)
            
            # Flatten for optimization
            x0 = init_points.flatten()
            
            # Define constraints
            cons = {'type': 'ineq', 'fun': constraint_func}
            
            # Optimize with higher iteration limits
            try:
                result = minimize(
                    objective,
                    x0,
                    method=strategy,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    ratio = calculate_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
    
    # Apply enhanced simulated annealing refinement for final improvement
    # This helps escape local optima that standard optimizers might get stuck in
    current_points = best_points.copy()
    current_ratio = calculate_min_max_ratio(current_points)
    
    # Enhanced Simulated Annealing parameters with better cooling schedule
    initial_temp = 0.2
    final_temp = 0.0001
    alpha = 0.97  # Slightly faster cooling
    max_iter = 8000  # More iterations for better search
    
    temp = initial_temp
    
    for iteration in range(max_iter):
        # Generate neighbor solution
        new_points = perturb_points(current_points, temp)
        new_ratio = calculate_min_max_ratio(new_points)
        
        # Accept or reject the new solution
        if new_ratio > current_ratio:
            current_points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
        else:
            # Accept with probability based on temperature
            if temp > 1e-10:  # Avoid division by zero
                acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
                if random.random() < acceptance_prob:
                    current_points = new_points
                    current_ratio = new_ratio
        
        # Cool down temperature with exponential decay
        temp = max(final_temp, temp * alpha)
        
        # Early stopping if improvement is minimal
        if iteration > 2000 and abs(current_ratio - best_ratio) < 1e-9:
            break
    
    return best_points


# EVOLVE-BLOCK-END
