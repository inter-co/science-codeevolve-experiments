# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions with advanced optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Computes the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0.0
    
    def objective_for_scipy(x):
        points = x.reshape(-1, 2)
        return -compute_min_max_ratio(points)
    
    # Enhanced circle initialization that works well (inspired by inspirations)
    def circle_initialization():
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Create points on a circle of radius 0.4 centered at (0.5, 0.5)
        points = np.zeros((n, 2))
        for i, angle in enumerate(angles):
            points[i, 0] = 0.5 + 0.4 * np.cos(angle)
            points[i, 1] = 0.5 + 0.4 * np.sin(angle)
        
        # Add moderate perturbations for better escape from local minima
        points += np.random.normal(0, 0.02, (n, 2))
        
        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Simulated Annealing with better parameters (from inspiration 2)
    def simulated_annealing(initial_points, max_iter=50000):
        """Perform simulated annealing to optimize point configuration."""
        # Initialize with good starting configuration
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Optimized parameters for time efficiency and better convergence
        temp = 1.0
        cooling_rate = 0.995
        min_temp = 1e-8  # Tighter minimum temperature
        max_iter = max_iter
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Store history for adaptive cooling
        ratios_history = []
        stagnant_count = 0  # Track stagnation for early stopping
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = random.randint(0, 15)
            
            # Perturb the selected point with adaptive magnitude
            perturbation_magnitude = max(0.005, temp * 0.03)  # Larger early on
            perturbation = np.random.normal(0, perturbation_magnitude, 2)
            neighbor_points[point_idx] += perturbation
            
            # Keep points within bounds with tighter bounds
            neighbor_points[:, 0] = np.clip(neighbor_points[:, 0], 0.05, 0.95)
            neighbor_points[:, 1] = np.clip(neighbor_points[:, 1], 0.05, 0.95)
            
            # Compute new ratio
            new_ratio = compute_min_max_ratio(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or random.random() < np.exp((new_ratio - current_ratio) / temp):
                current_points = neighbor_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = new_ratio
                    stagnant_count = 0  # Reset stagnation counter
                else:
                    stagnant_count += 1
            else:
                stagnant_count += 1
            
            # Adaptive cooling with better history tracking
            ratios_history.append(current_ratio)
            if len(ratios_history) > 100:
                ratios_history.pop(0)
            
            # Cool temperature
            temp *= cooling_rate
            
            if temp < min_temp:
                temp = min_temp
                
            # Occasionally reset temperature if stuck (more aggressive)
            if iteration % 1000 == 0 and len(ratios_history) > 10:
                if len(ratios_history) >= 2:
                    recent_improvement = ratios_history[-1] - ratios_history[0]
                    if recent_improvement < 1e-10:
                        temp = max(0.02, temp * 0.5)  # Reset temperature to escape local minima
            
            # Early stopping for stagnation
            if stagnant_count > 10000:
                break
        
        return best_points, best_ratio
    
    # Try multiple restarts with circle initialization (inspired by inspiration 2)
    best_ratio = -np.inf
    best_points = None
    
    # Try 20 restarts with different seeds
    for restart in range(20):
        np.random.seed(42 + restart)
        
        # Start with circle initialization
        initial_points = circle_initialization()
        
        # Apply simulated annealing first (more effective than pure local optimization)
        sa_points, sa_ratio = simulated_annealing(initial_points, max_iter=50000)
        
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
        
        # Then try local optimization on this result
        try:
            # Flatten the points for scipy optimization
            initial_flat = sa_points.flatten()
            
            # Use SLSQP with tight tolerances (like inspiration 1)
            result = minimize(
                objective_for_scipy,
                initial_flat,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-12}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points[:, 0] = np.clip(refined_points[:, 0], 0, 1)
                refined_points[:, 1] = np.clip(refined_points[:, 1], 0, 1)
                
                refined_ratio = compute_min_max_ratio(refined_points)
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
                    
        except Exception:
            continue
    
    # If no good solution found, fallback to circle initialization
    if best_points is None:
        best_points = circle_initialization()
    
    # Apply final aggressive refinement to the best solution found
    try:
        if best_points is not None:
            refined_points = best_points.copy()
            best_final_ratio = best_ratio
            
            # 1. First refine with very tight tolerances (like Inspiration 1)
            final_x = refined_points.flatten()
            result = minimize(
                objective_for_scipy,
                final_x,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_final_ratio:
                    best_final_ratio = ratio
                    refined_points = final_points
            
            # 2. Try L-BFGS-B with even tighter tolerances
            try:
                final_x = refined_points.flatten()
                result = minimize(
                    objective_for_scipy,
                    final_x,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points[:, 0] = np.clip(final_points[:, 0], 0, 1)
                    final_points[:, 1] = np.clip(final_points[:, 1], 0, 1)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        best_final_ratio = ratio
                        refined_points = final_points
            except:
                pass
            
            best_points = refined_points
                        
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
