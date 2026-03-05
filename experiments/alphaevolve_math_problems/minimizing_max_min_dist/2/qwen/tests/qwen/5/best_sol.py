# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random
import warnings
import time
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with aggressive optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Computes the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0.0
    
    # Strategy 1: Hexagonal arrangement (inspired by inspiration 2)
    def generate_hexagonal_initialization():
        points = []
        # Arrange in 4 rows with alternating positions
        for i in range(4):
            offset = 0.5 if i % 2 == 1 else 0.0
            for j in range(4):
                x = j + offset
                y = i * 0.866  # sqrt(3)/2 spacing
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to [0,1] range
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        
        # Scale to fit nicely in [0.05, 0.95] square and add small random noise
        points[:, 0] *= 0.9
        points[:, 1] *= 0.9
        points[:, 0] += 0.05
        points[:, 1] += 0.05
        
        # Add small random noise to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        
        # Ensure all points are within [0.05, 0.95] bounds
        points[:, 0] = np.clip(points[:, 0], 0.05, 0.95)
        points[:, 1] = np.clip(points[:, 1], 0.05, 0.95)
        
        return points
    
    # Strategy 2: Golden spiral pattern (inspired by inspiration 2)
    def generate_golden_spiral_initialization():
        points = []
        # Use golden spiral pattern for better distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(16):
            radius = 0.4 * np.sqrt(i / 15.0)  # Radial distribution
            angle = i * golden_angle  # Angular distribution
            
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add controlled noise to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        
        # Ensure all points are within [0.05, 0.95] bounds
        points[:, 0] = np.clip(points[:, 0], 0.05, 0.95)
        points[:, 1] = np.clip(points[:, 1], 0.05, 0.95)
        
        return points
    
    # Strategy 3: Regular polygon configuration (inspired by inspiration 2)
    def generate_regular_polygon_initialization():
        points = []
        # Create points around a circle
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add small random noise to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Ensure all points are within [0.05, 0.95] bounds
        points[:, 0] = np.clip(points[:, 0], 0.05, 0.95)
        points[:, 1] = np.clip(points[:, 1], 0.05, 0.95)
        
        return points
    
    # Strategy 4: Fibonacci sphere approach (inspired by inspiration 2)
    def generate_fibonacci_sphere_initialization():
        points = []
        for i in range(16):
            # Golden angle increment
            phi = np.arccos(-1 + (2 * i) / 15.0)
            theta = np.sqrt(16 * np.pi) * phi
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            
            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    # Strategy 5: Circle initialization (inspired by inspiration 1)
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
    
    # Try multiple initialization strategies and select the best one
    initial_strategies = [
        generate_hexagonal_initialization,
        generate_golden_spiral_initialization,
        generate_regular_polygon_initialization,
        generate_fibonacci_sphere_initialization,
        circle_initialization
    ]
    
    best_initial_ratio = -np.inf
    best_initial_points = None
    
    for strategy in initial_strategies:
        try:
            points = strategy()
            ratio = compute_min_max_ratio(points)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_initial_points = points.copy()
        except Exception:
            continue
    
    # If no good initial points found, fall back to circle initialization
    if best_initial_points is None:
        best_initial_points = circle_initialization()
    
    # Main optimization loop with multiple restarts
    best_ratio = -np.inf
    best_points = None
    
    # Try 30 restarts with different initialization strategies (more thorough than before)
    for restart in range(30):
        np.random.seed(42 + restart)
        
        # Alternate between different initialization methods for diversity
        if restart < 6:
            initial_points = circle_initialization()
        elif restart < 12:
            initial_points = generate_hexagonal_initialization()
        elif restart < 18:
            initial_points = generate_golden_spiral_initialization()
        elif restart < 24:
            initial_points = generate_regular_polygon_initialization()
        else:
            initial_points = generate_fibonacci_sphere_initialization()
        
        # Apply simulated annealing first (more effective than pure local optimization)
        sa_points, sa_ratio = simulated_annealing(initial_points, max_iter=50000)
        
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
        
        # Then try local optimization on this result
        try:
            def objective_for_scipy(x):
                points = x.reshape(-1, 2)
                return -compute_min_max_ratio(points)
            
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
            def objective_for_scipy(x):
                points = x.reshape(-1, 2)
                return -compute_min_max_ratio(points)
            
            # Try one final refinement with L-BFGS-B for better precision
            final_x = best_points.flatten()
            result = minimize(
                objective_for_scipy,
                final_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points[:, 0] = np.clip(final_points[:, 0], 0, 1)
                final_points[:, 1] = np.clip(final_points[:, 1], 0, 1)
                
                final_ratio = compute_min_max_ratio(final_points)
                if final_ratio > best_ratio:
                    best_points = final_points
                    
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
