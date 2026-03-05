# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies, advanced optimization,
    and time-aware execution to beat the benchmark.

    Strategy:
    1. Multiple high-quality initialization strategies based on mathematical constructions
    2. Multi-start optimization with different algorithms for robustness
    3. Time-aware execution that prioritizes quality over speed when possible
    4. Comprehensive validation and boundary handling

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set time limit to ensure we don't exceed 60 seconds
    start_time = time.time()
    timeout = 55  # Leave 5 seconds for final processing
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Handle edge case where all points are coincident
        if max_dist == 0:
            return 0
        
        return min_dist / max_dist
    
    def initialize_golden_spiral():
        """Initialize points using golden spiral for good distribution"""
        n = 16
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(n):
            theta = 2 * np.pi * i / phi
            r = np.sqrt(i / (n - 1)) if n > 1 else 0  # Radial distribution
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        
        # Normalize to [0,1] x [0,1] 
        points = np.array(points)
        if len(points) > 0:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
            
            # Shift to center and scale appropriately
            points[:, 0] = 0.5 + 0.4 * (points[:, 0] - 0.5)
            points[:, 1] = 0.5 + 0.4 * (points[:, 1] - 0.5)
        
        # Add slight random perturbations to escape local minima
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        
        return points
    
    def initialize_hexagonal_arrangement():
        """Initialize points in a hexagonal lattice pattern"""
        # Create a hexagonal arrangement
        rows = 4
        cols = 4
        
        points = []
        for i in range(rows):
            for j in range(cols):
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3) / 2
                points.append([x, y])
        
        # Take first 16 points
        points = np.array(points[:16])
        
        # Normalize to reasonable scale
        if len(points) > 0:
            # Scale to fit well in unit square
            ranges = np.max(points, axis=0) - np.min(points, axis=0)
            if np.any(ranges > 0):
                points = points / np.max(ranges) * 0.8
            
            # Center around origin
            points = points - np.mean(points, axis=0)
            
            # Shift to [0,1] range
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            if np.any(maxs - mins > 0):
                points = (points - mins) / (maxs - mins) * 0.9 + 0.05
        
        # Add slight random perturbations to escape local minima
        np.random.seed(42)
        points += np.random.normal(0, 0.02, points.shape)
        
        return points
    
    def initialize_regular_polygon():
        """Initialize points on a regular polygon (circle) with 16 vertices"""
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        # Scale to fit nicely in unit square
        points = points * 0.4 + 0.5  # Center at (0.5, 0.5) with radius 0.4
        
        # Add slight random perturbations to escape local minima
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        
        return points
    
    def initialize_grid_with_noise():
        """Initialize points in a regular grid pattern with random noise"""
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0, j/3.0])
        
        points = np.array(points)
        
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        
        # Normalize to [0,1] range
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        ranges = maxs - mins
        if np.any(ranges > 0):
            points = (points - mins) / ranges * 0.9 + 0.05
        
        return points
    
    def initialize_energy_based():
        """Initialize points using an energy-based approach mimicking repulsive forces"""
        # Start with a regular configuration and apply energy minimization concept
        n = 16
        
        # Create initial configuration using a combination of hexagonal and grid
        points = []
        
        # Hexagonal grid with slight randomness
        for i in range(4):
            for j in range(4):
                x = j + (i % 2) * 0.5 + np.random.normal(0, 0.02)
                y = i * np.sqrt(3) / 2 + np.random.normal(0, 0.02)
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to fit well in unit square
        if len(points) > 0:
            ranges = np.max(points, axis=0) - np.min(points, axis=0)
            if np.any(ranges > 0):
                points = points / np.max(ranges) * 0.8
            
            # Center around origin
            points = points - np.mean(points, axis=0)
            
            # Shift to [0,1] range
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            if np.any(maxs - mins > 0):
                points = (points - mins) / (maxs - mins) * 0.9 + 0.05
        
        return points
    
    def initialize_improved_spherical_code():
        """Improved spherical code approach with better parameter selection"""
        # Create a more sophisticated distribution
        points = []
        
        # Add points in a way that distributes them more evenly
        for i in range(4):
            for j in range(4):
                # Create a grid with some offset for better distribution
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                points.append([x, y])
        
        # Make sure we have exactly 16 points
        points = np.array(points[:16])
        
        # Apply some transformation to make distribution more uniform
        # This is a simple but effective way to distribute points more evenly
        np.random.seed(42)
        noise = np.random.normal(0, 0.03, points.shape)
        points = points + noise
        
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        return points
    
    def initialize_regular_polygon_arrangement():
        """Initialize points in a regular 16-gon arrangement (mathematical construction)"""
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Place points regularly on a circle
        np.random.seed(42)
        radius = 0.4
        points = np.array([[radius * np.cos(angle),
                           radius * np.sin(angle)] 
                          for angle in angles])
        
        # Add slight random perturbations to escape local minima
        points += np.random.normal(0, 0.01, points.shape)
        
        # Normalize to fit in [0,1] x [0,1] and center
        points[:, 0] = (points[:, 0] + 0.5) * 0.8 + 0.1  # Scale and shift x
        points[:, 1] = (points[:, 1] + 0.5) * 0.8 + 0.1  # Scale and shift y
        
        return points
    
    def initialize_fibonacci_spiral():
        """Initialize points using Fibonacci spiral for excellent distribution"""
        n = 16
        points = []
        
        # Fibonacci spiral approach
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Angular position
            theta = i * 2 * np.pi / golden_ratio
            
            # Radial position (spiral)
            r = np.sqrt(i / (n - 1)) if n > 1 else 0
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        
        # Normalize to [0,1] x [0,1]
        points = np.array(points)
        if len(points) > 0:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
            
            # Scale and shift to center
            points[:, 0] = 0.5 + 0.4 * (points[:, 0] - 0.5)
            points[:, 1] = 0.5 + 0.4 * (points[:, 1] - 0.5)
        
        # Add slight random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        
        return points
    
    def optimize_multiple_restarts(initial_points, remaining_time):
        """Apply multiple optimization restarts to find the best configuration"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Try multiple optimization runs with different strategies
        max_restarts = 6 if remaining_time > 10 else 4  # Reduce restarts if time is short
        
        for restart in range(max_restarts):  # More restarts for robustness
            if time.time() - start_time > timeout:
                break
                
            try:
                # Apply different optimization methods based on restart number
                if restart == 0:
                    # First restart: L-BFGS-B (fast local optimization)
                    optimized_points = optimize_single_run(initial_points, 'L-BFGS-B', remaining_time/4)
                elif restart == 1:
                    # Second restart: TNC (good for constrained problems)
                    optimized_points = optimize_single_run(initial_points, 'TNC', remaining_time/4)
                elif restart == 2:
                    # Third restart: SLSQP (good general purpose)
                    optimized_points = optimize_single_run(initial_points, 'SLSQP', remaining_time/4)
                elif restart == 3:
                    # Fourth restart: trust-constr (more robust)
                    optimized_points = optimize_single_run(initial_points, 'trust-constr', remaining_time/4)
                else:
                    # Fifth restart: Different random initialization with larger perturbation
                    np.random.seed(1000 + restart)
                    perturbed_initial = initial_points + np.random.normal(0, 0.05, initial_points.shape)
                    optimized_points = optimize_single_run(perturbed_initial, 'L-BFGS-B', remaining_time/4)
                
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Optimization restart {restart} failed: {str(e)}")
                continue
        
        return best_points
    
    def optimize_single_run(initial_points, method='L-BFGS-B', max_time: float = 10.0):
        """Apply a single, robust optimization run to find the best configuration"""
        points = initial_points.copy()
        
        # Define bounds for optimization (points must stay in [0,1] x [0,1])
        bounds = [(0, 1), (0, 1)] * len(points)
        
        # Convert to flattened array for scipy optimization
        flat_points = points.flatten()
        
        # Define objective function to maximize min/max ratio
        def objective(flat):
            # Reshape back to 2D array
            reshaped = flat.reshape(-1, 2)
            # Minimize negative of ratio (since scipy minimizes)
            return -compute_min_max_ratio(reshaped)
        
        # Use specified optimizer with reasonable parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Set maxiter based on time constraints
            max_iter = min(2000, int(max_time * 150))  # More iterations for better convergence
            
            result = minimize(
                objective, 
                flat_points, 
                method=method, 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            return optimized_points
        
        # If optimization fails, return the original points
        return points
    
    # Use multiple initialization strategies and select the best
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Golden spiral initialization (inspired by inspiration 1)
    points1 = initialize_golden_spiral()
    
    # Strategy 2: Hexagonal lattice with perturbations (from inspiration 1)
    points2 = initialize_hexagonal_arrangement()
    
    # Strategy 3: Regular polygon (circle) with perturbations (from inspiration 1)
    points3 = initialize_regular_polygon()
    
    # Strategy 4: Grid with noise (from inspiration 1)
    points4 = initialize_grid_with_noise()
    
    # Strategy 5: Energy-based initialization using potential field model (novel approach)
    points5 = initialize_energy_based()
    
    # Strategy 6: Improved spherical code approach (from inspiration 2)
    points6 = initialize_improved_spherical_code()
    
    # Strategy 7: Regular 16-gon with noise (enhanced mathematical construction)
    points7 = initialize_regular_polygon_arrangement()
    
    # Strategy 8: Fibonacci spiral approach (highly structured)
    points8 = initialize_fibonacci_spiral()
    
    # Test all initializations with optimization
    initial_strategies = [points1, points2, points3, points4, points5, points6, points7, points8]
    
    # Prioritize strategies that are likely to produce better results
    strategy_weights = [1.2, 1.0, 1.0, 1.0, 1.1, 1.3, 1.1, 1.2]  # Weight higher for better approaches
    
    for i, (initial_points, weight) in enumerate(zip(initial_strategies, strategy_weights)):
        if time.time() - start_time > timeout:
            break
            
        try:
            # Optimize each initialization with multiple restarts for robustness
            optimized_points = optimize_multiple_restarts(initial_points, timeout - (time.time() - start_time))
            ratio = compute_min_max_ratio(optimized_points)
            
            # Adjust ratio by weight to favor better strategies
            weighted_ratio = ratio * weight
            
            if weighted_ratio > best_ratio:
                best_ratio = weighted_ratio
                best_points = optimized_points.copy()
        except Exception as e:
            warnings.warn(f"Strategy {i} failed: {str(e)}")
            continue
    
    # If no optimization worked, return the best initialization directly
    if best_points is None:
        return initial_strategies[0]
    
    # Ensure points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
