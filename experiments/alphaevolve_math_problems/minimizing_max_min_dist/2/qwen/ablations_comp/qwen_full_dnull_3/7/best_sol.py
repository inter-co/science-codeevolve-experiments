# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical initialization and advanced optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance among all point pairs."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max
    
    def objective(points_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to 2D points
        points = points_flat.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -np.inf
            
        # Return negative because we want to maximize
        return -d_min / d_max
    
    # Strategy 1: Create initial configuration using Fibonacci spiral (inspired by top performers)
    def create_fibonacci_initial():
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(16):
            # Fibonacci spiral approach
            theta = i * 2 * np.pi / golden_ratio
            r = np.sqrt(i / 15.0) if i > 0 else 0
            
            x = 0.5 + 0.4 * r * np.cos(theta)
            y = 0.5 + 0.4 * r * np.sin(theta)
            
            # Ensure within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            points.append([x, y])
        
        return np.array(points)
    
    # Strategy 2: Create initial configuration using hexagonal lattice
    def create_hexagonal_initial():
        # Arrange points in a hexagonal pattern (4x4 grid with offset rows)
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                x = j + (i % 2) * 0.5  # Offset every other row
                y = i * math.sqrt(3) / 2
                points.append([x, y])
        
        # Normalize to unit square [0,1] x [0,1]
        points = np.array(points)
        
        # Find bounding box
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        # Avoid division by zero
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # Scale to fit in [0,1] x [0,1]
        points[:, 0] = (points[:, 0] - x_min) / x_range
        points[:, 1] = (points[:, 1] - y_min) / y_range
        
        # Center in unit square
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        points[:, 0] += 0.5
        points[:, 1] += 0.5
        
        # Ensure all points are within [0,1] x [0,1]
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    # Strategy 3: Simulated Annealing optimization for global search
    def simulated_annealing(initial_points, max_iter=10000, seed=42):
        np.random.seed(seed)
        
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Simulated Annealing parameters from best practices
        initial_temp = 0.1
        final_temp = 1e-6
        alpha = 0.995
        
        temp = initial_temp
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            new_points = current_points.copy()
            point_idx = np.random.randint(0, len(current_points))
            
            # Perturb one point with adaptive step size
            step_size = 0.02 / (1.0 + iteration / max_iter * 10)
            new_points[point_idx] = current_points[point_idx] + np.random.normal(0, step_size, 2)
            
            # Keep points within [0,1] bounds
            new_points = np.clip(new_points, 0, 1)
            
            # Calculate new ratio
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < math.exp((new_ratio - current_ratio) / temp):
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best solution if improved
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            
            # Cool down temperature
            temp = max(final_temp, temp * alpha)
            
            # Early stopping if temperature gets too low
            if temp < final_temp:
                break
        
        return best_points, best_ratio
    
    # Strategy 4: Multi-start approach with diverse strategies
    def multi_start_optimization():
        best_points = None
        best_ratio = -np.inf
        
        # Strategy A: Fibonacci initialization + differential evolution (primary)
        try:
            fib_points = create_fibonacci_initial()
            x0 = fib_points.flatten()
            
            # Use differential evolution for global optimization with better parameters
            result = differential_evolution(
                objective,
                bounds=[(0, 1) for _ in range(32)],
                seed=42,
                maxiter=100,      # Increased iterations for better convergence
                popsize=20,       # Larger population
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass
        
        # Strategy B: Hexagonal initialization + differential evolution (secondary)
        try:
            hex_points = create_hexagonal_initial()
            x0 = hex_points.flatten()
            
            # Use differential evolution for global optimization
            result = differential_evolution(
                objective,
                bounds=[(0, 1) for _ in range(32)],
                seed=100,
                maxiter=80,       # Fewer iterations since hexagonal is already structured
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass
        
        # Strategy C: Fibonacci + Simulated Annealing (for escaping local optima)
        try:
            fib_points = create_fibonacci_initial()
            sa_points, sa_ratio = simulated_annealing(fib_points, max_iter=12000, seed=42)
            if sa_ratio > best_ratio:
                best_ratio = sa_ratio
                best_points = sa_points.copy()
        except Exception:
            pass
        
        # Strategy D: Hexagonal + Simulated Annealing
        try:
            hex_points = create_hexagonal_initial()
            sa_points, sa_ratio = simulated_annealing(hex_points, max_iter=12000, seed=123)
            if sa_ratio > best_ratio:
                best_ratio = sa_ratio
                best_points = sa_points.copy()
        except Exception:
            pass
        
        # Strategy E: Random start + local optimization (final refinement)
        try:
            np.random.seed(42)
            random_points = np.random.rand(16, 2)
            x0 = random_points.flatten()
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass
        
        # Strategy F: Multiple random restarts to ensure robustness
        for restart in range(3):
            try:
                np.random.seed(200 + restart)
                random_points = np.random.rand(16, 2)
                x0 = random_points.flatten()
                
                # Use differential evolution from random start
                result = differential_evolution(
                    objective,
                    bounds=[(0, 1) for _ in range(32)],
                    seed=200 + restart,
                    maxiter=60,
                    popsize=10,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
            except Exception:
                continue
        
        return best_points, best_ratio
    
    # Main optimization process
    start_time = time.time()
    
    # Run multi-start optimization
    best_points, best_ratio = multi_start_optimization()
    
    # If no good solution found, fallback to Fibonacci initialization
    if best_points is None:
        best_points = create_fibonacci_initial()
    
    # Final refinement with local optimization if time permits
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < 55:  # Leave buffer for final processing
            try:
                x0 = best_points.flatten()
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 2500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_ratio = compute_min_max_ratio(final_points)
                    if final_ratio > best_ratio:
                        return final_points
            except Exception:
                pass
    
    return best_points


# EVOLVE-BLOCK-END
