# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated approach combining multiple optimization strategies, better initialization,
    and efficient constraint handling.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)  # For reproducibility
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        if len(distances) == 0:
            return -1.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -1.0
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -d_min / d_max
    
    def create_hexagonal_initialization() -> np.ndarray:
        """Create a better initial configuration using hexagonal packing principles."""
        # Create 16 points in a hexagonal arrangement
        n = 16
        points = np.zeros((n, 2))
        
        # Arrange in a hexagonal pattern with better spacing
        # 4 rows with alternating column offsets for hexagonal packing
        row_positions = [0, 0.5, 1.0, 1.5]  # y positions - using larger spacing
        col_positions = [0, 0.5, 1.0, 1.5]  # x positions - using larger spacing
        
        idx = 0
        for i, y in enumerate(row_positions):
            offset = 0.25 if i % 2 == 1 else 0  # Offset every other row for better packing
            for j, x in enumerate(col_positions):
                if idx < n:
                    # Use random perturbation to avoid degenerate cases
                    x_perturbed = x + offset + 0.01 * np.random.randn() * 0.1
                    y_perturbed = y + 0.01 * np.random.randn() * 0.1
                    
                    # Clip to unit square
                    points[idx] = [np.clip(x_perturbed, 0, 1), np.clip(y_perturbed, 0, 1)]
                    idx += 1
        
        return points
    
    def create_grid_initialization() -> np.ndarray:
        """Create a grid-based initial configuration."""
        n = 16
        points = np.zeros((n, 2))
        
        # Create a 4x4 grid with better spacing
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < n:
                    # Add slight perturbation to avoid degenerate cases
                    x = j * spacing_x + 0.01 * np.sin(i + j)
                    y = i * spacing_y + 0.01 * np.cos(i + j)
                    points[idx] = [np.clip(x, 0, 1), np.clip(y, 0, 1)]
                    idx += 1
        
        return points
    
    def create_spherical_code_like_initialization() -> np.ndarray:
        """Create initialization inspired by spherical codes - distribute points more evenly."""
        # Create points in a way that mimics a spherical code in 2D
        points = np.zeros((16, 2))
        
        # Distribute points more regularly using spiral pattern with better coverage
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.9, 16)  # Like inspirations
        
        # Create a more even distribution
        for i in range(16):
            angle = angles[i]
            radius = radii[i]  # Vary radius to get better coverage
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points[i] = [np.clip(x, 0, 1), np.clip(y, 0, 1)]
        
        return points
    
    def create_fibonacci_spiral_initialization() -> np.ndarray:
        """Create initialization using Fibonacci spiral for even distribution."""
        points = np.zeros((16, 2))
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        
        for i in range(16):
            # Modified approach to make it more uniform
            y = 1 - (i / 15) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)  # radius at y
            
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Project to 2D with some randomness to break symmetry
            points[i] = [
                np.clip(0.5 + 0.4*x + 0.01 * np.random.randn(), 0, 1),
                np.clip(0.5 + 0.4*z + 0.01 * np.random.randn(), 0, 1)
            ]
        
        return points
    
    def create_random_initialization() -> np.ndarray:
        """Create a random initialization with better spread."""
        points = np.random.rand(16, 2)
        # Add some structure to avoid completely random points
        for i in range(16):
            points[i] = np.clip(points[i] + 0.02 * np.random.randn(2), 0, 1)
        return points
    
    # Try multiple initialization strategies with early stopping
    initial_strategies = [
        create_fibonacci_spiral_initialization,
        create_hexagonal_initialization,
        create_grid_initialization,
        create_spherical_code_like_initialization,
        create_random_initialization
    ]
    
    best_ratio = -float('inf')
    best_points = None
    start_time = time.time()
    
    # Run optimization with different initializations, but limit total time
    for strategy in initial_strategies:
        if time.time() - start_time > 55:  # Leave 5 seconds for final refinement
            break
            
        try:
            # Get initial configuration
            initial_points = strategy()
            
            # Flatten for optimization
            x0 = initial_points.flatten()
            
            # Define bounds for optimization (points must stay in [0,1] x [0,1])
            bounds = [(0, 1) for _ in range(32)]
            
            # Use more aggressive optimization parameters for faster convergence
            # Prioritize methods that work well for this type of problem
            # Reduce iterations to fit within time budget
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('TNC', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            # Run optimization with multiple methods
            for method, options in methods_and_options:
                if time.time() - start_time > 58:  # Leave 2 seconds for cleanup
                    break
                    
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            options=options,
                            callback=lambda x: None  # Suppress progress output
                        )
                    
                    if result.success:
                        # Extract optimized points
                        optimized_points = result.x.reshape(-1, 2)
                        
                        # Ensure final points are within bounds
                        optimized_points = np.clip(optimized_points, 0, 1)
                        
                        # Compute ratio for this solution
                        distances = pdist(optimized_points)
                        if len(distances) > 0:
                            dmin = np.min(distances)
                            dmax = np.max(distances)
                            
                            if dmax > 0:
                                ratio = dmin / dmax
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = optimized_points.copy()
                                
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # If no optimization worked, return a good default configuration
    if best_points is None:
        # Use fibonacci spiral initialization as final fallback (often works well)
        best_points = create_fibonacci_spiral_initialization()
    
    # Final refinement with a single optimization run using the best found solution
    if best_points is not None and time.time() - start_time < 58:
        try:
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Final optimization with L-BFGS-B for high precision
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
                )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                
                # Check if final result is better
                distances = pdist(final_points)
                if len(distances) > 0:
                    dmin = np.min(distances)
                    dmax = np.max(distances)
                    if dmax > 0:
                        ratio = dmin / dmax
                        if ratio > best_ratio:
                            best_points = final_points
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
