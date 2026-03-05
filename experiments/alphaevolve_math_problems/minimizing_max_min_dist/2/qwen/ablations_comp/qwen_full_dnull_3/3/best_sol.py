# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated approach combining multiple initialization strategies and optimization methods.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return ratio, handle division by zero
        return d_min / d_max if d_max > 0 else 0
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -1.0
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -d_min / d_max
    
    def create_hexagonal_initialization() -> np.ndarray:
        """Create a better initial configuration using hexagonal packing principles."""
        # Create 16 points in a hexagonal arrangement with precise spacing
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
    
    def create_mathematical_initialization() -> np.ndarray:
        """Create initial points using mathematical principles inspired by optimal configurations."""
        # Use a configuration with outer ring and inner cluster (inspiration from top performers)
        points = []
        
        # Outer ring: 12 points evenly distributed
        for i in range(12):
            angle = 2 * np.pi * i / 12
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        # Inner cluster: 4 points forming a square
        inner_radius = 0.15
        for i in range(4):
            angle = i * np.pi/2 + np.pi/4  # Square pattern
            x = 0.5 + inner_radius * np.cos(angle)
            y = 0.5 + inner_radius * np.sin(angle)
            points.append([x, y])
        
        return np.array(points[:16])
    
    # Try multiple initialization strategies like INSPIRATION 3
    initial_strategies = [
        create_mathematical_initialization,
        create_hexagonal_initialization,
        create_grid_initialization,
        create_spherical_code_like_initialization
    ]
    
    best_ratio = -float('inf')
    best_points = None
    
    # Run optimization with different initializations (like INSPIRATION 3)
    for strategy in initial_strategies:
        try:
            # Get initial configuration
            initial_points = strategy()
            
            # Flatten for optimization
            x0 = initial_points.flatten()
            
            # Define bounds for optimization (points must stay in [0,1] x [0,1])
            bounds = [(0, 1) for _ in range(32)]
            
            # Use multiple optimization methods with tighter tolerances for better precision
            # Based on what works best in INSPIRATION 3
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 2500, 'ftol': 1e-16, 'gtol': 1e-16}),
                ('TNC', {'maxiter': 1200, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('SLSQP', {'maxiter': 1200, 'ftol': 1e-14, 'gtol': 1e-14})
            ]
            
            for method, options in methods_and_options:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            options=options
                        )
                    
                    if result.success:
                        # Extract optimized points
                        optimized_points = result.x.reshape(-1, 2)
                        
                        # Ensure final points are within bounds
                        optimized_points = np.clip(optimized_points, 0, 1)
                        
                        # Compute ratio for this solution
                        ratio = compute_min_max_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # If no optimization worked, return a good default configuration
    if best_points is None:
        # Use mathematical initialization as final fallback
        best_points = create_mathematical_initialization()
    
    return best_points


# EVOLVE-BLOCK-END
