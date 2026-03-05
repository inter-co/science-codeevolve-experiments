# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and multiple optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        return dmin / dmax if dmax > 0 else 0.0
    
    def objective_function(x_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return -np.inf
            
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Handle edge case where all points are coincident
        if max_dist == 0:
            return -np.inf
            
        # Return negative because we want to maximize
        return -min_dist / max_dist
    
    def initialize_fibonacci_points():
        """Initialize points using Fibonacci spiral for good distribution."""
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
    
    def initialize_hexagonal_points():
        """Initialize points using a hexagonal grid pattern."""
        points = []
        rows, cols = 4, 4
        
        # Generate hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                # Offset odd rows
                x_offset = j + 0.5 * (i % 2)
                y_offset = i * np.sqrt(3)/2
                
                # Scale to fit in [0,1] x [0,1]
                x = x_offset / (cols - 1) if cols > 1 else 0.5
                y = y_offset / (rows * np.sqrt(3)/2 - np.sqrt(3)/2) if rows > 1 else 0.5
                
                # Adjust to ensure points stay in [0,1] range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                points.append([x, y])
        
        return np.array(points[:16])
    
    def initialize_spherical_code_points():
        """Initialize points inspired by spherical codes for even distribution."""
        points = []
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.15, 0.85, 16)  # Centered distribution with better coverage
        
        for i in range(16):
            angle = angles[i]
            radius = radii[i]  # Vary radius to get better coverage
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points.append([max(0.05, min(0.95, x)), max(0.05, min(0.95, y))])
        
        return np.array(points)
    
    def simulated_annealing_optimization(initial_points, max_iter=10000):
        """Optimize using simulated annealing for better exploration."""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Simulated Annealing parameters - tuned for better performance
        initial_temp = 1.0
        final_temp = 0.0001
        alpha = 0.997
        
        temp = initial_temp
        
        for iteration in range(max_iter):
            # Create a neighbor solution by perturbing one point
            neighbor_points = points.copy()
            
            # Choose a random point to perturb
            idx = np.random.randint(0, len(points))
            
            # Perturb the point slightly with adaptive step size
            step_size = 0.02 / (1.0 + iteration / max_iter * 10)
            neighbor_points[idx] += np.random.normal(0, step_size, 2)
            
            # Keep points within unit square [0,1] x [0,1]
            neighbor_points[:, 0] = np.clip(neighbor_points[:, 0], 0, 1)
            neighbor_points[:, 1] = np.clip(neighbor_points[:, 1], 0, 1)
            
            # Compute ratio for neighbor
            neighbor_ratio = compute_min_max_ratio(neighbor_points)
            
            # Accept or reject the neighbor
            if neighbor_ratio > best_ratio or np.random.rand() < np.exp((neighbor_ratio - current_ratio) / temp):
                points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_ratio = neighbor_ratio
                    best_points = points.copy()
            
            # Cool down temperature
            temp = max(final_temp, temp * alpha)
            
        return best_points, best_ratio
    
    # Multi-start approach with aggressive optimization strategies
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Fibonacci spiral + Differential Evolution (Most effective approach)
    try:
        fib_points = initialize_fibonacci_points()
        x0 = fib_points.flatten()
        
        # Use differential evolution for global optimization with more iterations
        result = differential_evolution(
            objective_function,
            bounds=[(0, 1) for _ in range(32)],
            seed=42,
            maxiter=150,   # Increased iterations for better convergence
            popsize=25,    # Larger population for better exploration
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
    
    # Strategy 2: Fibonacci spiral + L-BFGS-B local optimization (fast refinement)
    try:
        fib_points = initialize_fibonacci_points()
        x0 = fib_points.flatten()
        
        # Local optimization with more iterations and tighter tolerances
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(32)],
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
    except Exception:
        pass
    
    # Strategy 3: Hexagonal grid + L-BFGS-B local optimization
    try:
        hex_points = initialize_hexagonal_points()
        x0 = hex_points.flatten()
        
        # Local optimization with tighter tolerances
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(32)],
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
    except Exception:
        pass
    
    # Strategy 4: Spherical code + L-BFGS-B local optimization
    try:
        sph_points = initialize_spherical_code_points()
        x0 = sph_points.flatten()
        
        # Local optimization with tight tolerances
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(32)],
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
    except Exception:
        pass
    
    # Strategy 5: Simulated Annealing from Fibonacci start (escape local optima)
    try:
        fib_points = initialize_fibonacci_points()
        sa_points, sa_ratio = simulated_annealing_optimization(fib_points, max_iter=15000)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            best_points = sa_points.copy()
    except Exception:
        pass
    
    # Strategy 6: Final refinement with second L-BFGS-B run from best solution
    if best_points is not None:
        try:
            x0 = best_points.flatten()
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass
    
    # If no good solution found, return Fibonacci points (known to perform well)
    if best_points is None:
        best_points = initialize_fibonacci_points()
    
    return best_points


# EVOLVE-BLOCK-END
