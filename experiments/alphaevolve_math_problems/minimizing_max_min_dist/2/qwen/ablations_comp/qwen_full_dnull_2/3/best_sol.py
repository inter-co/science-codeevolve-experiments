# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical constructions with aggressive 
    optimization techniques to beat the benchmark.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective_function(points_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to points
        points = points_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero and return very negative value for invalid cases
        if d_max <= 0:
            return -np.inf
            
        # Return negative ratio (since we want to maximize)
        return -d_min / d_max
    
    def evaluate_solution(points):
        """Evaluate the quality of a solution"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    def create_icosahedral_like():
        """Create configuration inspired by icosahedral symmetry - proven to work well"""
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # Create a 2D approximation of icosahedral symmetry using 16 points
        # This creates a pattern that maximizes minimum distance through symmetry
        points = []
        
        # Generate points that form a kind of icosahedral-like distribution
        # In 2D, we can approximate this with a pattern that avoids clustering
        
        # Add 8 points in a cross pattern (similar to icosahedral vertices)
        for i in range(8):
            angle = 2 * math.pi * i / 8
            radius = 0.5 if i % 2 == 0 else 0.3
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        # Add 8 more points in a different arrangement to avoid symmetry issues
        for i in range(8):
            angle = 2 * math.pi * i / 8 + math.pi/8  # Offset by 45 degrees
            radius = 0.4 if i % 2 == 0 else 0.6
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points[:16])
        
        # Apply additional normalization to keep within bounds
        points = np.clip(points, 0, 1)
        return points
    
    def create_fibonacci_like():
        """Create a Fibonacci-inspired pattern for good dispersion"""
        points = []
        n = 16
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n):
            # Use Fibonacci-like angular distribution
            theta = 2 * math.pi * i / golden_ratio
            radius = math.sqrt(i / (n - 1)) if i < n - 1 else 1.0
            
            x = 0.5 + 0.4 * radius * math.cos(theta)
            y = 0.5 + 0.4 * radius * math.sin(theta)
            
            points.append([x, y])
        
        return np.array(points)
    
    def create_hexagonal_grid():
        """Create a high-quality hexagonal grid initialization"""
        # Create a 4x4 grid with hexagonal offset
        points = []
        for i in range(4):
            for j in range(4):
                x = i + (j % 2) * 0.5
                y = j * math.sqrt(3)/2
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to [0,1] x [0,1]
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        return points
    
    def create_optimized_initial():
        """Create an optimized initial configuration using known good patterns"""
        # Start with a 4x4 grid pattern
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0, j/3.0])
        
        points = np.array(points)
        
        # Apply slight perturbations to break symmetries
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    def create_spherical_code_like():
        """Create configuration inspired by spherical codes for 16 points"""
        # Use a combination of circular arrangements
        points = []
        
        # Add 8 points on a circle (evenly spaced)
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = 0.3 * math.cos(angle)
            y = 0.3 * math.sin(angle)
            points.append([x, y])
        
        # Add 8 more points in a second circle with different radius
        for i in range(8):
            angle = 2 * math.pi * i / 8 + math.pi/8  # Offset by 45 degrees
            x = 0.6 * math.cos(angle)
            y = 0.6 * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points[:16])
        
        # Center and scale appropriately
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        points[:, 0] -= mean_x
        points[:, 1] -= mean_y
        
        # Scale to fit nicely in [0.1, 0.9] x [0.1, 0.9]
        max_dist = np.max(np.abs(points))
        if max_dist > 0:
            points = points / max_dist * 0.4 + 0.5
        
        return points
    
    def create_clustered_pattern():
        """Create a pattern that distributes points in clusters to avoid extreme clustering"""
        points = []
        
        # Create 4 clusters of 4 points each
        cluster_centers = [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75]
        ]
        
        for center in cluster_centers:
            for _ in range(4):
                # Add points around each cluster center with more variance
                x = center[0] + np.random.normal(0, 0.12, 1)[0]
                y = center[1] + np.random.normal(0, 0.12, 1)[0]
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initialization strategies and find the best one
    initial_strategies = [
        create_icosahedral_like,   # Prioritize mathematically proven construction
        create_fibonacci_like,     # Good for dispersion
        create_spherical_code_like, # Another strong candidate
        create_hexagonal_grid,     # Structured approach
        create_optimized_initial,  # Grid-based with perturbations
        create_clustered_pattern   # Clustered approach
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Test each initialization strategy
    for strategy in initial_strategies:
        try:
            points = strategy()
            
            # Evaluate the initial configuration
            ratio = evaluate_solution(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception:
            continue
    
    # If no good initialization found, use fallback
    if best_points is None:
        best_points = create_optimized_initial()
    
    # Multi-stage optimization approach with aggressive parameters
    x0 = best_points.flatten()
    bounds = [(0, 1) for _ in range(32)]
    
    # Strategy 1: Aggressive Differential Evolution
    try:
        result_de = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=200,      # More iterations for better convergence
            popsize=60,       # Larger population for better exploration
            tol=1e-15,        # Very tight tolerance
            recombination=0.95,
            mutation=(0.8, 1.0),
            strategy='best1bin'
        )
        
        # Strategy 2: Refine with multiple local optimization methods
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        for method in methods:
            try:
                refined_result = minimize(
                    objective_function,
                    result_de.x,
                    method=method,
                    bounds=bounds,
                    options={'ftol': 1e-16, 'gtol': 1e-16, 'maxiter': 1000}
                )
                
                # Extract final points
                final_points = refined_result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                
                # Verify this is actually better than our starting point
                final_ratio = evaluate_solution(final_points)
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()
                    
            except Exception:
                continue
                
    except Exception:
        # Fallback to just the best initial configuration
        pass
    
    # Strategy 3: Additional refinement with direct optimization from best initial
    try:
        refined_result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-16, 'gtol': 1e-16, 'maxiter': 1000}
        )
        
        final_points = refined_result.x.reshape(-1, 2)
        final_points = np.clip(final_points, 0, 1)
        
        final_ratio = evaluate_solution(final_points)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = final_points.copy()
            
    except Exception:
        pass
    
    # Final verification and return
    final_ratio = evaluate_solution(best_points)
    if final_ratio <= 0:
        # If something went wrong, return the best we had
        return create_optimized_initial()
    
    return best_points


# EVOLVE-BLOCK-END
