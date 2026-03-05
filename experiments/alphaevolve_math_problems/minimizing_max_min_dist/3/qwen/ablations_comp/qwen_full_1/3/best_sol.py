# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses improved initialization and robust optimization strategy with better restarts.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_distances(points):
        """Compute pairwise distances and return min/max ratio"""
        if len(points) < 2:
            return 0, 0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0, 0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min, d_max
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x.reshape(14, 3)
        d_min, d_max = compute_distances(points)
        
        # Avoid division by zero or invalid cases
        if d_max <= 1e-12:
            return 1e10  # Large penalty for invalid configurations
            
        # Return negative ratio (since we want to maximize)
        ratio = d_min / d_max
        return -ratio
    
    # Fibonacci spiral initialization (very good distribution)
    def fibonacci_spiral():
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(14):
            theta = np.arccos(1 - 2 * i / 13)  # 14 points from 0 to 13
            phi = 2 * np.pi * i / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)
    
    # Better cube-based initialization with diagonals (from inspiration programs)
    def cube_diagonal_initialization():
        # Start with vertices of a cube (8 points) 
        points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    points.append([i, j, k])
        
        # Add points along axes and diagonals for better distribution
        points.extend([
            [1.5, 0, 0], [-1.5, 0, 0],  # x-axis
            [0, 1.5, 0], [0, -1.5, 0],  # y-axis  
            [0, 0, 1.5], [0, 0, -1.5],  # z-axis
            [0.707, 0.707, 0.707], [-0.707, -0.707, -0.707],  # diagonals
            [0.707, -0.707, 0.707], [-0.707, 0.707, -0.707]   # more diagonals
        ])
        
        # Keep only first 14 points and normalize
        points = np.array(points[:14], dtype=float)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points
    
    # Icosahedral-based initialization (structured and symmetric)
    def icosahedral_initialization():
        # Correct icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add 2 more points for 14 total - place strategically along z-axis
        additional_points = np.array([
            [0, 0, 0.9],   # Near north pole
            [0, 0, -0.9]   # Near south pole
        ])
        
        points = np.vstack([vertices, additional_points])
        return points
    
    # Generate multiple initial configurations
    initial_strategies = [
        ("fibonacci", fibonacci_spiral()),
        ("cube_diagonal", cube_diagonal_initialization()),
        ("icosahedral", icosahedral_initialization()),
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Run optimization for each initialization strategy
    for strategy_name, initial_points in initial_strategies:
        # Try 3 restarts per strategy (more efficient for time budget)
        for restart in range(3):
            # Slight perturbation for restart
            if restart > 0:
                np.random.seed(42 + restart)
                perturbed = initial_points + np.random.normal(0, 0.03, initial_points.shape)
                # Normalize again to unit sphere
                norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                perturbed = perturbed / norms
            else:
                perturbed = initial_points.copy()
            
            # Flatten for optimization
            x0 = perturbed.flatten()
            
            # Use multiple optimization methods for robustness
            methods_to_try = ['L-BFGS-B', 'SLSQP']
            
            for method in methods_to_try:
                try:
                    # Early exit if we're close to benchmark
                    if best_ratio > 0.48:
                        return best_points
                    
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12},
                        tol=1e-12
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(14, 3)
                        d_min, d_max = compute_distances(optimized_points)
                        
                        if d_max > 1e-12:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                
                except Exception:
                    continue
    
    # Final refinement with highest precision
    if best_points is not None:
        try:
            # One final high-precision optimization
            final_result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
                tol=1e-14
            )
            
            if final_result.success:
                final_points = final_result.x.reshape(14, 3)
                d_min, d_max = compute_distances(final_points)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = final_points.copy()
        except Exception:
            pass
    
    # If no optimization worked, return the best initial configuration
    if best_points is None:
        # Return fibonacci spiral as the most reliable fallback
        return fibonacci_spiral()
    
    return best_points


# EVOLVE-BLOCK-END
