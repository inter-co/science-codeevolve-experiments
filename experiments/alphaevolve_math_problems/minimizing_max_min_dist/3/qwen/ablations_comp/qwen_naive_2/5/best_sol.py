# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric insights with advanced optimization techniques.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    d = 3
    
    # Strategy 1: Generate a better initial configuration based on geometric principles
    # Start with a configuration inspired by the icosahedral symmetry but adapted for 14 points
    
    # Create a base configuration using icosahedral symmetry with modifications
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Base icosahedron vertices (normalized to unit sphere)
    base_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    base_vertices = base_vertices / np.linalg.norm(base_vertices, axis=1, keepdims=True)
    
    # Create a more refined initial configuration
    # Start with 12 vertices of icosahedron and add 2 more strategically
    points = base_vertices.copy()
    
    # Add two more points that help distribute the configuration better
    # Place one point near north pole and one near south pole
    points = np.vstack([points, [0, 0, 1], [0, 0, -1]])
    
    # Apply a more sophisticated distribution using Fibonacci-like approach
    # But with a twist to avoid clustering
    theta = np.linspace(0, 2*np.pi, 14, endpoint=False)
    phi_vals = np.arccos(np.linspace(-1, 1, 14))
    
    # Generate points on sphere using spherical coordinates
    points_sphere = np.zeros((14, 3))
    for i in range(14):
        points_sphere[i] = [
            np.sin(phi_vals[i]) * np.cos(theta[i]),
            np.sin(phi_vals[i]) * np.sin(theta[i]),
            np.cos(phi_vals[i])
        ]
    
    # Combine the approaches: start with the icosahedral configuration but adjust
    # Use a weighted combination to balance both structures
    points = points_sphere * 0.7 + points * 0.3
    
    # Normalize again to ensure they're on unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Scale to [0,1]^3
    points = (points + 1) / 2
    
    # Add some structured noise to escape local minima
    noise_magnitude = 0.02
    points += np.random.normal(0, noise_magnitude, points.shape)
    points = np.clip(points, 0, 1)
    
    # Improved objective function with better handling of edge cases
    def objective_function(points_flat):
        points = points_flat.reshape(-1, 3)
        
        # Compute all pairwise distances efficiently
        distances = pdist(points)
        
        if len(distances) == 0:
            return float('inf')
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist < 1e-12:
            return float('inf')
            
        # Calculate the ratio we want to maximize
        ratio = min_dist / max_dist
        
        # Use a softer penalty system to encourage exploration
        # Instead of heavy quadratic penalties, use logarithmic penalties
        if ratio < 0.1:
            # Very small ratios get strong penalty
            penalty = 100000 * (0.1 - ratio)**2
        elif ratio < 0.2:
            # Moderate ratios get moderate penalty
            penalty = 1000 * (0.2 - ratio)**2
        else:
            penalty = 0
            
        # Return negative ratio (since we minimize) plus penalty
        return -ratio + penalty
    
    # Multi-stage optimization approach
    bounds = [(0, 1)] * (3 * n)
    
    # Stage 1: Global optimization with adaptive parameters
    try:
        de_result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-8,
            rtol=1e-8
        )
        
        if de_result.success:
            points = de_result.x.reshape(-1, 3)
            points = np.clip(points, 0, 1)
    except Exception as e:
        pass
    
    # Stage 2: Local refinement with multiple methods
    try:
        # Try multiple local optimizers to avoid getting stuck
        methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
        best_points = points.copy()
        best_ratio = -float('inf')
        
        for method in methods_to_try:
            try:
                result = minimize(
                    objective_function,
                    points.flatten(),
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    candidate_points = result.x.reshape(-1, 3)
                    candidate_points = np.clip(candidate_points, 0, 1)
                    
                    # Evaluate the candidate
                    distances = pdist(candidate_points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 1e-12:
                            candidate_ratio = min_dist / max_dist
                            if candidate_ratio > best_ratio:
                                best_ratio = candidate_ratio
                                best_points = candidate_points.copy()
                                
            except:
                continue
                
        points = best_points
        
    except Exception as e:
        pass
    
    # Stage 3: Final refinement with enhanced optimization
    try:
        # Use a more aggressive optimization with better tolerances
        result = minimize(
            objective_function,
            points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            points = result.x.reshape(-1, 3)
            points = np.clip(points, 0, 1)
    except:
        pass
    
    # Ensure final result is valid
    points = np.clip(points, 0, 1)
    return points


# EVOLVE-BLOCK-END
