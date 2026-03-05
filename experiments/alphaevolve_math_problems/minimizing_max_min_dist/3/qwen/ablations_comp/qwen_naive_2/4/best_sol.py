# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
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
    
    # Strategy 1: Generate a much better initial configuration
    # Based on known good configurations for 14 points in 3D
    # Using a combination of icosahedral structure with strategic additions
    
    # Start with a regular icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Base icosahedron vertices (normalized to unit sphere)
    base_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    base_vertices = base_vertices / np.linalg.norm(base_vertices, axis=1, keepdims=True)
    
    # Create a better initial configuration
    # Start with icosahedron and add 2 points along z-axis
    points = base_vertices.copy()
    
    # Add two additional points to create a more balanced distribution
    # These should be positioned to help spread out the points
    points = np.vstack([points, [0, 0, 0.95], [0, 0, -0.95]])
    
    # Generate points using a better spherical distribution
    # Use Fibonacci spiral on sphere for better distribution
    points_sphere = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    for i in range(14):
        y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = golden_angle * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points_sphere.append([x, y, z])
    
    points_sphere = np.array(points_sphere)
    
    # Combine both approaches: use the icosahedral structure as primary but 
    # adapt it with the Fibonacci-based distribution
    points = points_sphere * 0.7 + points * 0.3
    
    # Normalize to ensure points are on unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Scale to [0,1]^3 (centered around origin then mapped to [0,1])
    points = (points + 1) / 2
    
    # Add slight randomization to escape local minima
    noise_magnitude = 0.01
    points += np.random.normal(0, noise_magnitude, points.shape) * 0.5
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
        
        # Much simpler penalty system - just heavily penalize very poor solutions
        if ratio < 0.1:
            # Very poor ratios get severe penalty
            penalty = 1000000 * (0.1 - ratio)**2
        elif ratio < 0.2:
            # Poor ratios get moderate penalty  
            penalty = 10000 * (0.2 - ratio)**2
        else:
            penalty = 0
            
        # Return negative ratio (since we minimize) plus penalty
        return -ratio + penalty
    
    # Multi-stage optimization approach with improved strategy
    bounds = [(0, 1)] * (3 * n)
    
    # Stage 1: Global optimization with better parameters
    try:
        de_result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-10,
            rtol=1e-10
        )
        
        if de_result.success:
            points = de_result.x.reshape(-1, 3)
            points = np.clip(points, 0, 1)
    except Exception as e:
        pass
    
    # Stage 2: Local refinement with more aggressive optimization
    try:
        # Try multiple local optimizers with better settings
        methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
        best_points = points.copy()
        best_ratio = -float('inf')
        
        # Run optimization from the DE result
        for method in methods_to_try:
            try:
                result = minimize(
                    objective_function,
                    points.flatten(),
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
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
    
    # Stage 3: Even more aggressive refinement with multiple restarts
    try:
        # Create multiple random restarts to improve chances of finding better solution
        best_final_points = points.copy()
        best_final_ratio = -float('inf')
        
        # Try several different restart strategies
        for restart in range(5):
            # Different perturbation strategies
            if restart == 0:
                # Small random perturbation
                restart_points = points + np.random.normal(0, 0.005, points.shape)
            elif restart == 1:
                # Larger perturbation
                restart_points = points + np.random.normal(0, 0.02, points.shape)
            else:
                # Random initialization
                restart_points = np.random.rand(*points.shape)
            
            restart_points = np.clip(restart_points, 0, 1)
            
            # Local optimization from this restart
            result = minimize(
                objective_function,
                restart_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
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
                        if candidate_ratio > best_final_ratio:
                            best_final_ratio = candidate_ratio
                            best_final_points = candidate_points.copy()
        
        points = best_final_points
        
    except Exception as e:
        pass
    
    # Final check and refinement
    try:
        # Try one final optimization with very tight tolerances
        final_result = minimize(
            objective_function,
            points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
        )
        
        if final_result.success:
            final_points = final_result.x.reshape(-1, 3)
            final_points = np.clip(final_points, 0, 1)
            
            # Check if this improves our solution
            distances = pdist(final_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 1e-12:
                    final_ratio = min_dist / max_dist
                    if final_ratio > best_final_ratio:
                        points = final_points
                        
    except Exception as e:
        pass
    
    # Ensure final result is valid
    points = np.clip(points, 0, 1)
    return points


# EVOLVE-BLOCK-END
