# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and global optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    d = 3
    
    # Generate initial points using a more sophisticated spherical code approach
    # Using vertices of icosahedron plus carefully positioned additional points
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Create icosahedral vertices (12 vertices) - corrected version
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # For 14 points, use the 12 icosahedral vertices plus 2 more points
    points = vertices.copy()
    
    # Add 2 more points using a more refined distribution
    # Place them at specific latitudes for better spreading
    for i in range(2):
        idx = 12 + i
        # Distribute more strategically - place points near equator and poles
        if i == 0:
            # Near equator (y ~ 0)
            y = 0.0
        else:
            # Near pole (y ~ ±1) 
            y = (-1)**i * 0.7
            
        radius = np.sqrt(1 - y * y)
        
        # Use golden angle for even distribution
        golden_angle = 2 * np.pi * (3 - np.sqrt(5))
        angle = idx * golden_angle
        
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        points = np.vstack([points, [x, y, z]])
    
    # Normalize all points to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    # Convert to a proper optimization setup
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        # Minimize negative of ratio (maximize ratio)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -np.inf
            
        ratio = min_dist / max_dist
        return -ratio  # Negative because we want to maximize
    
    # Enhanced global optimization strategy - more efficient approach
    bounds = [(-1, 1) for _ in range(42)]  # 14 points * 3 coordinates
    
    # Try multiple global optimization approaches with balanced parameters
    best_ratio = -np.inf
    best_points = None
    
    # 1. Try differential evolution with moderate parameters for speed
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=50,      # Reduced iterations to save time
            popsize=15,      # Moderate population size
            mutation=(0.7, 1),  # Balanced mutation rate
            recombination=0.8,  # Balanced crossover rate
            seed=42,
            disp=False
        )
        
        if de_result.success:
            refined_points = de_result.x.reshape(-1, 3)
            norms = np.linalg.norm(refined_points, axis=1)
            refined_points = refined_points / norms[:, np.newaxis]
            
            # Local refinement with moderate settings
            x0 = refined_points.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if result.success:
                final_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(final_points, axis=1)
                final_points = final_points / norms[:, np.newaxis]
                
                # Evaluate final solution
                distances = pdist(final_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
    except Exception:
        pass
    
    # 2. Multiple restarts with different strategies - same as inspiration programs
    methods = ['L-BFGS-B', 'SLSQP']
    
    # Try same number of restarts as inspiration programs for consistency
    for restart in range(7):  # Same as inspiration programs
        np.random.seed(restart)
        # Add moderate perturbation (same as inspiration)
        perturbed_points = points + np.random.normal(0, 0.02, points.shape)  # Smaller perturbation
        perturbed_norms = np.linalg.norm(perturbed_points, axis=1)
        perturbed_points = perturbed_points / perturbed_norms[:, np.newaxis]
        
        x0 = perturbed_points.flatten()
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}  # Same tolerances as inspiration
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    distances = pdist(final_points)
                    
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                                
            except Exception:
                continue
    
    # If no optimization worked, return the initial configuration
    if best_points is None:
        return points
    
    return best_points


# EVOLVE-BLOCK-END
