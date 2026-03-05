# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses advanced mathematical frameworks including icosahedral symmetry, spherical codes,
    and multi-start optimization strategies.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        """Objective function: minimize negative of min/max ratio"""
        # Reshape flat array back to 3D points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio (since we want to maximize ratio)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def constraint_func(x):
        """Constraint function to keep points within unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Points should be within unit sphere (norm <= 1)
        return 1.0 - norms
    
    # Advanced initialization using icosahedral symmetry and spherical codes
    np.random.seed(42)
    
    # Generate points based on icosahedral symmetry (highly symmetric structure)
    # This provides a mathematically elegant starting configuration
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = []
    
    # Create vertices of a regular icosahedron scaled appropriately
    # These are 12 vertices of an icosahedron
    coords = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    
    # Normalize to unit sphere and create initial configuration
    for coord in coords:
        norm = np.sqrt(sum(c**2 for c in coord))
        vertices.append([c/norm for c in coord])
    
    # Start with 12 icosahedral vertices
    points = np.array(vertices)
    
    # Add 2 more points strategically
    # Add one at origin and one at a well-distributed location
    points = np.vstack([points, [[0, 0, 0], [0.5, 0.5, 0.5]]])
    
    # Perturb slightly to break perfect symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        points = points / max_norm * 0.9
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define bounds (points in unit sphere)
    bounds = [(-1, 1) for _ in range(42)]
    
    # Constraints for unit sphere
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Enhanced optimization using multiple strategies
    best_ratio = -np.inf
    best_points = points.copy()
    
    # Strategy 1: SLSQP with constraints (primary approach)
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            eval_ratio = -objective(result.x)
            if eval_ratio > best_ratio:
                best_ratio = eval_ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Differential Evolution for global search (inspired by evolutionary approaches)
    try:
        result = minimize(
            objective,
            x0,
            method='differential_evolution',
            options={'maxiter': 1000, 'popsize': 15, 'tol': 1e-12},
            bounds=[(-1, 1) for _ in range(42)]
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            eval_ratio = -objective(result.x)
            if eval_ratio > best_ratio:
                best_ratio = eval_ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Strategy 3: Local refinement with improved convergence criteria
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15},
            tol=1e-15
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_ratio = -objective(refined_points.flatten())
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()
    except Exception:
        pass
    
    # Final validation and cleanup
    # Ensure final points are within unit sphere
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask][:, np.newaxis]
    
    # Apply final refinement with SLSQP if needed
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16},
            tol=1e-16
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_ratio = -objective(final_points.flatten())
            if final_ratio > best_ratio:
                best_points = final_points.copy()
    except Exception:
        pass
    
    # Convert back to [0,1]^3 coordinate system
    # This maps from [-1,1]^3 to [0,1]^3
    best_points = (best_points + 1) / 2
    
    return best_points


# EVOLVE-BLOCK-END
