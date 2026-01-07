# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0.0
        
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
        
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    if d_max <= 0:
        return 0.0
        
    return d_min / d_max

def energy_minimization(points, max_iter=300):
    """Minimize potential energy between points on sphere to improve distribution."""
    n = points.shape[0]
    
    def energy_function(x):
        points = x.reshape((n, 3))
        distances = pdist(points)
        distances = np.maximum(distances, 1e-10)  # Avoid division by zero
        energy = np.sum(1.0 / (distances ** 2))
        return energy
    
    def constraint_sphere(x):
        points = x.reshape((n, 3))
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    x0 = points.flatten()
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    try:
        result = minimize(
            energy_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_sphere},
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
        )
        
        if result.success:
            final_points = result.x.reshape((n, 3))
            final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
            return final_points
    except:
        pass
    
    return points

def objective_function(x_flat):
    """Objective function to minimize (negative of ratio for maximization)."""
    points = x_flat.reshape(-1, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    # Handle case where norm might be zero
    norms = np.where(norms == 0, 1.0, norms)
    points = points / norms[:, np.newaxis]
    ratio = compute_min_max_ratio(points)
    return -ratio  # Return negative because we want to maximize

def initialize_points_14():
    """Initialize 14 points using a combination of icosahedral structure and symmetry."""
    # Regular icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = [
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]
    
    ico_points = np.array(vertices, dtype=float)
    norms = np.linalg.norm(ico_points, axis=1)
    ico_points = ico_points / norms[:, np.newaxis]
    
    # Add 2 more points for total 14 (at poles) - improved placement
    # Use points that are more evenly distributed
    additional_points = np.array([[0, 0, 0.99], [0, 0, -0.99]])
    
    points = np.vstack([ico_points, additional_points])
    
    # Add small random perturbations to avoid degeneracy
    points += np.random.normal(0, 0.01, points.shape)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, energy minimization, and 
    multiple optimization restarts with strategic perturbations.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    # Initialize with a good geometric configuration
    points = initialize_points_14()
    
    # Apply energy minimization to get better initial distribution
    points = energy_minimization(points, max_iter=300)
    
    # Try multiple optimization runs with different starting points
    best_ratio = 0
    best_points = points.copy()
    
    # Use a more focused restart strategy based on insights from inspirations
    # Strategy 1: Original points (no perturbation) - always try first
    # Strategy 2-3: Small perturbations for local refinement
    # Strategy 4-6: Medium perturbations for global search
    # Strategy 7-10: Large perturbations for extensive global exploration
    
    restarts_to_try = 10  # Reduced to improve efficiency while maintaining quality
    
    for restart in range(restarts_to_try):
        if restart == 0:
            # Start with current best (no perturbation)
            points = best_points.copy()
        elif restart <= 2:
            # Small perturbations for refinement
            perturbed_points = best_points + np.random.normal(0, 0.02, best_points.shape)
            # Project back to sphere
            norms = np.linalg.norm(perturbed_points, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            points = perturbed_points / norms[:, np.newaxis]
        elif restart <= 5:
            # Medium perturbations for global search
            perturbed_points = best_points + np.random.normal(0, 0.05, best_points.shape)
            # Project back to sphere
            norms = np.linalg.norm(perturbed_points, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            points = perturbed_points / norms[:, np.newaxis]
        else:
            # Large perturbations for extensive global exploration
            perturbed_points = best_points + np.random.normal(0, 0.1, best_points.shape)
            # Project back to sphere
            norms = np.linalg.norm(perturbed_points, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            points = perturbed_points / norms[:, np.newaxis]
        
        # Flatten for optimization
        x0 = points.flatten()
        
        # Define bounds for optimization
        bounds = [(-1.0, 1.0) for _ in range(len(x0))]
        
        # Use SLSQP method which handles constraints better
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
            )
            
            if result.success:
                # Extract optimized points and normalize
                optimized_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(optimized_points, axis=1)
                norms = np.where(norms == 0, 1.0, norms)
                optimized_points = optimized_points / norms[:, np.newaxis]
                
                # Calculate ratio
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            continue
    
    # Final normalization to ensure points are on unit sphere
    norms = np.linalg.norm(best_points, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points


# EVOLVE-BLOCK-END
