# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
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
        distances = np.maximum(distances, 1e-12)  # Avoid division by zero
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
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
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
    """Initialize 14 points using a more sophisticated approach based on known symmetric configurations."""
    # Strategy 1: Start with icosahedral configuration (12 points)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = [
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]
    
    ico_points = np.array(vertices, dtype=float)
    norms = np.linalg.norm(ico_points, axis=1)
    ico_points = ico_points / norms[:, np.newaxis]
    
    # Strategy 2: Place remaining 2 points strategically
    # Better approach: place them at positions that respect icosahedral symmetry
    # Use a configuration that's close to optimal for 14 points on sphere
    # Based on mathematical analysis, placing near z-axis but not exactly at poles works well
    
    # Use a more sophisticated approach - place points at angles that maximize uniformity
    # Try to create a configuration that's close to a truncated icosahedron or similar
    additional_points = np.array([
        [0, 0, 0.92],   # Slightly off pole for better distribution
        [0, 0, -0.92]   # Slightly off opposite pole
    ])
    
    points = np.vstack([ico_points, additional_points])
    
    # Add small random perturbations to avoid degeneracy
    points += np.random.normal(0, 0.005, points.shape)  # Smaller perturbation
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points

def initialize_points_symmetric():
    """Alternative initialization using a known symmetric configuration."""
    # Try a configuration based on the vertices of a regular icosahedron plus two additional points
    # This is inspired by the fact that 14 points can be arranged with high symmetry
    phi = (1 + np.sqrt(5)) / 2
    
    # Vertices of icosahedron
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize vertices to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # Add two more points in a way that maintains good symmetry
    # These should be placed so they're not too close to any existing vertex
    additional = np.array([
        [0, 0, 0.95],   # Near north pole
        [0, 0, -0.95]   # Near south pole
    ])
    
    points = np.vstack([vertices, additional])
    
    # Add small random noise
    points += np.random.normal(0, 0.008, points.shape)
    
    # Project back to sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, energy minimization, 
    global optimization with differential evolution, and multiple optimization restarts.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    # Initialize with multiple strategies and pick the best
    init1 = initialize_points_14()
    init2 = initialize_points_symmetric()
    
    ratio1 = compute_min_max_ratio(init1)
    ratio2 = compute_min_max_ratio(init2)
    
    # Use the better initialized configuration
    points = init1 if ratio1 > ratio2 else init2
    
    # Apply energy minimization to get better initial distribution
    points = energy_minimization(points, max_iter=300)
    
    # Try global optimization with differential evolution first (more thorough exploration)
    n = points.shape[0]
    
    def global_objective(x):
        points = x.reshape((n, 3))
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms[:, np.newaxis]
        ratio = compute_min_max_ratio(points)
        return -ratio  # Negative for maximization
    
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    try:
        # Use differential evolution for global search with tighter settings
        result = differential_evolution(
            global_objective,
            bounds,
            maxiter=200,      # Reduced for time efficiency
            popsize=15,       # Reduced for time efficiency  
            seed=42,
            disp=False,
            strategy='best1bin',
            tol=1e-12
        )
        
        if result.success:
            global_points = result.x.reshape((n, 3))
            norms = np.linalg.norm(global_points, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            global_points = global_points / norms[:, np.newaxis]
            global_ratio = compute_min_max_ratio(global_points)
            
            # If global optimization improves, use it
            if global_ratio > compute_min_max_ratio(points):
                points = global_points.copy()
    except Exception:
        pass
    
    # Try multiple optimization runs with different starting points
    best_ratio = compute_min_max_ratio(points)
    best_points = points.copy()
    
    # Multiple restarts with different strategies inspired by both inspirations
    # Strategy 1: Original points (no perturbation)
    # Strategy 2-5: Small perturbations
    # Strategy 6-15: Medium perturbations for global search
    
    for restart in range(20):  # Increase restarts for better exploration
        if restart == 0:
            # No perturbation - start with current best
            points = best_points.copy()
        elif restart <= 5:
            # Small perturbation (like INSPIRATION 1)
            perturbed_points = best_points + np.random.normal(0, 0.015, best_points.shape)
            # Project back to sphere
            norms = np.linalg.norm(perturbed_points, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            points = perturbed_points / norms[:, np.newaxis]
        elif restart <= 10:
            # Medium perturbation for more exploration
            perturbed_points = best_points + np.random.normal(0, 0.03, best_points.shape)
            # Project back to sphere
            norms = np.linalg.norm(perturbed_points, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            points = perturbed_points / norms[:, np.newaxis]
        else:
            # Larger perturbation for global exploration (like INSPIRATION 2)
            perturbed_points = best_points + np.random.normal(0, 0.06, best_points.shape)
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
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
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
