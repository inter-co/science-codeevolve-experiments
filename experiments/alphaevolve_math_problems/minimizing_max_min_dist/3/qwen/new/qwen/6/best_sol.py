# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, energy minimization, 
    and multiple optimization strategies to find high-quality solutions.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Strategy 1: Initialize with a hybrid geometric approach
    # Combines icosahedral structure with spectral clustering-inspired distribution
    points = _initialize_hybrid_geometric(n)
    
    # Strategy 2: Energy-based optimization to improve point distribution
    points = _energy_minimize_points(points, max_iter=500)
    
    # Strategy 3: Multi-strategy restart optimization with diversity
    best_points = points.copy()
    best_ratio = _compute_min_max_ratio(points)
    
    # Try multiple restart strategies with varying intensities
    restart_configs = [
        # Original configuration
        (lambda p: p.copy(), 0.0),
        # Small perturbations for fine-tuning
        (lambda p: p + np.random.normal(0, 0.01, (n, 3)), 0.0),
        (lambda p: p + np.random.normal(0, 0.02, (n, 3)), 0.0),
        # Medium perturbations
        (lambda p: p + np.random.normal(0, 0.05, (n, 3)), 0.0),
        (lambda p: p + np.random.normal(0, 0.08, (n, 3)), 0.0),
        # Larger perturbations for global exploration
        (lambda p: p + np.random.normal(0, 0.12, (n, 3)), 0.0),
        (lambda p: p + np.random.normal(0, 0.15, (n, 3)), 0.0),
        # Very large perturbations
        (lambda p: p + np.random.normal(0, 0.20, (n, 3)), 0.0),
        # Random uniform initialization for maximum diversity
        (lambda p: np.random.uniform(-1, 1, (n, 3)), 0.0),
    ]
    
    # Add some strategies that use different optimization methods
    restart_configs.append((lambda p: _hybrid_optimization(p), 0.0))
    
    # Try multiple restarts with different strategies
    for restart_idx, (strategy_func, _) in enumerate(restart_configs[:12]):  # Limit to 12 for time
        try:
            # Apply strategy to get perturbed points
            perturbed_points = strategy_func(best_points)
            # Project back to unit sphere
            perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
            
            # Optimize from this perturbed starting point
            optimized_points = _constrained_optimization_single(perturbed_points)
            current_ratio = _compute_min_max_ratio(optimized_points)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # Strategy 4: Final aggressive optimization with multiple methods
    try:
        # Try different optimization methods for final refinement
        final_points = _try_multiple_optimizers(best_points)
        final_ratio = _compute_min_max_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points
    except Exception:
        pass
    
    return best_points


def _initialize_hybrid_geometric(n):
    """Initialize points using a hybrid approach combining geometric constructions and spectral insights"""
    # Start with icosahedral vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    vertices = np.array(vertices)
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # For 14 points, add 2 more points strategically
    # Use a method inspired by spectral clustering to place points optimally
    # Place them along axes for better spread
    extra_points = np.array([
        [0, 0, 0.99],  # nearly north pole
        [0, 0, -0.99]  # nearly south pole
    ])
    
    points = np.vstack([vertices, extra_points])
    
    # Add small random perturbations to avoid degenerate cases
    points += np.random.normal(0, 0.01, points.shape)
    
    # Normalize again
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


def _energy_minimize_points(initial_points, max_iter=500):
    """Minimize potential energy between points on sphere - this naturally maximizes minimum distances"""
    
    n = initial_points.shape[0]
    
    # Energy function: sum of inverse squared distances (repulsive force)
    def energy_function(x):
        points = x.reshape((n, 3))
        # Compute pairwise distances
        distances = pdist(points)
        # Avoid division by zero and very small distances
        distances = np.maximum(distances, 1e-10)
        # Energy is sum of inverse squared distances (potential energy model)
        energy = np.sum(1.0 / (distances ** 2))
        return energy
    
    # Constraint function: points must remain on unit sphere
    def constraint_sphere(x):
        points = x.reshape((n, 3))
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Initial guess
    x0 = initial_points.flatten()
    
    # Use SLSQP optimization with tighter tolerances for better convergence
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
            # Normalize to ensure unit sphere constraint is maintained
            final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
            return final_points
    except:
        pass
    
    # If optimization fails, return original points
    return initial_points


def _constrained_optimization_single(initial_points, maxiter=2000, ftol=1e-12, gtol=1e-12):
    """Single constrained optimization run to refine solution"""
    n = len(initial_points)
    
    # Objective function: minimize negative of min/max ratio
    def objective(x):
        points = x.reshape((n, 3))
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0 or min_dist == 0:
            return -np.inf
            
        # Return negative ratio (since we want to maximize ratio, minimize negative)
        return -min_dist / max_dist
    
    # Constraint: keep points on unit sphere
    def constraint_sphere(x):
        points = x.reshape((n, 3))
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should equal zero for unit sphere
    
    # Run optimization with robust settings
    try:
        # Try multiple solvers for better chance of success
        solvers_to_try = ['SLSQP', 'L-BFGS-B']
        for method in solvers_to_try:
            try:
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method=method,
                    constraints={'type': 'eq', 'fun': constraint_sphere},
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
                )
                if result.success:
                    final_points = result.x.reshape((n, 3))
                    # Ensure points are on unit sphere
                    final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
                    return final_points
            except:
                continue
                
        # If all solvers fail, return initial points
        return initial_points.copy()
            
    except Exception:
        return initial_points.copy()


def _hybrid_optimization(initial_points):
    """Use a hybrid approach combining energy minimization and constrained optimization"""
    # First do energy minimization
    energy_points = _energy_minimize_points(initial_points, max_iter=300)
    
    # Then do constrained optimization
    optimized_points = _constrained_optimization_single(energy_points)
    
    return optimized_points


def _try_multiple_optimizers(initial_points):
    """Try multiple optimization approaches and return the best result"""
    best_points = initial_points.copy()
    best_ratio = _compute_min_max_ratio(initial_points)
    
    # Try different optimization parameters and methods
    configs = [
        # Standard approach
        {'method': 'SLSQP', 'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12},
        # L-BFGS-B variant
        {'method': 'L-BFGS-B', 'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12},
        # More aggressive settings
        {'method': 'SLSQP', 'maxiter': 2500, 'ftol': 1e-14, 'gtol': 1e-14},
    ]
    
    for config in configs:
        try:
            # Create fresh copy for each attempt
            test_points = initial_points.copy()
            # Run optimization
            optimized_points = _constrained_optimization_single(
                test_points, 
                maxiter=config['maxiter'], 
                ftol=config['ftol'], 
                gtol=config['gtol']
            )
            current_ratio = _compute_min_max_ratio(optimized_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = optimized_points.copy()
        except Exception:
            continue
    
    return best_points


def _compute_min_max_ratio(points):
    """Helper function to compute the min/max distance ratio"""
    if len(points) < 2:
        return 0.0
        
    # Compute pairwise distances
    distances = pdist(points)
    
    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max == 0:
        return 0.0
        
    return d_min / d_max


# EVOLVE-BLOCK-END
