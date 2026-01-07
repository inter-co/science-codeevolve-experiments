# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, energy minimization, and multiple optimization restarts.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Strategy: Multi-stage approach for better results
    # Stage 1: Initialize with a known good configuration
    points = _initialize_points_better(n)
    
    # Stage 2: Energy minimization to improve distribution (like INSPIRATION 1)
    points = _energy_minimize_points(points)
    
    # Stage 3: Multiple optimization restarts with refined approach (like INSPIRATION 2)
    points = _optimized_restart_refinement(points)
    
    # Final normalization to unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


def _initialize_points_better(n):
    """Initialize points using better geometric construction"""
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
    # Using two antipodal points for better symmetry
    extra_points = np.array([[0, 0, 1], [0, 0, -1]])
    
    # Combine all points
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
    
    # Use SLSQP optimization with tighter tolerances
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    try:
        result = minimize(
            energy_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_sphere},
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
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


def _optimized_restart_refinement(points):
    """Further refine using direct optimization of min/max distance ratio with multiple restarts"""
    
    n = points.shape[0]
    
    # Objective: maximize min/max distance ratio
    def objective(x):
        points = x.reshape((n, 3))
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist > 0 and min_dist > 0:
            # Return negative ratio (we minimize to maximize ratio)
            return -min_dist / max_dist
        else:
            # Return a large penalty if invalid
            return -1.0
    
    # Constraint: keep points on unit sphere
    def constraint_sphere(x):
        points = x.reshape((n, 3))
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Bounds
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    # Multiple restarts with different strategies - combining best of both inspirations
    best_ratio = -np.inf
    best_points = points.copy()
    
    # Use a more effective restart strategy based on INSPIRATION PROGRAM 1 and 2
    # Strategy variations:
    # 0: No perturbation (original points) 
    # 1-3: Small perturbations (0.03, 0.04, 0.05)
    # 4-6: Medium perturbations (0.08, 0.10, 0.12)
    # 7-9: Large perturbations (0.15, 0.18, 0.20)
    
    restarts_to_try = 10  # Balanced approach: enough restarts but not too many
    
    for restart in range(restarts_to_try):
        # Start with current points or slightly perturbed version
        if restart == 0:
            x0 = points.flatten()
        elif restart <= 3:
            # Small random perturbations
            scale = 0.03 + (restart - 1) * 0.01
            perturbed = points + np.random.normal(0, scale, points.shape)
            # Project back to sphere
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            x0 = perturbed.flatten()
        elif restart <= 6:
            # Medium random perturbations
            scale = 0.08 + (restart - 4) * 0.02
            perturbed = points + np.random.normal(0, scale, points.shape)
            # Project back to sphere
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            x0 = perturbed.flatten()
        else:
            # Large random perturbations for global exploration
            scale = 0.15 + (restart - 7) * 0.03
            perturbed = points + np.random.normal(0, scale, points.shape)
            # Project back to sphere
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            x0 = perturbed.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': constraint_sphere},
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
            )
            
            if result.success:
                final_points = result.x.reshape((n, 3))
                final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
                distances = pdist(final_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0 and min_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
        except Exception:
            continue
    
    return best_points


# EVOLVE-BLOCK-END
