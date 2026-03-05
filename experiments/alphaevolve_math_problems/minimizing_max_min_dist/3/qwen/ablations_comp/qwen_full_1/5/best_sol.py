# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust geometric initialization combined with efficient gradient-based optimization.
    
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
    
    def generate_initial_configuration():
        """Generate a high-quality initial configuration based on icosahedral symmetry"""
        # Start with icosahedral arrangement (12 vertices)
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add 2 more points for 14 total - use a more sophisticated placement
        # Try a configuration that spreads points more evenly
        additional_points = np.array([
            [0, 0, 0.9],   # Near north pole
            [0, 0, -0.9]   # Near south pole
        ])
        
        # Combine and add small random perturbations for diversity
        points = np.vstack([vertices, additional_points])
        np.random.seed(42)
        # Use slightly larger noise to allow better exploration
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        return points
    
    def objective_function(x):
        """Objective function to minimize (negative of ratio)"""
        points = x.reshape(-1, 3)
        d_min, d_max = compute_distances(points)
        
        # Avoid division by zero or invalid cases
        if d_max <= 1e-12:
            return 1e10  # Large penalty for invalid configurations
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
    def sphere_constraint(x):
        """Constraint function for unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should be zero for points on unit sphere
    
    # Generate initial configuration
    initial_points = generate_initial_configuration()
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Define constraints for optimization
    constraints = {'type': 'eq', 'fun': sphere_constraint}
    
    # Multiple restarts with different random seeds for better exploration
    best_ratio = -np.inf
    best_points = initial_points.copy()
    
    # Enhanced restart strategy with more aggressive optimization
    for restart in range(15):  # Increased from 10 to 15 for even better exploration
        # Set seed for reproducibility
        np.random.seed(restart * 100 + 42)
        
        # Add small random perturbation to initial points for each restart
        if restart > 0:
            # More varied perturbation scales to explore wider space
            if restart <= 5:
                perturbation_scale = 0.08
            elif restart <= 10:
                perturbation_scale = 0.04
            else:
                perturbation_scale = 0.02
                
            perturbed_x0 = x0 + np.random.normal(0, perturbation_scale, len(x0))
            # Renormalize to unit sphere
            perturbed_points = perturbed_x0.reshape(-1, 3)
            norms = np.linalg.norm(perturbed_points, axis=1)
            norms = np.maximum(norms, 1e-12)
            perturbed_points = perturbed_points / norms[:, np.newaxis]
            x0 = perturbed_points.flatten()
        
        try:
            # Use SLSQP with constraints for optimization with highest tolerance
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1200, 'ftol': 1e-14, 'gtol': 1e-14},
                tol=1e-14
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                d_min, d_max = compute_distances(final_points)
                
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
        except Exception:
            continue
    
    # Final aggressive optimization with even tighter tolerances
    try:
        final_result = minimize(
            objective_function,
            best_points.flatten(),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15},
            tol=1e-15
        )
        
        if final_result.success:
            final_points = final_result.x.reshape(-1, 3)
            d_min, d_max = compute_distances(final_points)
            if d_max > 1e-12:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_points = final_points.copy()
                    
    except Exception:
        pass
    
    # Additional fallback optimization
    try:
        # Try L-BFGS-B with even more iterations for fine-tuning
        lbfgs_result = minimize(
            objective_function,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15},
            tol=1e-15
        )
        
        if lbfgs_result.success:
            lbfgs_points = lbfgs_result.x.reshape(-1, 3)
            d_min, d_max = compute_distances(lbfgs_points)
            if d_max > 1e-12:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_points = lbfgs_points.copy()
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
