# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, mathematical optimization, 
    and strategic restarts for improved convergence and benchmark beating.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Generate initial points using a sophisticated spherical code approach
    # Using vertices of icosahedron plus carefully positioned additional points
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Create icosahedral vertices (12 vertices)
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
    
    # Enhanced global optimization strategy with algorithmic diversity
    bounds = [(-1, 1) for _ in range(42)]  # 14 points * 3 coordinates
    
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Mathematical optimization (gradient-based + global search)
    # Use more aggressive differential evolution settings for better convergence
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=80,      # Increased iterations for better convergence
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
            
            # Local refinement with tighter tolerances
            x0 = refined_points.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
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
    
    # Strategy 2: Physics-inspired energy minimization approach
    # This approach treats the problem as minimizing an energy function where points repel each other
    def energy_objective(x_flat):
        points = x_flat.reshape(-1, 3)
        distances = pdist(points)
        
        if len(distances) == 0:
            return np.inf
            
        # Energy function: sum of inverse squares of distances (repulsive forces)
        # This encourages points to spread out
        energy = 0
        for dist in distances:
            if dist > 0:  # Avoid division by zero
                energy += 1.0 / (dist * dist)
        return energy
    
    # Run energy-based optimization with different parameters
    try:
        # Start with the same initial points
        x0_energy = points.flatten()
        
        # Use a different optimization method for energy minimization
        result_energy = minimize(
            energy_objective,
            x0_energy,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result_energy.success:
            final_points_energy = result_energy.x.reshape(-1, 3)
            norms_energy = np.linalg.norm(final_points_energy, axis=1)
            final_points_energy = final_points_energy / norms_energy[:, np.newaxis]
            
            # Evaluate the energy-based solution
            distances_energy = pdist(final_points_energy)
            if len(distances_energy) > 0:
                min_dist_energy = np.min(distances_energy)
                max_dist_energy = np.max(distances_energy)
                if max_dist_energy > 0:
                    ratio_energy = min_dist_energy / max_dist_energy
                    if ratio_energy > best_ratio:
                        best_ratio = ratio_energy
                        best_points = final_points_energy.copy()
    except Exception:
        pass
    
    # Strategy 3: Multiple restarts with different strategies - optimized for time efficiency
    methods = ['L-BFGS-B', 'SLSQP']
    
    # Try fewer random initializations but with better perturbations
    for restart in range(10):  # More restarts than before to increase chances
        np.random.seed(restart)
        # Add moderate perturbation
        perturbed_points = points + np.random.normal(0, 0.02, points.shape)
        perturbed_norms = np.linalg.norm(perturbed_points, axis=1)
        perturbed_points = perturbed_points / perturbed_norms[:, np.newaxis]
        
        x0 = perturbed_points.flatten()
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8}
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
