# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import pdist
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses hybrid approach combining mathematical constructions with advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_ratio(points):
        """Compute the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0.0
    
    # Create high-quality initial configurations inspired by mathematical constructions
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Icosahedral configuration (strong mathematical foundation)
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    # Normalize vertices to unit sphere and convert to [0,1]^3
    icosahedral_points = []
    for v in vertices:
        norm = np.linalg.norm(v)
        icosahedral_points.append([(v[0]/norm + 1)/2, (v[1]/norm + 1)/2, (v[2]/norm + 1)/2])
    
    # Add 2 strategic points - use more carefully chosen positions for better distribution
    icosahedral_points.extend([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    icosahedral_points = np.array(icosahedral_points)
    
    # Fibonacci spiral configuration (good distribution)
    fib_points = []
    for i in range(14):
        phi_angle = np.arccos(-1 + 2*i/13)
        theta = np.sqrt(14 * np.pi) * i
        x = np.sin(phi_angle) * np.cos(theta)
        y = np.sin(phi_angle) * np.sin(theta)
        z = np.cos(phi_angle)
        fib_points.append([(x + 1)/2, (y + 1)/2, (z + 1)/2])
    fib_points = np.array(fib_points)
    
    # Random configuration (exploration)
    random_points = np.random.rand(14, 3)
    
    # Initialize with the best of these configurations
    configs = [
        ("icosahedral", icosahedral_points),
        ("fibonacci", fib_points),
        ("random", random_points)
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    for name, config in configs:
        ratio = compute_ratio(config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = config.copy()
    
    # Objective function for optimization
    def objective(x):
        points_test = x.reshape(-1, 3)
        distances = pdist(points_test)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        # Return negative ratio to maximize ratio
        return -min_dist / max_dist if max_dist > 0 else -1.0
    
    # Multi-stage optimization approach with enhanced parameters from INSPIRATION 2
    current_points = best_points.copy()
    current_ratio = compute_ratio(current_points)
    
    # Stage 1: Differential Evolution for global optimization with aggressive parameters
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,      # More iterations for better global search (from INSPIRATION 2)
            popsize=30,       # Larger population for better diversity (from INSPIRATION 2)
            mutation=(0.9, 1), # Higher mutation for more exploration (from INSPIRATION 2)
            recombination=0.95, # Better mixing (from INSPIRATION 2)
            seed=42,
            disp=False,
            maxfun=750
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_ratio(refined_points)
            if ratio > current_ratio:
                current_ratio = ratio
                current_points = refined_points.copy()
    except Exception:
        pass
    
    # Stage 2: Basin Hopping with better parameters for further exploration
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        bh_result = basinhopping(
            objective, 
            current_points.flatten(), 
            niter=50,        # More iterations for better exploration (from INSPIRATION 2)
            T=0.4,           # Higher temperature for better escape from local minima (from INSPIRATION 2)
            stepsize=0.05,   # Larger step size for broader search (from INSPIRATION 2)
            minimizer_kwargs={'bounds': bounds, 'method': 'L-BFGS-B'}
        )
        
        candidate_points = bh_result.x.reshape(14, 3)
        candidate_points = np.clip(candidate_points, 0, 1)
        ratio = compute_ratio(candidate_points)
        if ratio > current_ratio:
            current_ratio = ratio
            current_points = candidate_points.copy()
    except Exception:
        pass
    
    # Stage 3: Multiple restarts with diverse strategies and tighter tolerances
    # Use fewer restarts but more sophisticated diversification (from INSPIRATION 1)
    for restart in range(10):  # Reduce restarts for efficiency (from INSPIRATION 1)
        try:
            np.random.seed(restart * 1000 + 42)
            # Diverse perturbation strategies (from INSPIRATION 1)
            if restart < 5:
                # Small Gaussian perturbation
                perturbed = current_points + np.random.normal(0, 0.015, current_points.shape)
            else:
                # Medium Gaussian perturbation  
                perturbed = current_points + np.random.normal(0, 0.02, current_points.shape)
            
            perturbed = np.clip(perturbed, 0, 1)
            
            # Different optimization methods for diversity (from INSPIRATION 1)
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            method = methods[restart % 3]
            
            result = minimize(
                objective,
                perturbed.flatten(),
                method=method,
                bounds=[(0, 1) for _ in range(14 * 3)],
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}  # Standard tolerances (from INSPIRATION 1)
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)
                ratio = compute_ratio(refined_points)
                if ratio > current_ratio:
                    current_ratio = ratio
                    current_points = refined_points.copy()
        except Exception:
            continue
    
    # Stage 4: Simplified coordinate-wise refinement (from INSPIRATION 2)
    try:
        # Simplified coordinate descent with fewer iterations for efficiency
        for iteration in range(5):  # Fewer iterations for efficiency (from INSPIRATION 2)
            improved = False
            for coord_idx in range(3):
                best_coord = current_points[:, coord_idx].copy()
                best_ratio_local = compute_ratio(current_points)
                
                # Try fewer step sizes for efficiency
                step_sizes = [0.01, 0.02]
                for step_size in step_sizes:
                    for perturbation in [-step_size, step_size]:
                        test_points = current_points.copy()
                        test_points[:, coord_idx] += perturbation
                        test_points[:, coord_idx] = np.clip(test_points[:, coord_idx], 0, 1)
                        
                        ratio = compute_ratio(test_points)
                        if ratio > best_ratio_local:
                            best_ratio_local = ratio
                            best_coord = test_points[:, coord_idx].copy()
                            improved = True
                
                current_points[:, coord_idx] = best_coord
            
            # Early termination if no improvement
            if not improved:
                break
                
    except Exception:
        pass
    
    # Stage 5: Final optimization with moderate parameters
    try:
        result = minimize(
            objective,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-13}  # Moderate tolerances for efficiency
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_ratio(refined_points)
            if ratio > current_ratio:
                current_ratio = ratio
                current_points = refined_points.copy()
    except Exception:
        pass
    
    return current_points


# EVOLVE-BLOCK-END
