# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import pdist
from numba import jit
import warnings

@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """Fast computation of min/max distance ratio using compiled function"""
    n = points.shape[0]
    min_dist = float('inf')
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
                
    if max_dist > 0:
        return min_dist / max_dist
    else:
        return 0.0

def compute_ratio(points):
    """Compute the min/max distance ratio for given points"""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return min_dist / max_dist if max_dist > 0 else 0.0

def compute_forces_enhanced_jit(points):
    """Enhanced force computation with better repulsion model using JIT"""
    n = points.shape[0]
    forces = np.zeros_like(points)
    
    # Compute pairwise distances and forces with improved physics model
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            
            # Avoid division by zero and very small distances
            if dist_sq > 1e-12:
                # Use a more sophisticated force model with better handling
                dist = np.sqrt(dist_sq)
                # Different force behaviors for different distance ranges
                if dist < 0.05:  # Very close points - strong repulsion
                    force_magnitude = 500.0 / (dist_sq + 1e-12)
                elif dist < 0.5:  # Close points - strong repulsion
                    force_magnitude = 100.0 / (dist_sq + 1e-12)
                elif dist > 2.0:  # Very distant points - weak attraction
                    force_magnitude = 0.01 / (dist_sq + 1e-12)
                else:  # Moderate distances - normal repulsion
                    force_magnitude = 1.0 / (dist_sq + 1e-12)
                
                force_vector = force_magnitude * diff / (dist + 1e-12)
                forces[i] += force_vector
                forces[j] -= force_vector
    
    return forces

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses hybrid approach combining mathematical constructions with advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Create high-quality initial configurations inspired by INSPIRATION 1 and 3
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
    
    # Add 2 strategic points - more carefully chosen for better distribution
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
        ratio = compute_min_max_ratio_jit(config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = config.copy()
    
    # Objective function for optimization (using JIT for speed)
    def objective(x):
        points_test = x.reshape(-1, 3)
        return -compute_min_max_ratio_jit(points_test)
    
    # Physics-based refinement function
    def physics_refinement(points, max_iter=200):
        """Refine points using physics-based iterative improvement"""
        current_points = points.copy()
        best_ratio = compute_min_max_ratio_jit(current_points)
        best_points = current_points.copy()
        
        # Track improvement for early stopping
        last_improvement = 0
        patience_counter = 0
        patience_limit = 30
        
        for iteration in range(max_iter):
            forces = compute_forces_enhanced_jit(current_points)
            current_points += 0.002 * forces
            
            # Project back to unit sphere
            norms = np.linalg.norm(current_points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1, norms)
            current_points = current_points * 1.0 / norms
            
            # Occasionally check for improvement
            if iteration % 20 == 0:
                current_ratio = compute_min_max_ratio_jit(current_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
                    last_improvement = iteration
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping if no improvement for too long
                if patience_counter >= patience_limit:
                    break
        
        return best_points
    
    # Multi-stage optimization approach like INSPIRATION 1 but with JIT acceleration
    current_points = best_points.copy()
    current_ratio = compute_min_max_ratio_jit(current_points)
    
    # Stage 1: Differential Evolution for global optimization with aggressive parameters
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,      # More iterations for better global search (from INSPIRATION 1)
            popsize=30,       # Larger population for better diversity (from INSPIRATION 1)
            mutation=(0.9, 1), # Higher mutation for more exploration (from INSPIRATION 1)
            recombination=0.95, # Better mixing (from INSPIRATION 1)
            seed=42,
            disp=False,
            maxfun=750
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_min_max_ratio_jit(refined_points)
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
            niter=50,        # More iterations for better exploration (from INSPIRATION 1)
            T=0.4,           # Higher temperature for better escape from local minima (from INSPIRATION 1)
            stepsize=0.05,   # Larger step size for broader search (from INSPIRATION 1)
            minimizer_kwargs={'bounds': bounds, 'method': 'L-BFGS-B'}
        )
        
        candidate_points = bh_result.x.reshape(14, 3)
        candidate_points = np.clip(candidate_points, 0, 1)
        ratio = compute_min_max_ratio_jit(candidate_points)
        if ratio > current_ratio:
            current_ratio = ratio
            current_points = candidate_points.copy()
    except Exception:
        pass
    
    # Stage 3: Physics-based refinement for better local improvement
    try:
        physics_refined = physics_refinement(current_points, max_iter=200)
        ratio = compute_min_max_ratio_jit(physics_refined)
        if ratio > current_ratio:
            current_ratio = ratio
            current_points = physics_refined.copy()
    except Exception:
        pass
    
    # Stage 4: Multiple restarts with diverse strategies and tighter tolerances
    # Use fewer restarts but more sophisticated diversification (from INSPIRATION 2)
    for restart in range(10):  # Balanced number of restarts
        try:
            np.random.seed(restart * 1000 + 42)
            # Diverse perturbation strategies (from INSPIRATION 1)
            if restart < 5:
                # Small Gaussian perturbation
                perturbed = current_points + np.random.normal(0, 0.015, current_points.shape)
            elif restart < 8:
                # Medium Gaussian perturbation  
                perturbed = current_points + np.random.normal(0, 0.02, current_points.shape)
            else:
                # Uniform perturbation for broad search
                perturbed = current_points + np.random.uniform(-0.025, 0.025, current_points.shape)
            
            perturbed = np.clip(perturbed, 0, 1)
            
            # Different optimization methods for diversity (from INSPIRATION 1)
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            method = methods[restart % 3]
            
            result = minimize(
                objective,
                perturbed.flatten(),
                method=method,
                bounds=[(0, 1) for _ in range(14 * 3)],
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}  # Tighter tolerances (from INSPIRATION 1)
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)
                ratio = compute_min_max_ratio_jit(refined_points)
                if ratio > current_ratio:
                    current_ratio = ratio
                    current_points = refined_points.copy()
        except Exception:
            continue
    
    # Stage 5: Final high-precision optimization with more iterations
    try:
        result = minimize(
            objective,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-13}  # Even tighter tolerances
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_min_max_ratio_jit(refined_points)
            if ratio > current_ratio:
                current_ratio = ratio
                current_points = refined_points.copy()
    except Exception:
        pass
    
    # Final physics refinement for ultimate improvement
    try:
        final_refined = physics_refinement(current_points, max_iter=100)
        ratio = compute_min_max_ratio_jit(final_refined)
        if ratio > current_ratio:
            current_points = final_refined.copy()
    except Exception:
        pass
    
    return current_points


# EVOLVE-BLOCK-END
