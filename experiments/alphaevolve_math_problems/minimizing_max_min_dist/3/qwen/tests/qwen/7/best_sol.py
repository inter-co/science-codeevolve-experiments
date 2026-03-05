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

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum distance between all point pairs."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist <= 0:
        return 0.0
    return min_dist / max_dist

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

def generate_initial_configurations():
    """Generate multiple diverse initial configurations"""
    configs = []
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedral configuration - base structure
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
    
    # Add 2 more points to make 14 total - strategically placed
    icosahedral_points.extend([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    configs.append(("icosahedral", np.array(icosahedral_points)))
    
    # Fibonacci spiral configuration (for variety)
    fib_points = []
    for i in range(14):
        phi_angle = np.arccos(-1 + 2*i/13)
        theta = np.sqrt(14 * np.pi) * i
        x = np.sin(phi_angle) * np.cos(theta)
        y = np.sin(phi_angle) * np.sin(theta)
        z = np.cos(phi_angle)
        fib_points.append([(x + 1)/2, (y + 1)/2, (z + 1)/2])
    configs.append(("fibonacci", np.array(fib_points)))
    
    # Random configuration with fixed seed for reproducibility
    np.random.seed(42)
    random_points = np.random.rand(14, 3)
    configs.append(("random", random_points))
    
    # Perturbed icosahedral configuration
    perturbed = np.array(icosahedral_points) + np.random.normal(0, 0.03, (14, 3))
    # Ensure points stay in [0,1]^3
    perturbed = np.clip(perturbed, 0, 1)
    configs.append(("perturbed_icosahedral", perturbed))
    
    # Octahedral configuration (additional diversity)
    octahedral_points = []
    octahedron_vertices = [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ]
    for v in octahedron_vertices:
        norm = np.linalg.norm(v)
        octahedral_points.append([(v[0]/norm + 1)/2, (v[1]/norm + 1)/2, (v[2]/norm + 1)/2])
    
    # Add 8 more points for 14 total, distributed evenly
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = np.cos(angle) * 0.7
        y = np.sin(angle) * 0.7
        z = 0.4 * np.sin(np.pi * i / 4)  # Vary z to distribute points
        octahedral_points.append([(x + 1)/2, (y + 1)/2, (z + 1)/2])
    configs.append(("octahedral", np.array(octahedral_points)))
    
    return configs

def physics_based_refinement(initial_points, max_iter=200, learning_rate=0.002):
    """Refine points using physics-based iterative improvement with enhanced forces"""
    points = initial_points.copy()
    best_ratio = compute_min_max_ratio_jit(points)
    best_points = points.copy()
    
    # Track improvement for early stopping
    last_improvement = 0
    patience_counter = 0
    patience_limit = 30
    
    for iteration in range(max_iter):
        forces = compute_forces_enhanced_jit(points)
        points += learning_rate * forces
        
        # Project back to unit cube [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Occasionally check for improvement
        if iteration % 20 == 0:
            current_ratio = compute_min_max_ratio_jit(points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
                last_improvement = iteration
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping if no improvement for too long
            if patience_counter >= patience_limit:
                break
    
    return best_points

def objective_function(x):
    """Objective function to maximize: negative of min/max ratio"""
    points = x.reshape(14, 3)
    return -compute_min_max_ratio_jit(points)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses hybrid approach combining mathematical constructions with advanced optimization.
    """
    
    # Generate multiple initial configurations
    initial_configs = generate_initial_configurations()
    
    # Find the best initial configuration
    best_ratio = -np.inf
    best_points = None
    
    for name, config in initial_configs:
        ratio = compute_min_max_ratio_jit(config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = config.copy()
    
    # Multi-stage optimization approach inspired by INSPIRATION 1 and 2
    current_points = best_points.copy()
    current_ratio = compute_min_max_ratio_jit(current_points)
    
    # Stage 1: Differential Evolution for global optimization with aggressive parameters
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=150,      # More iterations for better global search
            popsize=30,       # Larger population for better diversity
            mutation=(0.9, 1), # Higher mutation for more exploration
            recombination=0.95, # Better mixing
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
            objective_function, 
            current_points.flatten(), 
            niter=50,        # More iterations for better exploration
            T=0.4,           # Higher temperature for better escape from local minima
            stepsize=0.05,   # Larger step size for broader search
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
        physics_refined = physics_based_refinement(current_points, max_iter=200)
        ratio = compute_min_max_ratio_jit(physics_refined)
        if ratio > current_ratio:
            current_ratio = ratio
            current_points = physics_refined.copy()
    except Exception:
        pass
    
    # Stage 4: Multiple restarts with diverse strategies and tighter tolerances
    for restart in range(10):  # Balanced number of restarts
        try:
            np.random.seed(restart * 1000 + 42)
            # Diverse perturbation strategies
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
            
            # Different optimization methods for diversity
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            method = methods[restart % 3]
            
            result = minimize(
                objective_function,
                perturbed.flatten(),
                method=method,
                bounds=[(0, 1) for _ in range(14 * 3)],
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}
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
            objective_function,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-13}
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
        final_refined = physics_based_refinement(current_points, max_iter=100)
        ratio = compute_min_max_ratio_jit(final_refined)
        if ratio > current_ratio:
            current_points = final_refined.copy()
    except Exception:
        pass
    
    return current_points


# EVOLVE-BLOCK-END
