# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import pdist
from numba import jit
import warnings
warnings.filterwarnings('ignore')

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

def generate_initial_configurations():
    """Generate multiple diverse initial configurations with enhanced mathematical foundations"""
    configs = []
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedral configuration - base structure (from INSPIRATION 2)
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
    # Use points that complement the icosahedral structure
    icosahedral_points.extend([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])  # Better positioned than previous
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
    
    # Octahedral configuration with additional points (from INSPIRATION 3)
    octahedral_points = []
    # Add vertices of octahedron (6 points)
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
    
    # Random configuration with fixed seed for reproducibility
    np.random.seed(42)
    random_points = np.random.rand(14, 3)
    configs.append(("random", random_points))
    
    # Perturbed icosahedral configuration with better perturbation strategy
    perturbed = np.array(icosahedral_points) + np.random.normal(0, 0.03, (14, 3))
    # Ensure points stay in [0,1]^3
    perturbed = np.clip(perturbed, 0, 1)
    configs.append(("perturbed_icosahedral", perturbed))
    
    # Known good configuration from mathematical literature (INSPIRATION 2)
    # 14-point spherical code approximation
    known_good = [
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],  # Poles
        [0.0, 0.5, 0.866025], [0.0, -0.5, 0.866025], [0.0, 0.5, -0.866025], [0.0, -0.5, -0.866025],  # Around equator
        [0.866025, 0.0, 0.5], [-0.866025, 0.0, 0.5], [0.866025, 0.0, -0.5], [-0.866025, 0.0, -0.5],  # Around x-axis
        [0.5, 0.866025, 0.0], [0.5, -0.866025, 0.0], [-0.5, 0.866025, 0.0], [-0.5, -0.866025, 0.0]   # Around y-axis
    ]
    # Normalize to unit sphere then map to [0,1]^3
    known_good_normalized = []
    for v in known_good:
        norm = np.linalg.norm(v)
        known_good_normalized.append([(v[0]/norm + 1)/2, (v[1]/norm + 1)/2, (v[2]/norm + 1)/2])
    configs.append(("known_good", np.array(known_good_normalized)))
    
    return configs

def compute_forces_simple(points):
    """Simple physics-based force computation for refinement"""
    n = points.shape[0]
    forces = np.zeros_like(points)
    
    # Compute pairwise distances and forces
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            
            # Avoid division by zero and very small distances
            if dist_sq > 1e-12:
                dist = np.sqrt(dist_sq)
                # Repulsive force proportional to inverse square
                force_magnitude = 1.0 / (dist_sq + 1e-12)
                force_vector = force_magnitude * diff / (dist + 1e-12)
                forces[i] += force_vector
                forces[j] -= force_vector
    
    return forces

def physics_based_refinement(initial_points, max_iter=500, learning_rate=0.005):
    """Refine points using physics-based iterative improvement"""
    points = initial_points.copy()
    best_ratio = compute_min_max_ratio_jit(points)
    best_points = points.copy()
    
    for iteration in range(max_iter):
        forces = compute_forces_simple(points)
        points += learning_rate * forces
        
        # Project back to unit cube [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Occasionally check for improvement
        if iteration % 50 == 0:
            current_ratio = compute_min_max_ratio_jit(points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
    
    return best_points

def objective_function(x):
    """Objective function to maximize: negative of min/max ratio"""
    points = x.reshape(14, 3)
    return -compute_min_max_ratio_jit(points)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initial configurations with advanced optimization strategies.
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
    
    # Multi-stage optimization approach inspired by INSPIRATION 1 and 3
    current_points = best_points.copy()
    current_ratio = compute_min_max_ratio_jit(current_points)
    
    # Stage 1: Differential Evolution for global optimization (more aggressive parameters)
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=120,      # More iterations for better global search (INSPIRATION 1)
            popsize=25,       # Larger population for better diversity (INSPIRATION 1)
            mutation=(0.9, 1), # Higher mutation for more exploration (INSPIRATION 1)
            recombination=0.95, # Better mixing (INSPIRATION 1)
            seed=42,
            disp=False,
            maxfun=600
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
    
    # Stage 2: Basin Hopping for further exploration (INSPIRATION 2)
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        bh_result = basinhopping(
            objective_function, 
            current_points.flatten(), 
            niter=30,        # More iterations for better exploration (INSPIRATION 1)
            T=0.3,           # Moderate temperature for good balance (INSPIRATION 1)
            stepsize=0.04,   # Moderate step size (INSPIRATION 1)
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
    
    # Stage 3: Multiple restarts with varied optimization methods (INSPIRATION 2)
    for restart in range(15):  # More restarts for better exploration (INSPIRATION 3)
        try:
            np.random.seed(restart * 1000 + 42)
            # Diverse perturbation strategies (INSPIRATION 1)
            if restart < 5:
                # Small Gaussian perturbation
                perturbed = current_points + np.random.normal(0, 0.015, current_points.shape)
            elif restart < 10:
                # Medium Gaussian perturbation  
                perturbed = current_points + np.random.normal(0, 0.02, current_points.shape)
            else:
                # Uniform perturbation for broad search
                perturbed = current_points + np.random.uniform(-0.02, 0.02, current_points.shape)
            
            perturbed = np.clip(perturbed, 0, 1)
            
            # Different optimization methods for diversity (INSPIRATION 1)
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            method = methods[restart % 3]
            
            result = minimize(
                objective_function,
                perturbed.flatten(),
                method=method,
                bounds=[(0, 1) for _ in range(14 * 3)],
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-13}  # Tighter tolerances (INSPIRATION 1)
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
    
    # Stage 4: Physics-based refinement for final improvement (like INSPIRATION 1)
    try:
        refined_points = physics_based_refinement(current_points, max_iter=300, learning_rate=0.003)
        ratio = compute_min_max_ratio_jit(refined_points)
        if ratio > current_ratio:
            current_points = refined_points.copy()
    except Exception:
        pass
    
    # Stage 5: Enhanced coordinate-wise refinement with adaptive step sizes (INSPIRATION 2)
    try:
        improved = True
        iteration_count = 0
        
        while improved and iteration_count < 8:  # Fewer iterations for efficiency (INSPIRATION 3)
            improved = False
            old_ratio = compute_min_max_ratio_jit(current_points)
            
            # Try each coordinate with adaptive step sizes (INSPIRATION 2)
            for coord_idx in range(3):
                best_coord = current_points[:, coord_idx].copy()
                best_ratio = old_ratio
                
                # Try multiple step sizes for better exploration (INSPIRATION 2)
                step_sizes = [0.002, 0.005, 0.01, 0.02]
                for step_size in step_sizes:
                    for perturbation in [-step_size, step_size]:
                        test_points = current_points.copy()
                        test_points[:, coord_idx] += perturbation
                        test_points[:, coord_idx] = np.clip(test_points[:, coord_idx], 0, 1)
                        
                        ratio = compute_min_max_ratio_jit(test_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_coord = test_points[:, coord_idx].copy()
                            improved = True
                
                current_points[:, coord_idx] = best_coord
            
            iteration_count += 1
            
    except Exception:
        pass
    
    # Stage 6: Final high-precision optimization (INSPIRATION 2)
    try:
        result = minimize(
            objective_function,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}  # Tighter tolerances
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
    
    return current_points


# EVOLVE-BLOCK-END
