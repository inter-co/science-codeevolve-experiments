# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize, basinhopping, dual_annealing
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

@jit(nopython=True)
def compute_forces_enhanced_jit(points):
    """Enhanced force computation with better repulsion model using JIT"""
    n = points.shape[0]
    forces = np.zeros_like(points)
    
    # Vectorized computation for better performance
    # Compute all pairwise differences
    diff_matrix = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # Compute squared distances
    dist_sq_matrix = np.sum(diff_matrix**2, axis=2)
    
    # Avoid division by zero and very small distances
    dist_sq_matrix = np.where(dist_sq_matrix < 1e-16, 1e16, dist_sq_matrix)
    
    # Compute forces (inverse square law with better handling)
    # Add a small constant to prevent extreme forces at very close distances
    force_magnitudes = 1.0 / (dist_sq_matrix + 1e-16)
    
    # Zero out diagonal (self-interactions)
    np.fill_diagonal(force_magnitudes, 0)
    
    # Normalize and sum forces
    unit_vectors = diff_matrix / np.sqrt(dist_sq_matrix)[..., np.newaxis]
    forces = np.sum(force_magnitudes[..., np.newaxis] * unit_vectors, axis=1)
    
    # Apply enhanced damping to prevent numerical instability
    force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
    max_force = np.max(force_norms)
    if max_force > 10000:
        forces = forces / (max_force / 10000 + 1e-12)
    elif max_force > 1000:
        forces = forces / (max_force / 1000 + 1e-10)
    
    return forces

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining multiple optimization strategies, 
    advanced initialization techniques, and physics-inspired refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_ratio(points):
        """Compute the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
        return compute_min_max_ratio_jit(points)
    
    def fibonacci_sphere(samples=14):
        """Generate points on a sphere using Fibonacci spiral for good distribution"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(samples):
            # Distribute points uniformly on sphere
            y = 1 - (i / (samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Golden angle increment
            phi = ((i * golden_ratio) % samples) * (2 * np.pi / samples)
            
            x = radius * np.cos(phi)
            z = radius * np.sin(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def generate_initial_configs():
        """Generate multiple high-quality initial configurations"""
        configs = []
        np.random.seed(42)
        
        # Strategy 1: Icosahedral-based construction with mathematical precision
        try:
            phi = (1 + np.sqrt(5)) / 2  # golden ratio
            icosahedron_points = np.array([
                [0, 1, phi], [0, -1, phi], [0, -1, -phi], [0, 1, -phi],
                [1, phi, 0], [-1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [phi, 0, 1], [-phi, 0, 1], [-phi, 0, -1], [phi, 0, -1]
            ], dtype=float)
            
            # Normalize to unit sphere
            norms = np.linalg.norm(icosahedron_points, axis=1, keepdims=True)
            icosahedron_points = icosahedron_points / norms
            
            # Add two more points to reach 14 - strategic placement
            additional_points = np.array([
                [0, 0, 0.98],  # Near north pole
                [0, 0, -0.98]  # Near south pole
            ])
            
            # Perturb slightly to break symmetry and improve results
            ico_perturbed = icosahedron_points + np.random.normal(0, 0.02, icosahedron_points.shape)
            norms = np.linalg.norm(ico_perturbed, axis=1, keepdims=True)
            ico_perturbed = ico_perturbed / norms
            
            config1 = np.vstack([ico_perturbed, additional_points])
            configs.append(("icosahedral_math", config1))
        except Exception:
            pass
        
        # Strategy 2: Fibonacci-based spherical distribution with mathematical precision
        try:
            fib_points = []
            for i in range(14):
                # Using Fibonacci spiral on sphere with proper mathematical treatment
                y = 1 - (i / 13.0) * 2  # y from 1 to -1
                radius = np.sqrt(1 - y * y)
                # Golden angle spacing for uniform distribution
                theta = np.arccos(y) + (i * 2.399963229728653)  # Golden angle offset
                x = radius * np.cos(theta)
                z = radius * np.sin(theta)
                fib_points.append([x, y, z])
            
            config2 = np.array(fib_points)
            norms = np.linalg.norm(config2, axis=1, keepdims=True)
            config2 = config2 / norms
            configs.append(("fibonacci_math", config2))
        except Exception:
            pass
        
        # Strategy 3: Random but carefully constrained to sphere
        try:
            config3 = np.random.randn(14, 3)
            norms = np.linalg.norm(config3, axis=1, keepdims=True)
            config3 = config3 / norms
            configs.append(("random_sphere", config3))
        except Exception:
            pass
        
        # Strategy 4: Hybrid approach with symmetry breaking
        try:
            # Start with fibonacci points
            fib_points = fibonacci_sphere(14)
            # Add perturbations for better distribution
            perturbed = fib_points + np.random.normal(0, 0.05, fib_points.shape)
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            config4 = perturbed / norms
            configs.append(("hybrid_fibonacci", config4))
        except Exception:
            pass
        
        # Strategy 5: Additional cube-based configuration
        try:
            # Create a more structured starting point
            cube_vertices = np.array([
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
            ])
            
            # Add 6 more points at midpoints of edges
            additional = np.array([
                [0.5, 0.5, 0], [0.5, 0.5, 1],  # Midpoints of vertical edges
                [0.5, 0, 0.5], [0.5, 1, 0.5],  # Midpoints of horizontal edges
                [0, 0.5, 0.5], [1, 0.5, 0.5]   # Midpoints of front/back edges
            ])
            
            all_points = np.vstack([cube_vertices, additional])
            
            # Add remaining points randomly
            if len(all_points) < 14:
                extra = np.random.rand(14 - len(all_points), 3)
                all_points = np.vstack([all_points, extra])
            elif len(all_points) > 14:
                all_points = all_points[:14]
            
            # Normalize to unit sphere and map to [0,1]^3
            norms = np.linalg.norm(all_points, axis=1, keepdims=True)
            all_points = all_points / norms
            all_points = (all_points + 1) / 2
            
            configs.append(("cube_structured", all_points))
        except Exception:
            pass
        
        # Strategy 6: Polar configuration for better spread
        try:
            # Create points arranged along polar directions
            polar_points = []
            # Add points along z-axis (north and south poles)
            polar_points.extend([[0, 0, 0.95], [0, 0, -0.95]])
            # Add points in equatorial plane
            for i in range(12):
                angle = 2 * np.pi * i / 12
                polar_points.append([np.cos(angle)*0.5, np.sin(angle)*0.5, 0])
            
            polar_points = np.array(polar_points)
            # Normalize to unit sphere
            norms = np.linalg.norm(polar_points, axis=1, keepdims=True)
            polar_points = polar_points / norms
            configs.append(("polar_config", polar_points))
        except Exception:
            pass
        
        # Strategy 7: Tetrahedral-based configuration
        try:
            # Regular tetrahedron vertices plus 10 additional points
            tetrahedron_points = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
            ], dtype=float)
            
            # Normalize to unit sphere
            norms = np.linalg.norm(tetrahedron_points, axis=1, keepdims=True)
            tetrahedron_points = tetrahedron_points / norms
            
            # Add 10 more points distributed around
            additional = []
            for i in range(10):
                # Random points on sphere with some clustering
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.arccos(np.random.uniform(-1, 1))
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                additional.append([x, y, z])
            
            all_points = np.vstack([tetrahedron_points, additional])
            # Normalize to unit sphere
            norms = np.linalg.norm(all_points, axis=1, keepdims=True)
            all_points = all_points / norms
            configs.append(("tetrahedral", all_points))
        except Exception:
            pass
        
        return configs
    
    # Generate initial configurations
    initial_configs = generate_initial_configs()
    
    # Find the best initial configuration
    best_ratio = -np.inf
    best_points = None
    
    for name, config in initial_configs:
        ratio = compute_ratio(config)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = config.copy()
    
    # If no good initial configuration was found, create a fallback
    if best_points is None:
        points = np.random.rand(14, 3) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = points / norms
        best_points = (best_points + 1) / 2
    
    # Objective function for optimization
    def objective(x):
        points_test = x.reshape(-1, 3)
        return -compute_min_max_ratio_jit(points_test)
    
    # Multi-stage optimization approach with more sophisticated strategies
    current_points = best_points.copy()
    current_ratio = compute_ratio(current_points)
    
    # Strategy 1: Global optimization with differential evolution (more robust)
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=500,      # Increased iterations
            popsize=50,       # Increased population size
            mutation=(0.98, 1), # Higher mutation rate
            recombination=0.99, # Higher recombination rate
            seed=42,
            disp=False,
            maxfun=3000,      # More function evaluations
            tol=1e-15       # Stricter tolerance
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
    
    # Strategy 2: Simulated Annealing for broader exploration
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        result = dual_annealing(
            objective,
            bounds,
            maxiter=800,      # More iterations
            seed=42,
            no_local_search=True
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
    
    # Strategy 3: Basin hopping for global exploration with multiple restarts
    try:
        bounds = [(0, 1) for _ in range(14 * 3)]
        bh_result = basinhopping(
            objective, 
            current_points.flatten(), 
            niter=300,        # More iterations
            T=0.7,           # Slightly higher temperature for better exploration
            stepsize=0.06,   # Slightly larger step size
            minimizer_kwargs={'bounds': bounds, 'method': 'L-BFGS-B', 'options': {'ftol': 1e-15, 'gtol': 1e-15}}
        )
        
        candidate_points = bh_result.x.reshape(14, 3)
        candidate_points = np.clip(candidate_points, 0, 1)
        ratio = compute_ratio(candidate_points)
        if ratio > current_ratio:
            current_ratio = ratio
            current_points = candidate_points.copy()
    except Exception:
        pass
    
    # Strategy 4: Multiple local optimizations with diverse restarts and methods
    for restart in range(35):  # Even more restarts for better exploration
        try:
            np.random.seed(restart * 1000 + 42)
            
            # Perturbation strategy with better variance control
            if restart < 15:
                # Gaussian perturbation with moderate variance
                perturbed = current_points + np.random.normal(0, 0.018, current_points.shape)
            elif restart < 30:
                # Uniform perturbation for more exploration
                perturbed = current_points + np.random.uniform(-0.025, 0.025, current_points.shape)
            else:
                # Smaller perturbation for fine-tuning
                perturbed = current_points + np.random.normal(0, 0.006, current_points.shape)
            
            perturbed = np.clip(perturbed, 0, 1)
            
            # Use different optimization methods for diversity
            if restart % 6 == 0:
                # L-BFGS-B for fine-tuning with strict tolerances
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(14 * 3)],
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
            elif restart % 6 == 1:
                # SLSQP for constrained optimization
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1) for _ in range(14 * 3)],
                    options={'maxiter': 800, 'ftol': 1e-15}
                )
            elif restart % 6 == 2:
                # TNC for robustness
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method='TNC',
                    bounds=[(0, 1) for _ in range(14 * 3)],
                    options={'maxiter': 800, 'ftol': 1e-15}
                )
            elif restart % 6 == 3:
                # COBYLA for derivative-free optimization
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method='COBYLA',
                    bounds=[(0, 1) for _ in range(14 * 3)],
                    options={'maxiter': 800, 'tol': 1e-15}
                )
            elif restart % 6 == 4:
                # trust-constr for robust optimization
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method='trust-constr',
                    bounds=[(0, 1) for _ in range(14 * 3)],
                    options={'maxiter': 700, 'gtol': 1e-15, 'ftol': 1e-15}
                )
            else:
                # Nelder-Mead as backup
                result = minimize(
                    objective,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'fatol': 1e-15, 'xatol': 1e-15}
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
    
    # Strategy 5: Enhanced physics-inspired iterative improvement with JIT
    try:
        points_refined = current_points.copy()
        learning_rate = 0.008  # Slightly higher learning rate
        max_iterations = 2500  # More iterations
        prev_ratio = -np.inf
        patience_counter = 0
        patience_limit = 80  # More patience
        
        # Adaptive learning rate decay and early stopping
        for iteration in range(max_iterations):
            forces = compute_forces_enhanced_jit(points_refined)
            points_refined += learning_rate * forces
            points_refined = np.clip(points_refined, 0, 1)
            
            # Adaptive learning rate decrease
            if iteration > 0 and iteration % 500 == 0:
                learning_rate *= 0.7  # More aggressive decay
            
            # Check convergence with patience
            if iteration % 25 == 0:
                ratio = compute_ratio(points_refined)
                if abs(ratio - prev_ratio) < 1e-17:
                    patience_counter += 1
                else:
                    patience_counter = 0
                prev_ratio = ratio
                
                if patience_counter >= patience_limit:
                    break  # Converged or stalled
        
        # Final validation - check if the refined version is better
        final_ratio = compute_ratio(points_refined)
        if final_ratio > current_ratio:
            current_points = points_refined
    except Exception:
        pass
    
    # Strategy 6: Final polishing with multiple methods
    try:
        # Try different optimization methods for final refinement
        # First, try L-BFGS-B with very strict tolerances
        result = minimize(
            objective,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 2500, 'ftol': 1e-18, 'gtol': 1e-18}
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_ratio(refined_points)
            if ratio > current_ratio:
                current_ratio = ratio
                current_points = refined_points.copy()
                
        # If still not optimal, try with SLSQP as backup
        if current_ratio < 0.49:  # Only do extra work if we're not close to target
            result = minimize(
                objective,
                current_points.flatten(),
                method='SLSQP',
                bounds=[(0, 1) for _ in range(14 * 3)],
                options={'maxiter': 1500, 'ftol': 1e-17, 'gtol': 1e-17}
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
    
    # Strategy 7: Extra refinement with trust-constr and COBYLA
    try:
        # Try trust-constr
        result = minimize(
            objective,
            current_points.flatten(),
            method='trust-constr',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 1200, 'gtol': 1e-18, 'ftol': 1e-18}
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_ratio(refined_points)
            if ratio > current_ratio:
                current_ratio = ratio
                current_points = refined_points.copy()
                
        # Try COBYLA as final backup
        result = minimize(
            objective,
            current_points.flatten(),
            method='COBYLA',
            bounds=[(0, 1) for _ in range(14 * 3)],
            options={'maxiter': 1200, 'tol': 1e-17}
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
    
    # Strategy 8: Additional Nelder-Mead refinement for robustness
    try:
        result = minimize(
            objective,
            current_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-17, 'xatol': 1e-17}
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
    
    # Map from [0,1]^3 to [0,1]^3 (already in correct range)
    return current_points


# EVOLVE-BLOCK-END
