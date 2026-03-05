# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')

def _compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute the minimum and maximum distances between all point pairs."""
    if len(points) < 2:
        return 0
    distances = pdist(points)
    dmin = np.min(distances)
    dmax = np.max(distances)
    return dmin / dmax if dmax > 0 else 0

def _compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute full distance matrix for given points."""
    return squareform(pdist(points))

def _icosahedral_plus_symmetric_construction() -> np.ndarray:
    """
    Construct points using icosahedral symmetry plus additional symmetric points.
    Based on mathematical principles of optimal point distributions.
    """
    # Start with icosahedron vertices (normalized to unit sphere)
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    # Convert to unit cube coordinates [0,1]^3
    ico_verts = (ico_verts + 1) / 2
    
    # Add strategic points for 14 total - more carefully chosen for better distribution
    additional_points = np.array([
        [0.5, 0.5, 0.5],  # center
        [0.25, 0.25, 0.25],  # corner
        [0.75, 0.75, 0.75],  # opposite corner
        [0.5, 0.5, 0.0],   # face center
        [0.5, 0.0, 0.5],   # face center
        [0.0, 0.5, 0.5],   # face center
        [0.5, 1.0, 0.5],   # additional face center
        [1.0, 0.5, 0.5],   # additional face center
        [0.5, 0.5, 1.0],   # additional face center
        [0.1, 0.1, 0.1],   # additional corner
        [0.9, 0.9, 0.9],   # additional corner
        [0.0, 0.0, 0.0],   # corner
        [1.0, 1.0, 1.0],   # corner
        [0.25, 0.75, 0.25], # diagonal
        [0.75, 0.25, 0.75], # diagonal
        [0.25, 0.25, 0.75], # diagonal
        [0.1, 0.9, 0.1],   # anti-diagonal
        [0.9, 0.1, 0.9],   # anti-diagonal
    ])
    
    # Combine icosahedral points with additional strategic points
    initial_points = np.vstack([ico_verts, additional_points[:2]])
    
    # If we have more than 14, select optimally
    if len(initial_points) > 14:
        # Greedy selection to maximize minimum distance
        selected_indices = [0]  # Start with first point
        remaining_indices = list(range(1, len(initial_points)))
        
        while len(selected_indices) < 14:
            best_idx = -1
            best_min_dist = -1
            
            for idx in remaining_indices:
                min_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = np.linalg.norm(initial_points[idx] - initial_points[sel_idx])
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        points = initial_points[selected_indices]
    else:
        points = initial_points[:14]
    
    return points

def _spherical_code_construction() -> np.ndarray:
    """
    Construct points based on spherical code principles.
    Uses principles similar to Thomson problem solutions.
    """
    # Start with icosahedral construction as base
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Convert to [0,1]^3
    vertices = (vertices + 1) / 2
    
    # Add more points that maintain good distribution
    additional = np.array([
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.5, 1.0], [0.5, 1.0, 0.5], [1.0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
        [0.1, 0.1, 0.1], [0.9, 0.9, 0.9],
        [0.3, 0.3, 0.7], [0.7, 0.7, 0.3],
        [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
        [0.2, 0.2, 0.8], [0.8, 0.8, 0.2], [0.2, 0.8, 0.2],
        [0.8, 0.2, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.8]
    ])
    
    # Combine and select 14 optimally
    all_points = np.vstack([vertices, additional])
    
    # Use greedy selection to maximize minimum distance
    if len(all_points) > 14:
        selected_indices = [0]
        remaining_indices = list(range(1, len(all_points)))
        
        while len(selected_indices) < 14:
            best_idx = -1
            best_min_dist = -1
            
            for idx in remaining_indices:
                min_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = np.linalg.norm(all_points[idx] - all_points[sel_idx])
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        points = all_points[selected_indices]
    else:
        points = all_points[:14]
    
    return points

def _fibonacci_spiral_construction() -> np.ndarray:
    """
    Construct points using Fibonacci spiral approach for good spherical distribution.
    """
    n = 14
    points = np.zeros((n, 3))
    
    # Fibonacci spiral method with better distribution
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        # Improved angle calculation for better distribution
        theta = np.arccos(y)  # polar angle
        golden_ratio = (1 + np.sqrt(5)) / 2
        phi_i = (i * 2 * np.pi) / golden_ratio  # azimuthal angle
        
        points[i, 0] = radius * np.cos(phi_i)
        points[i, 1] = radius * np.sin(phi_i)
        points[i, 2] = y
    
    # Normalize to unit sphere
    points = points / np.linalg.norm(points[0]) if np.linalg.norm(points[0]) != 0 else points
    
    # Add small random perturbations
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    # Renormalize
    for i in range(len(points)):
        norm = np.linalg.norm(points[i])
        if norm > 0:
            points[i] = points[i] / norm
            
    # Convert to [0,1]^3
    points = (points + 1) / 2
    
    return points

def _voronoi_refinement(points: np.ndarray, max_iterations: int = 3) -> np.ndarray:
    """
    Refine point positions using Voronoi-based approach.
    """
    try:
        current_points = points.copy()
        for _ in range(max_iterations):
            vor = Voronoi(current_points)
            
            # For each point, compute the centroid of its Voronoi cell
            new_points = []
            for i in range(len(current_points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    # Get vertices of Voronoi cell
                    cell_vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(cell_vertices) > 0:
                        # Compute centroid of Voronoi cell
                        centroid = np.mean(cell_vertices, axis=0)
                        # Clip to unit cube
                        centroid = np.clip(centroid, 0, 1)
                        new_points.append(centroid)
                    else:
                        new_points.append(current_points[i])
                else:
                    new_points.append(current_points[i])
            
            new_points = np.array(new_points)
            
            # Check if change is small enough to stop early
            if np.allclose(current_points, new_points, atol=1e-14):
                break
                
            current_points = new_points
            
        return current_points
    except Exception:
        return points

def _energy_minimization_refinement(points: np.ndarray) -> np.ndarray:
    """
    Refine point distribution using energy minimization approach.
    """
    def energy_function(x):
        points = x.reshape(-1, 3)
        distances = pdist(points)
        # Use inverse square for repulsion energy (similar to Coulomb potential)
        # But avoid very small distances that would cause numerical issues
        distances = np.maximum(distances, 1e-8)
        energy = np.sum(1.0 / (distances ** 2))
        return energy
    
    def objective(x):
        points = x.reshape(-1, 3)
        # We want to maximize the ratio, so minimize negative ratio
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        dmin = np.min(distances)
        dmax = np.max(distances)
        if dmax == 0:
            return 0
        # Minimize negative ratio to maximize ratio
        return -dmin / dmax
    
    try:
        # Try optimization with bounds [0,1] for each coordinate
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(42)]
        
        # Use multiple optimization methods for robustness
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        best_points = points.copy()
        best_ratio = _compute_min_max_ratio(points)
        
        for method in methods:
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}  # Increased precision
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    ratio = _compute_min_max_ratio(refined_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
            except Exception:
                continue
                
        return best_points
    except Exception:
        return points

def _constraint_satisfaction_step(points: np.ndarray, max_iterations: int = 1000, 
                                 step_size: float = 0.02) -> np.ndarray:
    """
    Use constraint satisfaction approach to iteratively improve point distribution.
    This helps escape local minima and find better configurations.
    """
    current_points = points.copy()
    ratio = _compute_min_max_ratio(current_points)
    
    for iteration in range(max_iterations):
        prev_ratio = ratio
        
        # For each point, try to improve by moving it slightly
        for i in range(len(current_points)):
            original_point = current_points[i].copy()
            best_point = original_point.copy()
            best_ratio = ratio
            
            # Try several candidate positions near current point
            for _ in range(200):  # More attempts for better search
                # Generate random perturbation with adaptive size
                perturbation_size = step_size if iteration < 500 else step_size * 0.5
                perturbation = np.random.normal(0, perturbation_size, 3)
                candidate_point = original_point + perturbation
                
                # Keep within bounds
                candidate_point = np.clip(candidate_point, 0, 1)
                
                # Test this candidate
                test_points = current_points.copy()
                test_points[i] = candidate_point
                
                new_ratio = _compute_min_max_ratio(test_points)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_point = candidate_point.copy()
                    
            # Update if we found an improvement
            if best_ratio > ratio:
                current_points[i] = best_point
                ratio = best_ratio
        
        # Early stopping if improvement is minimal
        if abs(ratio - prev_ratio) < 1e-16:
            break
    
    return current_points

def _spectral_geometry_optimization(points: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """
    Optimization using spectral geometry principles.
    Maximizes the spectral gap of the distance matrix to encourage uniform distribution.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = _compute_min_max_ratio(current_points)
    
    # Parameters for optimization
    step_size = 0.01
    decay_rate = 0.995
    
    for iteration in range(max_iter):
        # Compute current ratio
        ratio = _compute_min_max_ratio(current_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = current_points.copy()
        
        # Simple gradient approximation using small perturbations
        grad = np.zeros_like(current_points)
        eps = 1e-5
        for i in range(len(current_points)):
            for j in range(3):  # x, y, z components
                # Perturb point i component j
                perturbed = current_points.copy()
                perturbed[i, j] += eps
                ratio_pert = _compute_min_max_ratio(perturbed)
                
                # Finite difference gradient
                grad[i, j] = (ratio_pert - ratio) / eps
        
        # Update using gradient ascent (we want to maximize ratio)
        current_points += step_size * grad
        
        # Project back to valid domain
        current_points = np.clip(current_points, 0, 1)
        
        # Reduce step size over time
        step_size *= decay_rate
    
    return best_points

def _refinement_pipeline(points: np.ndarray) -> np.ndarray:
    """
    Apply a comprehensive refinement pipeline for better results.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = _compute_min_max_ratio(current_points)
    
    # Apply multiple refinement techniques in sequence with better convergence control
    refinements = [
        lambda p: _energy_minimization_refinement(p),
        lambda p: _voronoi_refinement(p, max_iterations=3),
        lambda p: _constraint_satisfaction_step(p, max_iterations=800, step_size=0.01),
        lambda p: _energy_minimization_refinement(p),
        lambda p: _spectral_geometry_optimization(p, max_iter=300),
    ]
    
    for refinement_func in refinements:
        try:
            refined_points = refinement_func(current_points)
            ratio = _compute_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
            current_points = refined_points.copy()
        except Exception:
            continue
    
    return best_points

def _global_optimization_approach(points: np.ndarray) -> np.ndarray:
    """
    Aggressive global optimization using differential evolution with optimized parameters.
    """
    def objective(x):
        """Objective function: minimize negative of min/max ratio"""
        points = x.reshape(-1, 3)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        dmin = np.min(distances)
        dmax = np.max(distances)
        if dmax == 0:
            return 0
        return -dmin / dmax
    
    try:
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(42)]
        
        # Use differential evolution with optimized parameters for faster convergence
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=50,  # Optimized iterations
            popsize=30,   # Larger population for better exploration
            seed=42,
            disp=False,
            tol=1e-12
        )
        
        if de_result.success:
            refined_points = de_result.x.reshape(-1, 3)
            return refined_points
        return points
    except Exception:
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions, energy minimization,
    and robust optimization techniques.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Enhanced icosahedral plus symmetric construction
    try:
        points = _icosahedral_plus_symmetric_construction()
        ratio = _compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception:
        pass
    
    # Strategy 2: Enhanced spherical code construction
    try:
        points = _spherical_code_construction()
        ratio = _compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception:
        pass
    
    # Strategy 3: Fibonacci spiral construction
    try:
        points = _fibonacci_spiral_construction()
        ratio = _compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception:
        pass
    
    # Strategy 4: Multiple random restarts with better initialization
    for seed in [42, 123, 456, 789, 999]:
        np.random.seed(seed)
        # Start with better structured initialization
        points = np.random.rand(14, 3)
        ratio = _compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Strategy 5: Global optimization with differential evolution for robustness
    if best_points is not None:
        try:
            refined_points = _global_optimization_approach(best_points)
            ratio = _compute_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception:
            pass
    
    # Strategy 6: Multi-stage refinement pipeline with enhanced techniques
    if best_points is not None:
        try:
            refined_points = _refinement_pipeline(best_points)
            ratio = _compute_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception:
            pass
    
    # Strategy 7: High-precision local optimization with more aggressive settings
    if best_points is not None:
        try:
            # Try another round with very tight tolerances and more iterations
            x0 = best_points.flatten()
            result = minimize(
                lambda x: -_compute_min_max_ratio(x.reshape(-1, 3)), 
                x0, 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(42)],
                options={'maxiter': 5000, 'ftol': 1e-18}  # Very high precision
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                ratio = _compute_min_max_ratio(refined_points)
                if ratio > best_ratio:
                    best_points = refined_points
        except Exception:
            pass
    
    # Strategy 8: Final multi-pass refinement with different strategies
    if best_points is not None:
        try:
            # Run multiple refinement passes with different approaches
            current_points = best_points.copy()
            for i in range(5):
                # Alternate between different refinement methods with increasing intensity
                if i % 4 == 0:
                    current_points = _energy_minimization_refinement(current_points)
                elif i % 4 == 1:
                    current_points = _voronoi_refinement(current_points, max_iterations=2)
                elif i % 4 == 2:
                    current_points = _constraint_satisfaction_step(current_points, max_iterations=500, step_size=0.005)
                else:
                    current_points = _spectral_geometry_optimization(current_points, max_iter=200)
                ratio = _compute_min_max_ratio(current_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = current_points.copy()
        except Exception:
            pass
    
    # Strategy 9: Final ultra-precise optimization
    if best_points is not None:
        try:
            # Final ultra-precise optimization with high tolerance
            x0 = best_points.flatten()
            result = minimize(
                lambda x: -_compute_min_max_ratio(x.reshape(-1, 3)), 
                x0, 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(42)],
                options={'maxiter': 3000, 'ftol': 1e-19}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                ratio = _compute_min_max_ratio(refined_points)
                if ratio > best_ratio:
                    best_points = refined_points
        except Exception:
            pass
    
    # Ensure we have valid output even if optimization fails
    if best_points is None:
        # Fallback to a reasonable configuration
        points = np.random.rand(14, 3)
        return points
    
    # Final validation
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
