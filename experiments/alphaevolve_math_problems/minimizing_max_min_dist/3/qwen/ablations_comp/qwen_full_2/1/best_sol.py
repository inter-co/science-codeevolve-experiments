# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

def _energy_minimization_approach(points: np.ndarray, iterations: int = 500) -> np.ndarray:
    """
    Apply energy minimization to spread points more uniformly.
    Simulates electrostatic repulsion between points with improved physics.
    """
    current_points = points.copy()
    
    # Use a more sophisticated energy model with better scaling
    for iteration in range(iterations):
        forces = np.zeros_like(current_points)
        
        # Calculate forces between all pairs
        for i in range(len(current_points)):
            for j in range(i+1, len(current_points)):
                diff = current_points[i] - current_points[j]
                dist_sq = np.sum(diff**2)
                
                # Avoid division by zero and use inverse square law with better scaling
                if dist_sq > 1e-15:
                    force_magnitude = 1.0 / (dist_sq + 1e-15)**1.8
                    force = force_magnitude * diff
                    forces[i] += force
                    forces[j] -= force
        
        # Apply forces to move points (with damping)
        step_size = 0.005  # Slightly larger step size
        current_points += step_size * forces
        
        # Keep within bounds
        current_points = np.clip(current_points, 0, 1)
    
    return current_points

def _adaptive_constraint_satisfaction(points: np.ndarray, max_iterations: int = 500) -> np.ndarray:
    """
    Enhanced constraint satisfaction with adaptive perturbation sizes and better convergence.
    """
    current_points = points.copy()
    current_ratio = _compute_min_max_ratio(current_points)
    
    # Track best solution found so far
    best_points = current_points.copy()
    best_ratio = current_ratio
    
    for iteration in range(max_iterations):
        prev_ratio = current_ratio
        
        # For each point, try to improve by moving it
        improved = False
        
        # Randomize order for better exploration
        point_order = list(range(len(current_points)))
        np.random.shuffle(point_order)
        
        for i in point_order:
            original_point = current_points[i].copy()
            best_point = original_point.copy()
            best_ratio = current_ratio
            
            # Adaptive search with different perturbation sizes based on iteration
            search_attempts = 50  # More searches for better exploration
            if iteration < 100:
                perturbation_size = 0.04
            elif iteration < 300:
                perturbation_size = 0.02
            else:
                perturbation_size = 0.01
            
            for _ in range(search_attempts):
                # Generate random perturbation with adaptive size
                perturbation = np.random.normal(0, perturbation_size, 3)
                candidate_point = original_point + perturbation
                
                # Keep within bounds
                candidate_point = np.clip(candidate_point, 0, 1)
                
                # Test this candidate
                test_points = current_points.copy()
                test_points[i] = candidate_point
                
                ratio = _compute_min_max_ratio(test_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_point = candidate_point.copy()
                    
            # Update if we found an improvement
            if best_ratio > current_ratio:
                current_points[i] = best_point
                current_ratio = best_ratio
                improved = True
                # Update best solution found
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
        
        # If no improvement, perturb all points to escape local minima
        if not improved:
            for i in range(len(current_points)):
                if np.random.random() < 0.3:  # Higher chance to perturb
                    perturbation = np.random.normal(0, 0.005, 3)  # Slightly larger perturbation
                    current_points[i] = np.clip(current_points[i] + perturbation, 0, 1)
            
            current_ratio = _compute_min_max_ratio(current_points)
        
        # Early stopping if improvement is minimal
        if abs(current_ratio - prev_ratio) < 1e-14:
            break
    
    return best_points

def _local_search_improvement(points: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """
    Perform local search improvement to fine-tune the configuration.
    """
    current_points = points.copy()
    current_ratio = _compute_min_max_ratio(current_points)
    
    for iteration in range(max_iterations):
        # Try small moves for each point
        for i in range(len(current_points)):
            original_point = current_points[i].copy()
            best_point = original_point.copy()
            best_ratio = current_ratio
            
            # Try multiple small perturbations with more thorough search
            for _ in range(100):  # More thorough search
                # Very small perturbation
                perturbation = np.random.normal(0, 0.0005, 3)
                candidate_point = original_point + perturbation
                
                # Keep within bounds
                candidate_point = np.clip(candidate_point, 0, 1)
                
                # Test this candidate
                test_points = current_points.copy()
                test_points[i] = candidate_point
                
                ratio = _compute_min_max_ratio(test_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_point = candidate_point.copy()
                    
            # Update if we found an improvement
            if best_ratio > current_ratio:
                current_points[i] = best_point
                current_ratio = best_ratio
    
    return current_points

def _icosahedral_plus_center_construction() -> np.ndarray:
    """
    Construct points using icosahedral symmetry plus strategic additional points.
    This leverages known optimal configurations for symmetric point distributions.
    """
    # Icosahedron vertices (normalized to unit sphere)
    phi = (1 + np.sqrt(5)) / 2
    
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
    
    # Add center point and a few strategically placed points
    center = np.array([[0.5, 0.5, 0.5]])
    
    # Add 2 more points that are well-distributed
    # Use points that maximize spread from existing points
    additional = np.array([
        [0.2, 0.2, 0.2],      # corner
        [0.8, 0.8, 0.8]       # opposite corner
    ])
    
    # Combine all points
    all_points = np.vstack([ico_verts, center, additional])
    
    # Ensure we have exactly 14 points
    if len(all_points) > 14:
        # Use greedy selection to maximize minimum distance
        selected_indices = [0]  # Start with first point
        remaining_indices = list(range(1, len(all_points)))
        
        while len(selected_indices) < 14:
            # Find point that maximizes minimum distance to selected points
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

def _fibonacci_sphere_construction() -> np.ndarray:
    """
    Construct points using Fibonacci spiral on sphere for good distribution.
    """
    n = 14
    points = []
    
    # Fibonacci spiral on sphere
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        # Improved distribution with better spacing
        phi = np.arccos(1 - 2 * (i / (n - 1)))  # Polar angle
        theta = i * golden_ratio  # Azimuthal angle
        
        # Add more randomness to break perfect patterns
        theta += np.random.normal(0, 0.15)
        phi += np.random.normal(0, 0.08)
        
        # Clamp angles to valid ranges
        phi = np.clip(phi, 0, np.pi)
        theta = theta % (2 * np.pi)
        
        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Map to [0,1]^3
    points = (points + 1) / 2
    
    return points

def _objective(x):
    """Objective function: minimize negative of min/max ratio"""
    # Reshape flat array back to 3D points
    points = x.reshape(-1, 3)
    
    # Compute pairwise distances
    distances = pdist(points)
    
    # Avoid division by zero
    if len(distances) == 0:
        return 0
        
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Return negative ratio (since we want to maximize ratio)
    if max_dist == 0:
        return 0
    return -min_dist / max_dist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions, energy minimization,
    and robust optimization techniques.
    """
    
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Icosahedral construction (high-quality starting point)
    try:
        points = _icosahedral_plus_center_construction()
        points = _energy_minimization_approach(points, iterations=300)
        ratio = _compute_min_max_ratio(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception:
        pass
    
    # Strategy 2: Fibonacci sphere construction for alternative good starting point
    try:
        points = _fibonacci_sphere_construction()
        points = _energy_minimization_approach(points, iterations=300)
        ratio = _compute_min_max_ratio(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception:
        pass
    
    # Strategy 3: Multiple random restarts with better initialization
    for seed in [42, 123, 456, 789, 999, 555, 111, 333]:
        np.random.seed(seed)
        # Start with better structured initialization
        points = np.random.rand(14, 3)
        points = _energy_minimization_approach(points, iterations=200)
        ratio = _compute_min_max_ratio(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Strategy 4: Refinement using scipy optimization with multiple methods
    if best_points is not None:
        try:
            # Use L-BFGS-B with bounds [0,1] for each coordinate
            x0 = best_points.flatten()
            
            # Multiple optimization attempts for robustness
            optimization_attempts = [
                {'method': 'L-BFGS-B', 'bounds': [(0, 1) for _ in range(42)], 'options': {'maxiter': 2500, 'ftol': 1e-16, 'gtol': 1e-16}},
                {'method': 'TNC', 'bounds': [(0, 1) for _ in range(42)], 'options': {'maxiter': 2500, 'ftol': 1e-16}}, 
                {'method': 'SLSQP', 'bounds': [(0, 1) for _ in range(42)], 'options': {'maxiter': 2500, 'ftol': 1e-16, 'gtol': 1e-16}}
            ]
            
            for attempt in optimization_attempts:
                try:
                    result = minimize(
                        _objective, 
                        x0, 
                        method=attempt['method'],
                        bounds=attempt['bounds'],
                        options=attempt['options']
                    )
                    
                    if result.success:
                        refined_points = result.x.reshape(-1, 3)
                        ratio = _compute_min_max_ratio(refined_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # Strategy 5: Final adaptive constraint satisfaction refinement with more iterations
    try:
        if best_points is not None:
            best_points = _adaptive_constraint_satisfaction(best_points, max_iterations=1000)
            ratio = _compute_min_max_ratio(best_points)
            if ratio > best_ratio:
                best_ratio = ratio
    except Exception:
        pass
    
    # Strategy 6: Local search improvement for fine-tuning
    try:
        if best_points is not None:
            best_points = _local_search_improvement(best_points, max_iterations=300)
            ratio = _compute_min_max_ratio(best_points)
            if ratio > best_ratio:
                best_ratio = ratio
    except Exception:
        pass
    
    # Strategy 7: Additional high-precision optimization with even stricter tolerances
    try:
        if best_points is not None:
            # Try with very high precision
            x0 = best_points.flatten()
            result = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(42)],
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                ratio = _compute_min_max_ratio(refined_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points
    except Exception:
        pass
    
    # Strategy 8: Final constraint satisfaction refinement with even more iterations
    try:
        if best_points is not None:
            best_points = _adaptive_constraint_satisfaction(best_points, max_iterations=1000)
            ratio = _compute_min_max_ratio(best_points)
            if ratio > best_ratio:
                best_ratio = ratio
    except Exception:
        pass
    
    # Strategy 9: One final local search
    try:
        if best_points is not None:
            best_points = _local_search_improvement(best_points, max_iterations=200)
            ratio = _compute_min_max_ratio(best_points)
            if ratio > best_ratio:
                best_ratio = ratio
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
