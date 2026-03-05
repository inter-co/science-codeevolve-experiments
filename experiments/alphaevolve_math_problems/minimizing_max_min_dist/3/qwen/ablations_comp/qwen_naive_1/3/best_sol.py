# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from numba import jit
import math
from scipy.spatial import ConvexHull


@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """Compute min/max distance ratio using numba for speed"""
    n = points.shape[0]
    min_dist_sq = np.inf
    max_dist_sq = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
    
    if max_dist_sq == 0:
        return 0.0
    return np.sqrt(min_dist_sq) / np.sqrt(max_dist_sq)


def generate_spherical_initial_config():
    """Generate initial configuration using spherical geometry and symmetry breaking"""
    np.random.seed(42)
    
    # Create a known good starting configuration based on geometric principles
    # Using vertices of a truncated octahedron-like structure for 14 points
    # This provides a good starting point with reasonable distribution
    
    # Base structure: 8 vertices of cube + 6 face centers + 2 opposite points
    # But since we have 14 points, let's use a more sophisticated approach
    
    # Generate points using spherical coordinates with careful spacing
    points = []
    
    # Use fibonacci-based spherical distribution with slight perturbations
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Generate points on unit sphere
    for i in range(14):
        # Distribute points more evenly using Fibonacci-like approach
        y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        # Add some variation to avoid perfect symmetry
        if i == 0:
            theta = 0
        else:
            theta = (i * 2 * np.pi) / (14 - 1) + np.random.uniform(-0.1, 0.1)
        
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / np.max(norms) * 0.9  # Scale down slightly
    
    # Add controlled randomness to break symmetries
    noise_magnitude = 0.08
    points += np.random.normal(0, noise_magnitude, points.shape)
    
    # Project back to sphere to maintain spherical constraint
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis] * 0.9
    
    # Convert to [0,1]^3 by normalizing to bounding box
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    points = (points - mins) / (maxs - mins + 1e-10)
    points = np.clip(points, 0, 1)
    
    return points


def compute_energy_gradient(points, alpha=1.0):
    """
    Compute gradient for energy-based optimization where energy is proportional
    to inverse of distances raised to power alpha.
    """
    n = points.shape[0]
    grad = np.zeros_like(points)
    
    # Compute pairwise distances and their gradients
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            
            if dist_sq > 1e-12:  # Avoid division by zero
                inv_dist_alpha = 1.0 / (dist_sq**(alpha/2))
                force = alpha * inv_dist_alpha * diff / dist_sq
                
                grad[i] -= force
                grad[j] += force
    
    return grad


def energy_minimization_approach():
    """
    Alternative approach: Energy minimization with repulsive forces
    This explores a completely different paradigm from evolutionary optimization
    """
    np.random.seed(42)
    
    # Start with a good initial configuration
    points = generate_spherical_initial_config()
    
    # Energy minimization parameters
    learning_rate = 0.01
    max_iterations = 1000
    tolerance = 1e-6
    
    prev_energy = float('inf')
    
    for iteration in range(max_iterations):
        # Compute current energy and gradient
        grad = compute_energy_gradient(points, alpha=2.0)
        
        # Update points using gradient descent
        points = points - learning_rate * grad
        
        # Keep points within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Check for convergence
        if iteration % 100 == 0:
            current_ratio = compute_min_max_ratio_jit(points)
            if abs(prev_energy - current_ratio) < tolerance:
                break
            prev_energy = current_ratio
    
    return points


def constrained_optimization_approach():
    """
    Another approach: Use constrained optimization with geometric constraints
    """
    # Generate initial configuration
    initial_points = generate_spherical_initial_config()
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Define bounds for [0,1]^3
    bounds = [(0, 1) for _ in range(42)]
    
    # Define objective function
    def objective(x):
        points = x.reshape(-1, 3)
        try:
            ratio = compute_min_max_ratio_jit(points)
            return -ratio  # Negative because we want to maximize
        except:
            return -1e-10
    
    # Use SLSQP for constrained optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
    )
    
    final_points = result.x.reshape(-1, 3)
    final_points = np.clip(final_points, 0, 1)
    
    return final_points


def hybrid_geometric_approach():
    """
    Hybrid approach combining multiple geometric insights:
    1. Start with spherical distribution
    2. Apply iterative improvement using local search
    3. Use convex hull properties for boundary handling
    """
    np.random.seed(42)
    
    # Generate initial configuration
    points = generate_spherical_initial_config()
    
    # Iterative improvement loop
    best_ratio = 0
    best_points = points.copy()
    
    for iteration in range(50):  # Limited iterations to stay within time budget
        # Local optimization step
        current_ratio = compute_min_max_ratio_jit(points)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
        
        # Simple local search: perturb each point slightly and accept improvements
        for i in range(14):
            # Save current point
            original_point = points[i].copy()
            
            # Try small perturbations
            for _ in range(10):
                # Small random perturbation
                delta = np.random.normal(0, 0.005, 3)
                points[i] += delta
                
                # Keep within bounds
                points[i] = np.clip(points[i], 0, 1)
                
                # Check if this improves the ratio
                new_ratio = compute_min_max_ratio_jit(points)
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                else:
                    # Revert if no improvement
                    points[i] = original_point.copy()
    
    return best_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid geometric approach combining energy minimization and local search.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Try multiple approaches and pick the best
    approaches = [
        energy_minimization_approach,
        constrained_optimization_approach,
        hybrid_geometric_approach
    ]
    
    best_ratio = 0
    best_points = None
    
    for approach in approaches:
        try:
            points = approach()
            ratio = compute_min_max_ratio_jit(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception as e:
            continue
    
    # Final refinement with local optimization if needed
    if best_points is not None:
        # Use simple local search for final improvement
        current_points = best_points.copy()
        current_ratio = compute_min_max_ratio_jit(current_points)
        
        for _ in range(100):  # Limited iterations
            improved = False
            for i in range(14):
                original_point = current_points[i].copy()
                # Try small perturbations
                for _ in range(5):
                    delta = np.random.normal(0, 0.001, 3)
                    current_points[i] += delta
                    current_points[i] = np.clip(current_points[i], 0, 1)
                    
                    new_ratio = compute_min_max_ratio_jit(current_points)
                    if new_ratio > current_ratio:
                        current_ratio = new_ratio
                        improved = True
                    else:
                        current_points[i] = original_point.copy()
            
            if not improved:
                break
        
        return current_points
    
    # Fallback to default approach
    return generate_spherical_initial_config()


# EVOLVE-BLOCK-END
