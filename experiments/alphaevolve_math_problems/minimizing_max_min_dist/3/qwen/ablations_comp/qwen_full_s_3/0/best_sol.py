# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import random


def compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute the minimum distance to maximum distance ratio for a set of points."""
    if len(points) < 2:
        return 0
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return min_dist / max_dist if max_dist > 0 else 0


def construct_icosahedral_based_points() -> np.ndarray:
    """Construct points based on icosahedral symmetry for good distribution."""
    # Vertices of regular icosahedron scaled to unit sphere
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    
    # Create vertices of icosahedron
    vertices = []
    
    # Add vertices at positions (±1, ±φ, 0), (0, ±1, ±φ), (±φ, 0, ±1)
    for i in [1, -1]:
        for j in [1, -1]:
            vertices.append([i, j * phi, 0])
            vertices.append([0, i, j * phi])
            vertices.append([i * phi, 0, j])
    
    # Normalize to unit sphere
    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / np.maximum(norms, 1e-10)
    
    # Use 12 vertices plus 2 strategic additional points for 14 total
    # Select best 12 vertices (they're already good)
    selected_vertices = vertices[:12]
    
    # Add 2 more points for better distribution - use poles and some diagonals
    additional_points = np.array([
        [0, 0, 1],   # North pole
        [0, 0, -1]   # South pole
    ])
    
    # Combine and normalize
    all_points = np.vstack([selected_vertices, additional_points])
    norms = np.linalg.norm(all_points, axis=1, keepdims=True)
    all_points = all_points / np.maximum(norms, 1e-10)
    
    return all_points


def construct_fibonacci_points(n: int) -> np.ndarray:
    """Generate Fibonacci spiral points on sphere."""
    points = []
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        theta = math.acos(z)
        phi = math.sqrt(n * math.pi) * theta
        
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        
        points.append([x, y, z])
    
    return np.array(points)


def optimize_with_gradient_methods(initial_points: np.ndarray) -> np.ndarray:
    """Optimize point distribution using gradient-based methods with proper constraints."""
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 3)
        
        # Keep points within unit sphere bounds
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero and normalize
        safe_norms = np.where(norms > 0, norms, 1.0)
        points = points / safe_norms
        
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return float('inf')
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio to convert maximization to minimization
        if max_dist <= 0:
            return float('inf')
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint to keep points on unit sphere"""
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should equal zero for unit sphere
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Define constraints properly as equality constraint
    cons = {'type': 'eq', 'fun': constraint_func}
    
    # Optimization bounds (keep within reasonable limits)
    n = len(initial_points)
    bounds = [(-1.0, 1.0) for _ in range(3*n)]
    
    # Try multiple optimization attempts with different methods
    methods = ['SLSQP', 'trust-constr', 'L-BFGS-B']
    best_points = initial_points.copy()
    best_ratio = compute_min_max_ratio(initial_points)
    
    for method in methods:
        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                # Extract and normalize final points
                final_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(final_points, axis=1, keepdims=True)
                final_points = final_points / np.maximum(norms, 1e-10)
                
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception:
            continue
    
    return best_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and gradient-based optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Multiple initialization strategies based on proven good configurations
    strategies = []
    
    # Strategy 1: Icosahedral-based construction (known good for 12 points + extras)
    ico_points = construct_icosahedral_based_points()
    strategies.append(("icosahedral", ico_points))
    
    # Strategy 2: Fibonacci spiral points
    fib_points = construct_fibonacci_points(14)
    strategies.append(("fibonacci", fib_points))
    
    # Strategy 3: Random points on sphere with seed for reproducibility
    np.random.seed(42)
    random_points = np.random.rand(14, 3) * 2 - 1  # Range [-1, 1]
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    random_points = random_points / np.maximum(norms, 1e-10)  # Avoid zero norm
    strategies.append(("random", random_points))
    
    # Strategy 4: Perturbed version of Fibonacci
    np.random.seed(123)
    fib_perturbed = fib_points.copy()
    perturbation = np.random.normal(0, 0.05, fib_perturbed.shape)
    fib_perturbed = fib_perturbed + perturbation
    norms = np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
    fib_perturbed = fib_perturbed / np.maximum(norms, 1e-10)
    strategies.append(("fibonacci_perturbed", fib_perturbed))
    
    # Strategy 5: Another variation with better spread
    np.random.seed(999)
    # Start with icosahedral points and add some random noise
    ico_noisy = ico_points.copy()
    noise = np.random.normal(0, 0.03, ico_noisy.shape)
    ico_noisy = ico_noisy + noise
    norms = np.linalg.norm(ico_noisy, axis=1, keepdims=True)
    ico_noisy = ico_noisy / np.maximum(norms, 1e-10)
    strategies.append(("icosahedral_noisy", ico_noisy))
    
    # Optimization parameters
    n = 14
    best_ratio = float('-inf')
    best_points = None
    
    # Try each strategy with optimization
    for name, initial_points in strategies:
        # Optimize the initial points
        optimized_points = optimize_with_gradient_methods(initial_points)
        
        # Evaluate and keep track of best result
        ratio = compute_min_max_ratio(optimized_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points
    
    # Final refinement using the best configuration
    if best_points is not None:
        # Try one more round of optimization on the best found solution
        refined_points = optimize_with_gradient_methods(best_points)
        final_ratio = compute_min_max_ratio(refined_points)
        if final_ratio > best_ratio:
            best_points = refined_points
    
    # If still no good solution, return the best we found
    if best_points is not None:
        return best_points
    
    # Fallback to Fibonacci points
    return strategies[1][1]


# EVOLVE-BLOCK-END
