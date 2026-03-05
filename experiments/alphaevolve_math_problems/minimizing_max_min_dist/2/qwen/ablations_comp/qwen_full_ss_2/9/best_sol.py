# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing
import math
import warnings
warnings.filterwarnings('ignore')

def calculate_min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0
    distances = pdist(points)
    if len(distances) == 0:
        return 0
    d_min = np.min(distances)
    d_max = np.max(distances)
    if d_max == 0:
        return 0
    return d_min / d_max

def initialize_regular_16gon():
    """Create points on a regular 16-gon - highly symmetric approach."""
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Scale and center appropriately
    points = points * 0.4 + 0.5
    return points

def initialize_hexagonal():
    """Create a hexagonal grid initialization - clean and effective."""
    # Create a proper hexagonal grid that fits well in [0,1] x [0,1]
    points = []
    rows = 4
    cols = 4
    
    # Use exact spacing for better results
    spacing_x = 1.0
    spacing_y = math.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            x = j * spacing_x + (i % 2) * spacing_x / 2
            y = i * spacing_y
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Normalize to fit nicely in [0,1] x [0,1]
    if points.shape[0] > 0:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
        # Scale to fit nicely within [0.1, 0.9] to avoid edge effects
        points = points * 0.8 + 0.1
        
    return points

def initialize_golden_spiral():
    """Create points using golden spiral approach for better distribution."""
    points = []
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    
    # Generate points along a spiral pattern
    for i in range(16):
        theta = i * 2 * math.pi / phi
        r = math.sqrt(i / 15.0) * 0.4  # Scale to fit nicely
        x = 0.5 + r * math.cos(theta)
        y = 0.5 + r * math.sin(theta)
        points.append([x, y])
    
    return np.array(points)

def initialize_random_perturbed():
    """Create random points with slight perturbations from a regular grid."""
    # Start with a regular 4x4 grid
    points = []
    for i in range(4):
        for j in range(4):
            points.append([j/3.0, i/3.0])
    
    points = np.array(points[:16])
    
    # Add small random perturbations
    np.random.seed(42)
    points += np.random.normal(0, 0.03, points.shape)
    
    # Clip to bounds
    points = np.clip(points, 0, 1)
    return points

def objective(params):
    """Objective function for optimization - returns negative ratio to maximize."""
    # Reshape parameters back to points
    pts = params.reshape(-1, 2)
    
    # Ensure points stay within bounds [0,1] x [0,1]
    pts = np.clip(pts, 0, 1)
    
    # Calculate all pairwise distances
    distances = pdist(pts)
    
    if len(distances) == 0:
        return 0
    
    # Calculate min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max <= 0:
        return 0
        
    # Return negative ratio to maximize (since we're minimizing)
    return -d_min / d_max

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a multi-start approach with diverse initial configurations and robust optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    # Strategy: Multi-start with diverse initial configurations
    initial_strategies = [
        ("Regular 16-gon", initialize_regular_16gon),
        ("Hexagonal grid", initialize_hexagonal),
        ("Golden spiral", initialize_golden_spiral),
        ("Random perturbed", initialize_random_perturbed)
    ]
    
    best_points = None
    best_ratio = 0
    
    # Try multiple initial configurations
    for name, init_func in initial_strategies:
        try:
            points = init_func()
            
            # Use dual annealing for robust global optimization
            result = dual_annealing(
                objective,
                bounds=[(0, 1) for _ in range(n * 2)],
                maxiter=600,  # Reduced to stay within time budget
                initial_temp=500,
                seed=42 + hash(name) % 1000,
                no_local_search=True
            )
            
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = -objective(optimized_points.flatten())
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # If we still don't have a good solution, use a fallback with more iterations
    if best_points is None or best_ratio < 0.05:
        # Try with hexagonal initialization and more iterations
        points = initialize_hexagonal()
        
        result = dual_annealing(
            objective,
            bounds=[(0, 1) for _ in range(n * 2)],
            maxiter=800,
            initial_temp=300,
            seed=42,
            no_local_search=True
        )
        
        optimized_points = result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
        ratio = -objective(optimized_points.flatten())
        
        if ratio > best_ratio:
            best_points = optimized_points
            best_ratio = ratio
    
    # Final refinement with local optimization if we have a decent solution
    if best_points is not None and best_ratio > 0.05:
        try:
            from scipy.optimize import minimize
            # Use L-BFGS-B for local refinement
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_ratio = -objective(final_points.flatten())
                
                if final_ratio > best_ratio:
                    best_points = final_points
                    
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
