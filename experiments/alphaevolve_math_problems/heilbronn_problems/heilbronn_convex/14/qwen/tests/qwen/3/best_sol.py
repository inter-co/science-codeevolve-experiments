# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def compute_min_triangle_area(points):
    """Vectorized computation of minimum triangle area among all combinations of 3 points."""
    if len(points) < 3:
        return 0
    
    # Vectorized approach for better performance
    points = np.array(points)
    n = len(points)
    
    # Generate all combinations of 3 points using meshgrid
    i, j, k = np.meshgrid(range(n), range(n), range(n), indexing='ij')
    
    # Filter out invalid combinations (where indices are equal)
    mask = (i < j) & (j < k)
    i_vals, j_vals, k_vals = i[mask], j[mask], k[mask]
    
    # Extract the three points for each combination
    p1 = points[i_vals]
    p2 = points[j_vals]
    p3 = points[k_vals]
    
    # Compute triangle areas using cross product formula
    # Area = 0.5 * |det([p2-p1, p3-p1])| = 0.5 * |(p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])|
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                         (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    
    return np.min(areas) if len(areas) > 0 else 0

def initialize_hexagonal_grid(n, seed=42):
    """Initialize points using a hexagonal grid pattern."""
    np.random.seed(seed)
    
    # Create hexagonal lattice with some randomness
    points = []
    rows = 4
    cols = 4
    
    for row in range(rows):
        for col in range(cols):
            if len(points) >= n:
                break
            # Hexagonal offset
            x = col + (row % 2) * 0.5
            y = row * np.sqrt(3) / 2
            
            # Add randomness to avoid degeneracy
            x += np.random.normal(0, 0.03)
            y += np.random.normal(0, 0.03)
            
            points.append([x, y])
            
        if len(points) >= n:
            break
    
    # Scale and center in unit square
    points = np.array(points[:n])
    if len(points) > 0:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0 and y_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale to fit nicely in [0.1, 0.9] range to avoid boundary issues
        points = points * 0.8 + 0.1
        
    # Ensure we have exactly n points
    while len(points) < n:
        points = np.vstack([points, np.random.rand(1, 2)])
    
    return points[:n]

def initialize_fibonacci_spiral(n, seed=100):
    """Initialize points using Fibonacci spiral for good distribution."""
    np.random.seed(seed)
    
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n):
        # Fibonacci spiral approach
        theta = i * 2.4  # Modified angle for better spread
        radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1.0
        
        x = radius * np.cos(theta) * 0.8 + 0.1
        y = radius * np.sin(theta) * 0.8 + 0.1
        
        # Add small random perturbation
        x += np.random.normal(0, 0.02)
        y += np.random.normal(0, 0.02)
        
        points.append([x, y])
    
    return np.array(points)

def initialize_structured_points(n, seed=42):
    """Initialize points using a structured configuration that works well for Heilbronn problem."""
    np.random.seed(seed)
    
    # Create a combination of center point and radial distribution
    points = []
    
    # Start with a central point
    points.append([0.5, 0.5])
    
    # Add points in a circular pattern around the center
    # This creates good separation and avoids degenerate triangles
    for i in range(n - 1):
        angle = i * 2 * np.pi / (n - 1)
        radius = 0.3
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        # Keep within reasonable bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        points.append([x, y])
    
    return np.array(points)

def adaptive_local_search(points, max_iter=70, early_stopping=True):
    """Enhanced adaptive local search with better convergence detection."""
    current_points = points.copy()
    current_min_area = compute_min_triangle_area(current_points)
    
    last_improvement_iter = 0
    
    for iteration in range(max_iter):
        improved = False
        # Adaptive step size that decreases over iterations
        step_size = max(0.001, 0.02 * (1 - iteration / max_iter))
        
        # Try moving each point with multiple attempts
        for i in range(len(current_points)):
            best_point = current_points[i].copy()
            best_area = current_min_area
            
            # Try more perturbations for better exploration
            for attempt in range(25):  # Increased attempts from 20 to 25
                # Generate random perturbation with adaptive step size
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                
                test_point = current_points[i] + np.array([dx, dy])
                # Keep within bounds
                test_point = np.clip(test_point, 0, 1)
                
                # Create new configuration
                new_points = current_points.copy()
                new_points[i] = test_point
                
                new_min_area = compute_min_triangle_area(new_points)
                if new_min_area > best_area:
                    best_area = new_min_area
                    best_point = test_point.copy()
                    improved = True
                    
            if improved:
                current_points[i] = best_point
                current_min_area = best_area
                last_improvement_iter = iteration
                
        # Early stopping if no improvement for several iterations
        if early_stopping and not improved and iteration - last_improvement_iter > 8:
            break
            
    return current_points, current_min_area

def global_optimization_with_restart(initial_points, max_evaluations=500):
    """Use differential evolution for global optimization with restarts."""
    n = len(initial_points)
    
    def objective(x_flat):
        points = x_flat.reshape(n, 2)
        points = np.clip(points, 0, 1)
        min_area = compute_min_triangle_area(points)
        return -min_area  # Negative because we want to maximize
    
    # Set up bounds
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Try multiple restarts with different seeds
    best_points = initial_points.copy()
    best_min_area = compute_min_triangle_area(best_points)
    
    # Try 9 restarts for better chance of finding global optimum
    for restart in range(9):
        try:
            # Use different random seeds for each restart
            np.random.seed(42 + restart)
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=min(100, max_evaluations // 9),
                popsize=25,  # Larger population for better exploration
                seed=42 + restart,
                disp=False,
                atol=1e-8,
                rtol=1e-8
            )
            
            optimized_points = result.x.reshape(n, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            optimized_min_area = compute_min_triangle_area(optimized_points)
            
            if optimized_min_area > best_min_area:
                best_min_area = optimized_min_area
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    return best_points, best_min_area

def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of 14 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. 

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    
    # Strategy 1: Multiple diverse initialization strategies (including structured points from INSPIRATION 1)
    initial_strategies = [
        lambda s: initialize_hexagonal_grid(n, seed=s),
        lambda s: initialize_fibonacci_spiral(n, seed=s),
        lambda s: initialize_structured_points(n, seed=s),
        lambda s: np.random.rand(n, 2)  # Random initialization as fallback
    ]
    
    # Generate multiple initial configurations with different seeds
    initial_configurations = []
    seeds_to_try = [42, 100, 200, 300, 400, 500, 600, 700]  # More seeds for diversity
    
    for seed in seeds_to_try:
        for strategy in initial_strategies:
            try:
                config = strategy(seed)
                initial_configurations.append(config)
            except:
                continue
    
    # Strategy 2: Global optimization from a limited but strategic set of initial points
    global_optimized_configs = []
    
    # Process only a few key initial configurations to stay within time limits
    for i, initial_config in enumerate(initial_configurations[:8]):  # Reduced from 10 to 8 for efficiency
        try:
            # Run global optimization with moderate evaluation count
            global_config, _ = global_optimization_with_restart(initial_config, max_evaluations=250)
            global_optimized_configs.append(global_config)
        except:
            continue
    
    # Strategy 3: Local refinement of global results with more iterations for better quality
    refined_configs = []
    for config in global_optimized_configs:
        try:
            refined_config, _ = adaptive_local_search(config, max_iter=70)
            refined_configs.append(refined_config)
        except:
            continue
    
    # Strategy 4: Final optimization pass on top results
    final_configs = []
    if refined_configs:
        # Take the top 3 configurations for final processing to balance quality and efficiency
        refined_areas = [compute_min_triangle_area(c) for c in refined_configs]
        top_indices = np.argsort(refined_areas)[-3:] if len(refined_configs) >= 3 else range(len(refined_configs))
        
        for idx in top_indices:
            try:
                final_config, _ = global_optimization_with_restart(refined_configs[idx], max_evaluations=150)
                final_configs.append(final_config)
            except:
                continue
    
    # Strategy 5: Direct optimization of structured configurations
    direct_configs = []
    for seed in [999, 888, 777, 666]:  # More direct optimizations
        try:
            # Use structured initialization for better starting points
            structured_init = initialize_structured_points(n, seed=seed)
            direct_config, _ = adaptive_local_search(structured_init, max_iter=30)
            direct_configs.append(direct_config)
        except:
            continue
    
    # Collect all candidates and select the best
    all_candidates = (
        initial_configurations + 
        global_optimized_configs + 
        refined_configs + 
        final_configs + 
        direct_configs
    )
    
    # Filter out any invalid configurations
    valid_candidates = [c for c in all_candidates if c.shape == (n, 2)]
    
    if not valid_candidates:
        # Fallback to a simple configuration
        return np.random.rand(n, 2)
    
    # Evaluate all candidates and pick the best one
    candidate_areas = [compute_min_triangle_area(p) for p in valid_candidates]
    best_idx = np.argmax(candidate_areas)
    
    return valid_candidates[best_idx]


# EVOLVE-BLOCK-END
