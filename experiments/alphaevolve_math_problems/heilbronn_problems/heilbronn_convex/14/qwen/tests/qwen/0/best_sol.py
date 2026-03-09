# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def triangle_area(p1, p2, p3):
    """Calculate the area of triangle formed by three points using cross product."""
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def min_triangle_area(points):
    """Calculate the minimum triangle area among all combinations of three points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Use vectorized approach for better performance
    min_area = float('inf')
    
    # Check all combinations of 3 points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = triangle_area(points[i], points[j], points[k])
                min_area = min(min_area, area)
    
    return min_area if min_area != float('inf') else 0

def fibonacci_spiral_distribution(n):
    """Generate points using Fibonacci spiral distribution on a disk, then project to unit square"""
    points = []
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Generate points on a disk using Fibonacci spiral
    for i in range(n):
        # Distribute points radially and angularly
        r = np.sqrt(i / (n - 1)) if n > 1 else 0
        theta = i * (2 - 2 * phi) * np.pi  # Modified angle for better distribution
        
        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Map to unit square [-1,1] to [0,1] 
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        points.append([x, y])
    
    return np.array(points)

def hexagonal_grid_initialization(n, seed=42):
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

def regular_grid_initialization(n, seed=42):
    """Initialize points on a regular grid pattern."""
    np.random.seed(seed)
    # Create a grid pattern
    side = int(np.ceil(np.sqrt(n)))
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = i / (side - 1) if side > 1 else 0.5
            y = j / (side - 1) if side > 1 else 0.5
            # Add slight randomness
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            points.append([x, y])
        if len(points) >= n:
            break
    
    # Scale to [0.1, 0.9] range
    points = np.array(points[:n])
    points = np.clip(points, 0.1, 0.9)
    return points

def random_initialization(n, seed=42):
    """Generate completely random initialization."""
    np.random.seed(seed)
    return np.random.rand(n, 2)

def structured_initialization(n, seed=200):
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

def adaptive_local_search(points, max_iter=50):
    """Enhanced local search with better convergence detection and more thorough exploration."""
    current_points = points.copy()
    current_min_area = min_triangle_area(current_points)
    
    last_improvement_iter = 0
    
    for iteration in range(max_iter):
        improved = False
        # Adaptive step size that decreases over iterations
        step_size = max(0.001, 0.03 * (1 - iteration / max_iter))
        
        # Try moving each point with multiple attempts
        for i in range(len(current_points)):
            best_point = current_points[i].copy()
            best_area = current_min_area
            
            # Try more perturbations for better exploration
            for attempt in range(20):
                # Generate random perturbation with adaptive step size
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                
                test_point = current_points[i] + np.array([dx, dy])
                # Keep within bounds
                test_point = np.clip(test_point, 0, 1)
                
                # Create new configuration
                new_points = current_points.copy()
                new_points[i] = test_point
                
                new_min_area = min_triangle_area(new_points)
                if new_min_area > best_area:
                    best_area = new_min_area
                    best_point = test_point.copy()
                    improved = True
                    
            if improved:
                current_points[i] = best_point
                current_min_area = best_area
                last_improvement_iter = iteration
                
        # Early stopping if no improvement for several iterations
        if not improved and iteration - last_improvement_iter > 10:
            break
            
    return current_points, current_min_area

def global_optimization_with_restart(initial_points, max_evaluations=300):
    """Use differential evolution for global optimization with restarts."""
    n = len(initial_points)
    
    def objective(x_flat):
        points = x_flat.reshape(n, 2)
        points = np.clip(points, 0, 1)
        min_area = min_triangle_area(points)
        return -min_area  # Negative because we want to maximize
    
    # Set up bounds
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Try multiple restarts with different seeds
    best_points = initial_points.copy()
    best_min_area = min_triangle_area(best_points)
    
    # Try 5 restarts for better chance of finding global optimum
    for restart in range(5):
        try:
            # Use different random seeds for each restart
            np.random.seed(42 + restart)
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=min(50, max_evaluations // 5),
                popsize=15,  # Moderate population size for exploration
                seed=42 + restart,
                disp=False,
                atol=1e-8,
                rtol=1e-8
            )
            
            optimized_points = result.x.reshape(n, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            optimized_min_area = min_triangle_area(optimized_points)
            
            if optimized_min_area > best_min_area:
                best_min_area = optimized_min_area
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    return best_points, best_min_area

def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of 14 points on or inside a convex region to maximize 
    the area of the smallest triangle formed by these points.
    
    Uses a hybrid approach combining multiple initialization strategies,
    global optimization with differential evolution, and local refinement.
    
    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    np.random.seed(42)
    n = 14
    
    # Strategy 1: Multiple diverse initializations to get good starting points
    initial_strategies = []
    
    # Hexagonal grid with different seeds
    for seed in [42, 100, 200]:
        initial_strategies.append(hexagonal_grid_initialization(n, seed=seed))
    
    # Fibonacci spiral initialization
    initial_strategies.append(fibonacci_spiral_distribution(n))
    
    # Structured initialization (inspired by successful approaches)
    initial_strategies.append(structured_initialization(n, seed=200))
    
    # Random initialization
    initial_strategies.append(random_initialization(n, seed=42))
    
    # Regular grid initialization
    initial_strategies.append(regular_grid_initialization(n, seed=42))
    
    # Strategy 2: Global optimization on each initialization with reduced computational cost
    global_results = []
    for i, initial_points in enumerate(initial_strategies):
        try:
            global_points, _ = global_optimization_with_restart(initial_points, max_evaluations=200)
            global_results.append(global_points)
        except Exception:
            # If optimization fails, keep the initial points
            global_results.append(initial_points)
    
    # Strategy 3: Local refinement of all global results
    refined_results = []
    for result in global_results:
        try:
            refined_points, _ = adaptive_local_search(result, max_iter=30)
            refined_results.append(refined_points)
        except Exception:
            refined_results.append(result)
    
    # Strategy 4: Final refinement with even more aggressive local search
    final_results = []
    for refined_result in refined_results:
        try:
            final_points, _ = adaptive_local_search(refined_result, max_iter=25)
            final_results.append(final_points)
        except Exception:
            final_results.append(refined_result)
    
    # Strategy 5: Additional optimization on the very best candidates
    # Select top 3 candidates by area and optimize them further
    all_candidates = (
        initial_strategies + 
        global_results + 
        refined_results + 
        final_results
    )
    
    candidate_areas = [min_triangle_area(p) for p in all_candidates]
    sorted_indices = np.argsort(candidate_areas)[::-1][:3]  # Top 3 candidates
    
    extra_optimization_results = []
    for idx in sorted_indices:
        try:
            extra_points, _ = global_optimization_with_restart(all_candidates[idx], max_evaluations=150)
            extra_optimization_results.append(extra_points)
        except Exception:
            extra_optimization_results.append(all_candidates[idx])
    
    # Collect all candidates and select the best
    all_candidates = (
        initial_strategies + 
        global_results + 
        refined_results + 
        final_results +
        extra_optimization_results
    )
    
    # Filter out any invalid configurations
    valid_candidates = [c for c in all_candidates if c.shape == (n, 2)]
    
    if not valid_candidates:
        # Fallback to a simple configuration
        return random_initialization(n, seed=42)
    
    # Evaluate all candidates and pick the best one
    candidate_areas = [min_triangle_area(p) for p in valid_candidates]
    best_idx = np.argmax(candidate_areas)
    
    return valid_candidates[best_idx]


# EVOLVE-BLOCK-END
