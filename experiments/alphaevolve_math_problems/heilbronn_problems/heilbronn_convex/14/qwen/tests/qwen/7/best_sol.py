# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

def triangle_area(p1, p2, p3):
    """Calculate the area of triangle formed by three points."""
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))


def min_triangle_area(points):
    """Calculate the minimum triangle area among all combinations of three points."""
    n = len(points)
    if n < 3:
        return 0
    
    min_area = float('inf')
    
    # Check all combinations of 3 points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = triangle_area(points[i], points[j], points[k])
                min_area = min(min_area, area)
    
    return min_area


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
    """Initialize points using a hexagonal grid pattern with some randomness."""
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


def geometric_construction_initialization(n):
    """Create initial points using geometric construction principles."""
    # Start with a regular polygon and add interior points strategically
    points = []
    
    # Place points around the perimeter of a circle (regular hexagon approximation)
    angles = np.linspace(0, 2*np.pi, min(6, n), endpoint=False)
    for angle in angles:
        points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
    
    # Add interior points in a structured way
    if n > 6:
        # Add central point
        if n >= 7:
            points.append([0.5, 0.5])
        
        # Add additional points in concentric rings
        ring_positions = [
            [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
            [0.2, 0.5], [0.8, 0.5], [0.5, 0.2], [0.5, 0.8]
        ]
        
        for i, pos in enumerate(ring_positions):
            if len(points) < n:
                points.append(pos)
    
    # Fill remaining points with random distribution
    while len(points) < n:
        points.append([np.random.random(), np.random.random()])
    
    return np.array(points[:n])


def energy_based_initialization(n, seed=42):
    """Initialize points using an energy-based approach inspired by physics simulations."""
    np.random.seed(seed)
    
    # Start with random points
    points = np.random.rand(n, 2)
    
    # Apply simple repulsion to spread points initially
    for _ in range(100):  # Few iterations of repulsion
        for i in range(n):
            force = np.zeros(2)
            for j in range(n):
                if i != j:
                    diff = points[i] - points[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq > 1e-10:
                        force += diff / (dist_sq ** 1.5)
            points[i] += 0.001 * force
    
    # Keep within bounds
    points = np.clip(points, 0, 1)
    
    return points


def adaptive_local_search(points, max_iter=70, early_stopping=True):
    """Enhanced adaptive local search with better convergence detection."""
    current_points = points.copy()
    current_min_area = min_triangle_area(current_points)
    
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
            for attempt in range(25):
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
        if early_stopping and not improved and iteration - last_improvement_iter > 8:
            break
            
    return current_points


def global_optimization_with_restart(initial_points, max_evaluations=500):
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
    
    # Try 7 restarts for better chance of finding global optimum
    for restart in range(7):
        try:
            # Use different random seeds for each restart
            np.random.seed(42 + restart)
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=min(100, max_evaluations // 7),
                popsize=25,  # Larger population for better exploration
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
                
        except Exception as e:
            continue
    
    return best_points


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of 14 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    np.random.seed(42)
    n = 14
    
    # Multi-strategy approach combining best initialization strategies from inspirations
    initial_strategies = []
    
    # Strategy 1: Multiple diverse initializations (combining best approaches from inspirations)
    
    # Fibonacci spiral (highly effective approach from inspiration 1)
    initial_strategies.append(fibonacci_spiral_distribution(n))
    
    # Hexagonal grid (structured approach from inspiration 3)
    initial_strategies.append(hexagonal_grid_initialization(n, seed=42))
    
    # Geometric construction (inspired by inspiration 2)
    initial_strategies.append(geometric_construction_initialization(n))
    
    # Energy-based initialization (inspired by inspiration 2)
    initial_strategies.append(energy_based_initialization(n, seed=42))
    
    # Random initialization as fallback
    initial_strategies.append(np.random.rand(n, 2))
    
    # Strategy 2: Global optimization on each initialization with more thorough approach
    optimized_results = []
    
    for i, initial_points in enumerate(initial_strategies):
        try:
            # Apply global optimization with multiple restarts
            optimized_points = global_optimization_with_restart(initial_points, max_evaluations=250)
            optimized_results.append(optimized_points)
        except Exception as e:
            optimized_results.append(initial_points)
    
    # Strategy 3: Enhanced local refinement of all optimized results
    refined_results = []
    for result in optimized_results:
        try:
            refined_points = adaptive_local_search(result, max_iter=70)
            refined_results.append(refined_points)
        except Exception as e:
            refined_results.append(result)
    
    # Strategy 4: Additional global optimization on refined results
    final_candidates = []
    for refined_candidate in refined_results:
        try:
            final_points = global_optimization_with_restart(refined_candidate, max_evaluations=150)
            final_candidates.append(final_points)
        except Exception as e:
            final_candidates.append(refined_candidate)
    
    # Strategy 5: Even more aggressive local search on final results
    most_refined_candidates = []
    for final_candidate in final_candidates:
        try:
            most_refined_points = adaptive_local_search(final_candidate, max_iter=50)
            most_refined_candidates.append(most_refined_points)
        except Exception as e:
            most_refined_candidates.append(final_candidate)
    
    # Strategy 6: Direct optimization of a few good configurations
    direct_configs = []
    for seed in [999, 888, 777, 666]:
        try:
            # Use structured initialization for better starting points
            structured_init = geometric_construction_initialization(n, seed=seed)
            direct_config = adaptive_local_search(structured_init, max_iter=30)
            direct_configs.append(direct_config)
        except Exception as e:
            continue
    
    # Collect all candidates and select the best one
    all_candidates = (
        initial_strategies + 
        optimized_results + 
        refined_results + 
        final_candidates + 
        most_refined_candidates + 
        direct_configs
    )
    
    # Filter out any invalid configurations
    valid_candidates = [c for c in all_candidates if c.shape == (n, 2)]
    
    if not valid_candidates:
        # Fallback to a simple configuration
        return np.random.rand(n, 2)
    
    # Evaluate all candidates and pick the best one
    candidate_areas = [min_triangle_area(p) for p in valid_candidates]
    best_idx = np.argmax(candidate_areas)
    
    # Final comprehensive optimization on the best candidate
    best_points = valid_candidates[best_idx]
    
    # Try dual annealing for final polishing (as in inspiration 1)
    try:
        bounds = [(0, 1) for _ in range(2 * n)]
        def objective(x_flat):
            points = x_flat.reshape(n, 2)
            points = np.clip(points, 0, 1)
            min_area = min_triangle_area(points)
            return -min_area  # Negative because we want to maximize
        
        result_da = dual_annealing(
            objective,
            bounds,
            maxiter=30,
            initial_temp=800,
            seed=42,
            disp=False
        )
        
        if result_da.success:
            da_points = result_da.x.reshape(n, 2)
            da_points = np.clip(da_points, 0, 1)
            da_min_area = min_triangle_area(da_points)
            
            # If dual annealing improved the solution, use it
            if da_min_area > min_triangle_area(best_points):
                best_points = da_points
    except:
        pass
    
    # Final local search refinement
    final_points = adaptive_local_search(best_points, max_iter=30)
    
    return final_points


# EVOLVE-BLOCK-END
