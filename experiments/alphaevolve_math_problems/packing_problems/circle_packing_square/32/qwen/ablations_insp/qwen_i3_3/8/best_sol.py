# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from scipy.optimize import differential_evolution
import time

# Global constants
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

@jit(nopython=True)
def validate_circles_numba(circles: np.ndarray) -> bool:
    """Fast validation using numba for better performance"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
    
    # Check non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_distance_sq = (r1 + r2)**2
            if distance_sq < min_distance_sq:
                return False
    
    return True

def validate_circles(circles: np.ndarray) -> bool:
    """Validate that all circles are within bounds and non-overlapping"""
    return validate_circles_numba(circles)

def compute_objective(circles: np.ndarray) -> float:
    """Compute the sum of radii"""
    return np.sum(circles[:, 2])

def initialize_voronoi_better(n: int) -> np.ndarray:
    """Initialize circles using enhanced Voronoi approach"""
    # Create a more sophisticated Voronoi-based initialization
    circles = np.zeros((n, 3))
    
    # Use a systematic grid with more randomness to avoid regular patterns
    grid_rows = int(np.ceil(np.sqrt(n)))
    grid_cols = int(np.ceil(n / grid_rows))
    
    # Create points with more sophisticated distribution
    points = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            if len(points) < n:
                # Add jitter with better distribution control
                x = (j + 0.5 + np.random.normal(0, 0.15)) * (1.0 / grid_cols)
                y = (i + 0.5 + np.random.normal(0, 0.15)) * (1.0 / grid_rows)
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
    
    points = np.array(points[:n])
    
    try:
        vor = Voronoi(points)
        
        # Enhanced Voronoi approach with better radius calculation
        for i in range(n):
            if i < len(vor.points):
                point = vor.points[i]
                
                # Calculate maximum possible radius more carefully
                min_distance = float('inf')
                
                # Distance to boundaries
                min_distance = min(min_distance, point[0])  # left
                min_distance = min(min_distance, 1 - point[0])  # right
                min_distance = min(min_distance, point[1])  # bottom
                min_distance = min(min_distance, 1 - point[1])  # top
                
                # Distance to nearest neighbor points (more careful)
                if len(vor.points) > 1:
                    distances = np.linalg.norm(vor.points - point, axis=1)
                    distances[distances == 0] = float('inf')  # Ignore self-distance
                    min_neighbor_dist = np.min(distances)
                    min_distance = min(min_distance, min_neighbor_dist / 2)
                
                # Set radius with better safety margin
                radius = min(0.15, min_distance * 0.65)
                radius = max(0.01, radius)
                
                circles[i] = [point[0], point[1], radius]
    except:
        # Fallback to more robust initialization
        for i in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            radius = np.random.uniform(0.02, 0.12)
            circles[i] = [x, y, radius]
    
    return circles

def initialize_spiral_placement(n: int) -> np.ndarray:
    """Initialize circles using spiral pattern for even better coverage"""
    circles = np.zeros((n, 3))
    
    # Spiral pattern with better radial distribution
    angle_step = 2 * np.pi / 10  # More frequent angular sampling
    radius_factor = 0.8
    
    idx = 0
    for layer in range(1, 6):  # 5 layers
        if idx >= n:
            break
        radius = 0.1 + (layer - 1) * 0.15
        num_points_in_layer = min(10, n - idx)
        for i in range(num_points_in_layer):
            if idx >= n:
                break
            angle = i * angle_step + layer * 0.1  # Add some phase variation
            x = 0.5 + radius * np.cos(angle) * radius_factor
            y = 0.5 + radius * np.sin(angle) * radius_factor
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Radius decreases with distance from center
            r = min(0.1, max(0.02, 0.08 - (layer - 1) * 0.01))
            circles[idx] = [x, y, r]
            idx += 1
    
    # Fill remaining positions with random valid placements
    for i in range(idx, n):
        while True:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            # Check if circle fits in square
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                circles[i] = [x, y, r]
                break
                
    return circles

def initialize_grid_with_refinement(n: int) -> np.ndarray:
    """Initialize with grid and then refine using local search"""
    circles = np.zeros((n, 3))
    
    # Start with a regular grid pattern
    rows = cols = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Slightly randomized for better distribution
            x += np.random.uniform(-spacing_x*0.1, spacing_x*0.1)
            y += np.random.uniform(-spacing_y*0.1, spacing_y*0.1)
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = min(spacing_x, spacing_y) * 0.35
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with random valid placements
    for i in range(idx, n):
        while True:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            # Check if circle fits in square
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                circles[i] = [x, y, r]
                break
    
    return circles

def enhanced_repulsive_forces(circles: np.ndarray, iterations: int = 1500, learning_rate: float = 0.01) -> np.ndarray:
    """Enhanced physics-based repulsive forces with better convergence"""
    circles = circles.copy()
    n = len(circles)
    
    for iter_num in range(iterations):
        forces = np.zeros((n, 2))
        
        # Calculate repulsive forces between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Apply force when circles are overlapping or very close
                if distance > 0 and distance < (r1 + r2) * 1.1:  # Allow some tolerance
                    force_magnitude = (r1 + r2 - distance) / (distance + 1e-8)
                    # Apply force with stronger magnitude near overlap
                    force_magnitude *= 1.0 + 0.5 * (1.0 - distance/(r1 + r2))
                    forces[i, 0] += force_magnitude * dx / distance
                    forces[i, 1] += force_magnitude * dy / distance
                    forces[j, 0] -= force_magnitude * dx / distance
                    forces[j, 1] -= force_magnitude * dy / distance
        
        # Apply forces with boundary constraints
        for i in range(n):
            # Move circle
            new_x = circles[i, 0] + learning_rate * forces[i, 0]
            new_y = circles[i, 1] + learning_rate * forces[i, 1]
            
            # Boundary constraints with soft clipping
            safe_margin = circles[i, 2] * 1.1
            new_x = np.clip(new_x, safe_margin, 1 - safe_margin)
            new_y = np.clip(new_y, safe_margin, 1 - safe_margin)
            
            circles[i, 0] = new_x
            circles[i, 1] = new_y
            
        # Occasionally try to increase radii - more aggressive approach
        if iter_num % 50 == 0:
            for i in range(n):
                x, y, r = circles[i]
                # Try to increase radius if possible
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check distance to other circles more carefully
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        max_radius = min(max_radius, distance - r2 * 0.95)  # Slight buffer
                
                # Only increase if beneficial
                if max_radius > r * 1.05 and max_radius > 0.001:
                    circles[i, 2] = min(max_radius, r * 1.15)  # Moderate increase
    
    return circles

def optimize_with_improved_constraints(initial_circles: np.ndarray, max_iter: int = 1500) -> np.ndarray:
    """Improved optimization with better constraints and robustness"""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds with tighter ranges
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraint functions with better numerical stability
    def containment_constraints(x_array):
        """Ensure all circles are within unit square"""
        constraints = []
        for i in range(n):
            idx = i * 3
            x, y, r = x_array[idx], x_array[idx+1], x_array[idx+2]
            # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1 - x >= r
                y - r,           # y >= r
                1 - y - r        # 1 - y >= r
            ])
        return np.array(constraints)
    
    def overlap_constraints(x_array):
        """Ensure no overlaps between circles"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                idx_i = i * 3
                idx_j = j * 3
                x1, y1, r1 = x_array[idx_i], x_array[idx_i+1], x_array[idx_i+2]
                x2, y2, r2 = x_array[idx_j], x_array[idx_j+1], x_array[idx_j+2]
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                min_distance = r1 + r2
                constraints.append(distance - min_distance)  # Should be >= 0
        return np.array(constraints)
    
    def objective(x_array):
        """Minimize negative sum of radii (maximize sum of radii)"""
        total_radius = 0
        for i in range(n):
            total_radius += x_array[i * 3 + 2]  # Add radius (third component)
        return -total_radius  # Negative because we minimize
    
    # Try multiple optimization methods with better fallbacks
    methods = ['SLSQP', 'trust-constr', 'L-BFGS-B']
    
    for method in methods:
        try:
            result = minimize(
                objective,
                initial_flat,
                method=method,
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
                    {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
                ],
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                final_circles = []
                for i in range(n):
                    idx = i * 3
                    x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                    final_circles.append([x, y, r])
                final_array = np.array(final_circles)
                if validate_circles(final_array):
                    return final_array
                    
        except Exception as e:
            continue
    
    # If optimization fails, return original
    return initial_circles

def smart_placement_strategy() -> np.ndarray:
    """Use a multi-strategy approach with better evaluation"""
    
    # Strategy 1: Enhanced Voronoi approach (best for coverage)
    voronoi_config = initialize_voronoi_better(N_CIRCLES)
    
    # Strategy 2: Spiral placement (good for even distribution)
    spiral_config = initialize_spiral_placement(N_CIRCLES)
    
    # Strategy 3: Grid with refinement (good baseline)
    grid_config = initialize_grid_with_refinement(N_CIRCLES)
    
    # Strategy 4: Physics-based (robust but may need refinement)
    physics_config = initialize_grid_with_refinement(N_CIRCLES)  # Simplified version
    
    # Evaluate all strategies
    configs = [
        ("voronoi", voronoi_config),
        ("spiral", spiral_config),
        ("grid", grid_config),
        ("physics", physics_config)
    ]
    
    best_config = None
    best_sum = -1
    
    for name, config in configs:
        # Apply enhanced physics-based refinement
        refined = enhanced_repulsive_forces(config, iterations=800, learning_rate=0.005)
        
        # Apply optimization
        optimized = optimize_with_improved_constraints(refined, max_iter=500)
        sum_radii = compute_objective(optimized)
        
        if sum_radii > best_sum and validate_circles(optimized):
            best_sum = sum_radii
            best_config = optimized
    
    # If none worked, use default
    if best_config is None:
        best_config = initialize_voronoi_better(N_CIRCLES)
    
    return best_config

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses enhanced Voronoi initialization + improved optimization + advanced physics simulation.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Use the enhanced smart placement strategy
    circles = smart_placement_strategy()
    
    # Apply more aggressive refinement with enhanced physics simulation
    refined_circles = enhanced_repulsive_forces(circles, iterations=1200, learning_rate=0.003)
    
    # Try global optimization with differential evolution for better results
    global_optimized = optimize_with_improved_constraints(refined_circles, max_iter=800)
    
    # Final optimization step with scipy
    final_circles = optimize_with_improved_constraints(global_optimized, max_iter=1200)
    
    # Final validation and fallback
    if not validate_circles(final_circles):
        # If invalid, fall back to a more conservative approach
        fallback = initialize_voronoi_better(N_CIRCLES)
        final_circles = optimize_with_improved_constraints(fallback, max_iter=600)
    
    return final_circles


# EVOLVE-BLOCK-END
