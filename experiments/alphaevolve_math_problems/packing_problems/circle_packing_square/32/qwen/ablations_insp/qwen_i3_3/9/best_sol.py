# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from scipy.optimize import differential_evolution
from scipy.spatial import KDTree

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

def initialize_hexagonal_layout(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal grid pattern for good starting configuration"""
    circles = np.zeros((n, 3))
    
    # Create a hexagonal lattice pattern
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + 1) * spacing_x + x_offset * spacing_x
            y = (i + 1) * spacing_y
            # Initial radius - small enough to fit in square
            r = min(spacing_x, spacing_y) * 0.4
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

def initialize_lattice_placement(n: int) -> np.ndarray:
    """Initialize circles in a structured lattice pattern"""
    circles = np.zeros((n, 3))
    
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Grid spacing
    grid_size = min(1.0/cols, 1.0/rows)
    
    # Fill with circles
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) * grid_size
            y = (i + 0.5) * grid_size
            r = grid_size * 0.4  # Make radii smaller than spacing to allow for optimization
            
            # Ensure it fits in the unit square
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                circles[idx] = [x, y, r]
                idx += 1
            else:
                # Fallback for boundary cases
                circles[idx] = [0.5, 0.5, 0.05]
                idx += 1
        if idx >= n:
            break
    
    return circles

def initialize_voronoi_placement(n: int) -> np.ndarray:
    """Initialize circles using Voronoi-like distribution"""
    circles = np.zeros((n, 3))
    
    # Generate random points first
    points = np.random.rand(n, 2)
    
    # Create initial circles with small radii
    for i in range(n):
        x, y = points[i]
        # Small initial radius
        r = 0.02
        circles[i] = [x, y, r]
    
    return circles

def initialize_physics_based(n: int) -> np.ndarray:
    """Initialize using physics-inspired approach with spatial data structure"""
    circles = np.zeros((n, 3))
    
    # Start with a basic grid pattern
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
            r = min(spacing_x, spacing_y) * 0.3
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions randomly but with better spatial awareness
    for i in range(idx, n):
        # Try to place with some consideration for neighbors
        attempts = 0
        while attempts < 1000:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            
            # Check if circle fits in square
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                # Quick check against existing circles using KDTree for efficiency
                if idx > 0:
                    positions = circles[:idx, :2]
                    tree = KDTree(positions)
                    distances, indices = tree.query([[x, y]], k=min(5, idx))
                    valid = True
                    for d, idx_neighbor in zip(distances[0], indices[0]):
                        if d < (r + circles[idx_neighbor, 2]):
                            valid = False
                            break
                    if valid:
                        circles[i] = [x, y, r]
                        break
                else:
                    circles[i] = [x, y, r]
                    break
            attempts += 1
        if attempts >= 1000:
            # Fallback
            circles[i] = [0.5, 0.5, 0.05]
    
    return circles

def generate_improved_initial(n: int) -> np.ndarray:
    """Generate improved initial circle placement using Voronoi with better radius calculation."""
    # Use a more systematic approach to Voronoi generation
    # Generate points in a structured way to get better initial distribution
    points = []
    
    # Create a grid-like pattern with some randomness
    grid_size = int(np.ceil(np.sqrt(n)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n:
                # Add some jitter to make it less regular
                x = (j + 0.5 + np.random.normal(0, 0.1)) * (1.0 / grid_size)
                y = (i + 0.5 + np.random.normal(0, 0.1)) * (1.0 / grid_size)
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
    
    points = np.array(points[:n])
    
    # Use Voronoi approach but with better radius calculation
    circles = np.zeros((n, 3))
    
    try:
        vor = Voronoi(points)
        
        # For each Voronoi cell, place a circle at the centroid with appropriate radius
        for i in range(n):
            if i < len(vor.points):
                # Get the point
                point = vor.points[i]
                
                # Calculate maximum possible radius for this circle
                # Find the minimum distance to any boundary or other points
                min_distance = float('inf')
                
                # Check distance to boundaries of unit square
                min_distance = min(min_distance, point[0])  # left boundary
                min_distance = min(min_distance, 1 - point[0])  # right boundary
                min_distance = min(min_distance, point[1])  # bottom boundary
                min_distance = min(min_distance, 1 - point[1])  # top boundary
                
                # Check distance to other points (for Voronoi cell size estimation)
                for j in range(len(vor.points)):
                    if i != j:
                        dist = np.linalg.norm(point - vor.points[j])
                        min_distance = min(min_distance, dist/2)
                
                # Set radius to a reasonable fraction of the minimum distance
                radius = min(0.1, min_distance * 0.7)
                # Make sure it's not too small
                radius = max(0.01, radius)
                circles[i] = [point[0], point[1], radius]
    except:
        # Fallback to simple random placement
        for i in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            radius = np.random.uniform(0.02, 0.08)
            circles[i] = [x, y, radius]
    
    return circles

def optimize_with_constraints(initial_circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Optimize using constrained optimization approach."""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds: [x0, y0, r0, x1, y1, r1, ...]
    bounds = []
    for i in range(n):
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.49)])  # x, y, r bounds
    
    # Define constraint functions
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
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    try:
        # Use SLSQP optimization method
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-6, 'eps': 1e-4}
        )
        
        if result.success:
            final_circles = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_circles.append([x, y, r])
            return np.array(final_circles)
        else:
            # Return initial circles if optimization fails
            return initial_circles
            
    except Exception as e:
        # Return initial circles if optimization fails
        return initial_circles

def optimize_with_differential_evolution(circles: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Use differential evolution for global optimization"""
    n = len(circles)
    
    # Flatten for optimization
    x0 = circles.flatten()
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x_flat):
        circles_flat = x_flat.reshape(-1, 3)
        return -np.sum(circles_flat[:, 2])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Try differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result.success:
            circles_opt = result.x.reshape(-1, 3)
            # Validate final solution
            if validate_circles(circles_opt):
                return circles_opt
    except Exception as e:
        pass
    
    return circles

def apply_repulsive_forces(circles: np.ndarray, iterations: int = 1000, learning_rate: float = 0.01) -> np.ndarray:
    """Apply physics-based repulsive forces for better packing"""
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
                
                # Only apply force if circles are overlapping or close
                if distance > 0 and distance < (r1 + r2):
                    force_magnitude = (r1 + r2 - distance) / (distance + 1e-8)
                    forces[i, 0] += force_magnitude * dx / distance
                    forces[i, 1] += force_magnitude * dy / distance
                    forces[j, 0] -= force_magnitude * dx / distance
                    forces[j, 1] -= force_magnitude * dy / distance
        
        # Apply forces with boundary constraints
        for i in range(n):
            # Move circle
            new_x = circles[i, 0] + learning_rate * forces[i, 0]
            new_y = circles[i, 1] + learning_rate * forces[i, 1]
            
            # Boundary constraints
            new_x = np.clip(new_x, circles[i, 2], 1 - circles[i, 2])
            new_y = np.clip(new_y, circles[i, 2], 1 - circles[i, 2])
            
            circles[i, 0] = new_x
            circles[i, 1] = new_y
            
        # Occasionally try to increase radii
        if iter_num % 100 == 0:
            for i in range(n):
                x, y, r = circles[i]
                # Try to increase radius if possible
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check distance to other circles
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        max_radius = min(max_radius, distance - r2)
                
                if max_radius > r and max_radius > 0.001:
                    circles[i, 2] = max_radius
    
    return circles

def smart_placement_strategy() -> np.ndarray:
    """Use a multi-strategy approach to find good initial configuration"""
    
    # Strategy 1: Physics-based initialization (most promising)
    physics_config = initialize_physics_based(N_CIRCLES)
    
    # Strategy 2: Hexagonal layout 
    hex_config = initialize_hexagonal_layout(N_CIRCLES)
    
    # Strategy 3: Lattice placement
    lattice_config = initialize_lattice_placement(N_CIRCLES)
    
    # Strategy 4: Voronoi-like placement
    voronoi_config = initialize_voronoi_placement(N_CIRCLES)
    
    # Strategy 5: Random placement
    random_config = np.zeros((N_CIRCLES, 3))
    for i in range(N_CIRCLES):
        r = np.random.uniform(0.01, 0.1)
        x = np.random.uniform(r, 1-r)
        y = np.random.uniform(r, 1-r)
        random_config[i] = [x, y, r]
    
    # Strategy 6: Improved Voronoi placement
    voronoi_improved = generate_improved_initial(N_CIRCLES)
    
    # Evaluate all strategies
    configs = [
        ("physics", physics_config),
        ("hexagonal", hex_config),
        ("lattice", lattice_config),
        ("voronoi", voronoi_config),
        ("random", random_config),
        ("voronoi_improved", voronoi_improved)
    ]
    
    best_config = None
    best_sum = -1
    
    for name, config in configs:
        # Apply physics-based refinement first
        refined = apply_repulsive_forces(config, iterations=500, learning_rate=0.005)
        
        # Then apply optimization
        optimized = optimize_with_constraints(refined, max_iter=300)
        sum_radii = compute_objective(optimized)
        
        if sum_radii > best_sum and validate_circles(optimized):
            best_sum = sum_radii
            best_config = optimized
    
    # If none worked, use default
    if best_config is None:
        best_config = initialize_physics_based(N_CIRCLES)
    
    return best_config

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved Voronoi initialization + gradient-based optimization + physics simulation.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Use the smart placement strategy to get a good initial configuration
    circles = smart_placement_strategy()
    
    # Apply additional refinement with physics simulation
    refined_circles = apply_repulsive_forces(circles, iterations=1000, learning_rate=0.002)
    
    # Try global optimization with differential evolution for better results
    global_optimized = optimize_with_differential_evolution(refined_circles, max_iter=50)
    
    # Final optimization step with scipy
    final_circles = optimize_with_constraints(global_optimized, max_iter=1000)
    
    # Final validation
    if not validate_circles(final_circles):
        # If invalid, fall back to a more conservative approach
        fallback = initialize_physics_based(N_CIRCLES)
        final_circles = optimize_with_constraints(fallback, max_iter=500)
    
    return final_circles


# EVOLVE-BLOCK-END
