# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def triangle_area(p1, p2, p3):
    """Calculate the area of triangle formed by three points using cross product formula"""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def compute_min_triangle_area(points):
    """Compute the minimum area of all triangles formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0
    
    min_area = float('inf')
    
    # Use more efficient combination generation
    for combo in combinations(range(n), 3):
        area = triangle_area(points[combo[0]], points[combo[1]], points[combo[2]])
        min_area = min(min_area, area)
    
    return min_area

def compute_min_area_normalized(points):
    """Compute normalized minimum triangle area (min area / convex hull area)"""
    min_area = compute_min_triangle_area(points)
    # For unit area convex hull, we just return min_area
    return min_area

def generate_multiple_initial_configurations():
    """Generate multiple diverse initial configurations inspired by best practices."""
    configs = []
    n = 13
    
    # Configuration 1: Dodecagon + center (improved mathematical construction)
    n_polygon = 12
    radius = 0.34
    center = np.array([0.5, 0.5])
    
    points = np.zeros((n_polygon, 2))
    for i in range(n_polygon):
        angle = 2 * np.pi * i / n_polygon
        points[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
    
    # Add central point
    points = np.vstack([points, center])
    
    # Apply rotation to break symmetry
    rotation_angle = np.pi / 14
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_points = points.copy()
    for i in range(n_polygon):
        rotated_points[i] = rotation_matrix @ (points[i] - center) + center
    
    # Add slight perturbations with better distribution
    np.random.seed(42)
    perturbations = np.random.uniform(-0.014, 0.014, (13, 2))
    rotated_points += perturbations
    rotated_points = np.clip(rotated_points, 0, 1)
    configs.append(rotated_points.copy())
    
    # Configuration 2: Spiral pattern with better spacing
    points_spiral = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        radius = 0.4 * (i / n) + 0.05  # Add offset to avoid center clustering
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points_spiral.append([x, y])
    points_spiral = np.array(points_spiral)
    points_spiral += np.random.normal(0, 0.013, points_spiral.shape)  # Slightly smaller noise
    points_spiral[:, 0] = np.clip(points_spiral[:, 0], 0, 1)
    points_spiral[:, 1] = np.clip(points_spiral[:, 1], 0, 1)
    configs.append(points_spiral.copy())
    
    # Configuration 3: Grid-based with random perturbations
    np.random.seed(42)
    grid_points = []
    rows, cols = 4, 4
    for i in range(rows):
        for j in range(cols):
            if len(grid_points) < n:
                x = 0.1 + 0.8 * j / max(1, cols - 1)
                y = 0.1 + 0.8 * i / max(1, rows - 1)
                # Add some hexagonal offset
                if i % 2 == 1:
                    x += 0.2 / max(1, cols - 1)
                grid_points.append([x, y])
    
    points_grid = np.array(grid_points[:n])
    points_grid += np.random.normal(0, 0.012, points_grid.shape)
    points_grid = np.clip(points_grid, 0, 1)
    configs.append(points_grid.copy())
    
    # Configuration 4: Strategic geometric construction inspired by known solutions
    np.random.seed(123)
    points_geometric = []
    
    # Add center point
    points_geometric.append([0.5, 0.5])
    
    # Add points in strategic positions around the circle
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for angle in angles:
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        points_geometric.append([x, y])
    
    # Add outer ring
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    for angle in angles_outer:
        x = 0.5 + 0.45 * np.cos(angle)
        y = 0.5 + 0.45 * np.sin(angle)
        points_geometric.append([x, y])
    
    # Add remaining points randomly but distributed
    while len(points_geometric) < n:
        points_geometric.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    points_geometric = np.array(points_geometric[:n])
    points_geometric += np.random.normal(0, 0.012, points_geometric.shape)
    points_geometric = np.clip(points_geometric, 0, 1)
    configs.append(points_geometric.copy())
    
    # Configuration 5: Golden ratio spiral for better distribution
    points_golden = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(n):
        angle = 2 * np.pi * i / phi
        radius = np.sqrt(i / (n - 1)) * 0.4 + 0.05
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points_golden.append([x, y])
    points_golden = np.array(points_golden)
    points_golden += np.random.normal(0, 0.01, points_golden.shape)
    points_golden[:, 0] = np.clip(points_golden[:, 0], 0, 1)
    points_golden[:, 1] = np.clip(points_golden[:, 1], 0, 1)
    configs.append(points_golden.copy())
    
    return configs

def enhanced_local_search(points, max_iter=100):
    """Enhanced local search with comprehensive direction set and better convergence."""
    current_points = points.copy()
    best_min_area = compute_min_triangle_area(current_points)
    
    # Track improvement for early stopping
    improvement_threshold = 1e-12
    no_improvement_count = 0
    max_no_improvement = 15  # More conservative for better convergence
    
    # Comprehensive direction set inspired by best practices
    directions = [
        [1, 0], [0, 1], [-1, 0], [0, -1],  # Cardinal
        [1, 1], [-1, 1], [-1, -1], [1, -1],  # Diagonal
        [0.707, 0.707], [-0.707, 0.707], [-0.707, -0.707], [0.707, -0.707],  # Unit diagonals
        [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5],  # Half steps
        [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]  # Half diagonal steps
    ]
    
    for iteration in range(max_iter):
        improved = False
        total_improvement = 0
        
        # Try moving each point systematically
        for i in range(len(current_points)):
            old_point = current_points[i].copy()
            
            # Adaptive step sizes - start with larger steps, decrease if needed
            step_sizes = [0.01, 0.005, 0.002, 0.001]  # Added finer steps for better precision
            best_move = old_point.copy()
            best_improvement = 0
            
            for step_size in step_sizes:
                # Test moves in multiple directions for better coverage
                for dx, dy in directions:
                    move = np.array([dx * step_size, dy * step_size])
                    new_point = old_point + move
                    
                    # Keep within bounds
                    new_point[0] = np.clip(new_point[0], 0, 1)
                    new_point[1] = np.clip(new_point[1], 0, 1)
                    
                    # Test this move
                    test_points = current_points.copy()
                    test_points[i] = new_point
                    
                    min_area = compute_min_triangle_area(test_points)
                    improvement = min_area - best_min_area
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = new_point.copy()
            
            # Apply the best move if it improves the configuration
            if best_improvement > improvement_threshold:
                current_points[i] = best_move.copy()
                best_min_area += best_improvement
                total_improvement += best_improvement
                improved = True
        
        # Early stopping if no significant improvement
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                break
        else:
            no_improvement_count = 0
    
    return current_points

def hybrid_optimization(points, max_evaluations=200):
    """Hybrid optimization combining differential evolution and local search."""
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        min_area = compute_min_triangle_area(points)
        return -min_area  # Negative because we want to maximize
    
    n = len(points)
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # First try differential evolution for global exploration
    try:
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            popsize=15,
            seed=42,
            disp=False,
            strategy='best1bin',
            atol=1e-8,
            rtol=1e-8
        )
        
        if result_de.success:
            de_points = result_de.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_min_area = compute_min_triangle_area(de_points)
            
            # If DE found a better solution, refine it locally
            if de_min_area > compute_min_triangle_area(points):
                points = de_points.copy()
    except Exception:
        pass
    
    # Then apply enhanced local search for fine-tuning
    refined_points = enhanced_local_search(points, max_iter=80)
    
    return refined_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize 
    the area of the smallest triangle formed by these points.
    
    Uses a hybrid approach combining multiple initial configurations with 
    enhanced local search, differential evolution, and advanced refinement strategies.
    """
    n = 13
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configurations()
    
    best_points = None
    best_min_area = -1
    
    # Try each initial configuration with hybrid optimization
    for i, initial_config in enumerate(initial_configs):
        # Apply hybrid optimization (DE + local search)
        optimized_config = hybrid_optimization(initial_config, max_evaluations=200)
        
        # Evaluate the result
        min_area = compute_min_triangle_area(optimized_config)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = optimized_config.copy()
    
    # Additional refinement with multiple restarts
    if best_points is not None:
        # Try a few more local optimizations from different starting points
        for restart in range(3):
            np.random.seed(restart + 100)
            # Perturb the best solution slightly
            perturbed = best_points.copy()
            noise = np.random.normal(0, 0.005, perturbed.shape)
            perturbed += noise
            perturbed = np.clip(perturbed, 0, 1)
            
            # Apply hybrid optimization to the perturbed points
            refined = hybrid_optimization(perturbed, max_evaluations=150)
            refined_area = compute_min_triangle_area(refined)
            
            if refined_area > best_min_area:
                best_min_area = refined_area
                best_points = refined.copy()
    
    # Final comprehensive local search
    if best_points is not None:
        final_points = enhanced_local_search(best_points, max_iter=60)
        final_min_area = compute_min_triangle_area(final_points)
        
        if final_min_area > best_min_area:
            return final_points
        else:
            return best_points
    else:
        # Fallback to a well-known configuration
        n_polygon = 12
        radius = 0.34
        center = np.array([0.5, 0.5])
        
        points = np.zeros((n_polygon, 2))
        for i in range(n_polygon):
            angle = 2 * np.pi * i / n_polygon
            points[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Add central point
        points = np.vstack([points, center])
        
        # Apply rotation to break symmetry
        rotation_angle = np.pi / 12
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_points = points.copy()
        for i in range(n_polygon):
            rotated_points[i] = rotation_matrix @ (points[i] - center) + center
        
        # Add slight perturbations
        np.random.seed(42)
        perturbations = np.random.uniform(-0.012, 0.012, (13, 2))
        rotated_points += perturbations
        rotated_points = np.clip(rotated_points, 0, 1)
        return rotated_points


# EVOLVE-BLOCK-END
