# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
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

def convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    if len(points) < 3:
        return 0
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume gives the area
    except:
        # Fallback: use bounding box area
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

def compute_min_area_normalized(points):
    """Compute normalized minimum triangle area (min area / convex hull area)"""
    min_area = compute_min_triangle_area(points)
    hull_area = convex_hull_area(points)
    if hull_area > 1e-10:
        return min_area / hull_area
    else:
        return 0

def generate_multiple_initial_configurations():
    """Generate multiple diverse initial configurations inspired by best practices."""
    configs = []
    n = 13
    
    # Configuration 1: Dodecagon + center (optimized mathematical construction)
    n_polygon = 12
    radius = 0.335
    center = np.array([0.5, 0.5])
    
    points = np.zeros((n_polygon, 2))
    for i in range(n_polygon):
        angle = 2 * np.pi * i / n_polygon
        points[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
    
    # Add central point
    points = np.vstack([points, center])
    
    # Apply rotation to break symmetry
    rotation_angle = np.pi / 16
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_points = points.copy()
    for i in range(n_polygon):
        rotated_points[i] = rotation_matrix @ (points[i] - center) + center
    
    # Add slight perturbations with better distribution
    np.random.seed(42)
    perturbations = np.random.uniform(-0.013, 0.013, (13, 2))
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
    points_spiral += np.random.normal(0, 0.012, points_spiral.shape)  # Slightly smaller noise
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
    points_grid += np.random.normal(0, 0.01, points_grid.shape)
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
    points_geometric += np.random.normal(0, 0.01, points_geometric.shape)
    points_geometric = np.clip(points_geometric, 0, 1)
    configs.append(points_geometric.copy())
    
    # Configuration 5: Fibonacci spiral with better distribution
    golden_ratio = (1 + np.sqrt(5)) / 2
    points_spiral = []
    for i in range(n):
        angle = i * 2 * np.pi / golden_ratio
        radius = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points_spiral.append([x, y])
    points_spiral = np.array(points_spiral)
    points_spiral += np.random.uniform(-0.01, 0.01, points_spiral.shape)
    points_spiral = np.clip(points_spiral, 0, 1)
    configs.append(points_spiral.copy())
    
    return configs

def smart_local_search(points, max_iter=50):
    """Smart local search with more efficient early stopping and strategic moves."""
    current_points = points.copy()
    best_min_area = compute_min_triangle_area(current_points)
    
    # Track improvement for early stopping
    improvement_threshold = 1e-12
    no_improvement_count = 0
    max_no_improvement = 5  # More aggressive early stopping for time efficiency
    
    # Comprehensive direction set for better exploration
    directions = [
        [1, 0], [0, 1], [-1, 0], [0, -1],  # Cardinal
        [1, 1], [-1, 1], [-1, -1], [1, -1],  # Diagonal
        [0.707, 0.707], [-0.707, 0.707], [-0.707, -0.707], [0.707, -0.707],  # Unit diagonals
        [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5],  # Half steps
        [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]  # Half diagonal steps
    ]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each point systematically
        for i in range(len(current_points)):
            old_point = current_points[i].copy()
            
            # Adaptive step sizes - start with larger steps, decrease if needed
            step_sizes = [0.01, 0.005, 0.002, 0.001]  # Added finer steps for precision
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
                improved = True
        
        # Early stopping if no significant improvement
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                break
        else:
            no_improvement_count = 0
    
    return current_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize 
    the area of the smallest triangle formed by these points.
    
    Uses a hybrid approach combining multiple initial configurations with 
    smart local search and advanced refinement strategies.
    """
    np.random.seed(42)
    
    # Generate multiple initial configurations (like INSPIRATION 1 and 3)
    initial_configs = generate_multiple_initial_configurations()
    
    best_points = None
    best_min_area = -1
    
    # Try each initial configuration with smart local search refinement
    for i, initial_config in enumerate(initial_configs):
        # Apply smart local search with moderate iterations for efficiency
        local_refined = smart_local_search(initial_config, max_iter=60)
        
        # Evaluate the result
        min_area = compute_min_triangle_area(local_refined)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = local_refined.copy()
    
    # Strategy 2: Multiple restarts with better perturbation strategy
    if best_points is not None:
        for restart in range(4):  # Reduced restarts to balance quality vs time
            np.random.seed(restart + 100)
            # Use more systematic perturbations
            perturbed = best_points.copy()
            for i in range(len(perturbed)):
                # Add controlled perturbations with varied magnitudes
                magnitude = 0.003 + np.random.random() * 0.004  # Random magnitude between 0.003 and 0.007
                perturbation = np.random.uniform(-magnitude, magnitude, 2)
                perturbed[i] += perturbation
            
            perturbed = np.clip(perturbed, 0, 1)
            
            # Apply smart local search to the perturbed points
            local_refined = smart_local_search(perturbed, max_iter=40)
            local_min_area = compute_min_triangle_area(local_refined)
            
            if local_min_area > best_min_area:
                best_points = local_refined
                best_min_area = local_min_area
    
    # Strategy 3: Final refinement with smart local search
    if best_points is not None:
        final_points = smart_local_search(best_points, max_iter=30)
        final_min_area = compute_min_triangle_area(final_points)
        
        if final_min_area > best_min_area:
            return final_points
        else:
            return best_points
    else:
        # Fallback to mathematical construction
        n_polygon = 12
        radius = 0.335
        center = np.array([0.5, 0.5])
        
        points = np.zeros((n_polygon, 2))
        for i in range(n_polygon):
            angle = 2 * np.pi * i / n_polygon
            points[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Add central point
        points = np.vstack([points, center])
        
        # Apply rotation to break symmetry
        rotation_angle = np.pi / 16
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_points = points.copy()
        for i in range(n_polygon):
            rotated_points[i] = rotation_matrix @ (points[i] - center) + center
        
        # Add slight perturbations
        np.random.seed(42)
        perturbations = np.random.uniform(-0.013, 0.013, (13, 2))
        rotated_points += perturbations
        rotated_points = np.clip(rotated_points, 0, 1)
        return rotated_points


# EVOLVE-BLOCK-END
