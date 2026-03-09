# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
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
    
    # Configuration 1: Dodecagon + center (improved mathematical construction)
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
    
    # Configuration 2: Golden spiral pattern with better distribution (inspired by INSPIRATION 2)
    np.random.seed(246)
    points_spiral = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(n):
        angle = 2 * np.pi * i * phi
        radius = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points_spiral.append([x, y])
    points_spiral = np.array(points_spiral)
    points_spiral += np.random.normal(0, 0.01, points_spiral.shape)
    points_spiral[:, 0] = np.clip(points_spiral[:, 0], 0, 1)
    points_spiral[:, 1] = np.clip(points_spiral[:, 1], 0, 1)
    configs.append(points_spiral.copy())
    
    # Configuration 3: Hexagonal grid pattern (inspired by INSPIRATION 2)
    np.random.seed(42)
    hex_points = []
    rows, cols = 4, 4
    for i in range(rows):
        for j in range(cols):
            if len(hex_points) < n:
                x = 0.1 + 0.8 * j / max(1, cols - 1)
                y = 0.1 + 0.8 * i / max(1, rows - 1)
                # Add hexagonal offset for better distribution
                if i % 2 == 1:
                    x += 0.15 / max(1, cols - 1)
                hex_points.append([x, y])
    
    points_hex = np.array(hex_points[:n])
    points_hex += np.random.normal(0, 0.01, points_hex.shape)
    points_hex = np.clip(points_hex, 0, 1)
    configs.append(points_hex.copy())
    
    # Configuration 4: Strategic geometric pattern (inspired by INSPIRATION 1)
    np.random.seed(123)
    points_geometric = []
    
    # Add center point
    points_geometric.append([0.5, 0.5])
    
    # Add points in concentric rings
    radii = [0.25, 0.4, 0.45]
    for r in radii:
        n_points = 6 if r == 0.25 else 8 if r == 0.4 else 6
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        for angle in angles:
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            points_geometric.append([x, y])
    
    # Fill remaining spots with uniform random distribution
    while len(points_geometric) < n:
        points_geometric.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    points_geometric = np.array(points_geometric[:n])
    points_geometric += np.random.normal(0, 0.015, points_geometric.shape)
    points_geometric = np.clip(points_geometric, 0, 1)
    configs.append(points_geometric.copy())
    
    # Configuration 5: Improved circle pattern (inspired by INSPIRATION 2)
    np.random.seed(246)
    circle_points = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        # Distribute points more evenly in polar coordinates
        radius = 0.4 * (0.7 + 0.3 * np.sin(i * 0.5))  # Slight variation in radius
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        circle_points.append([x, y])
    circle_points = np.array(circle_points)
    circle_points += np.random.normal(0, 0.012, circle_points.shape)
    circle_points = np.clip(circle_points, 0, 1)
    configs.append(circle_points.copy())
    
    return configs

def enhanced_smart_local_search(points, max_iter=50):
    """Enhanced smart local search with better convergence and reduced iterations."""
    current_points = points.copy()
    best_min_area = compute_min_triangle_area(current_points)
    
    # Track improvement for early stopping with more aggressive criteria
    improvement_threshold = 1e-12
    no_improvement_count = 0
    max_no_improvement = 5  # More aggressive early stopping for efficiency
    
    # Direction set inspired by INSPIRATION 2 - more focused directions
    directions = [
        [1, 0], [0, 1], [-1, 0], [0, -1],  # Cardinal
        [0.707, 0.707], [-0.707, 0.707], [-0.707, -0.707], [0.707, -0.707],  # Unit diagonals
        [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5],  # Half steps
    ]
    
    # More efficient step size progression
    step_sizes = [0.01, 0.005, 0.002, 0.001]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each point to potentially improve minimum triangle area
        for i in range(len(current_points)):
            # Save current point
            old_point = current_points[i].copy()
            
            best_move = old_point.copy()
            best_improvement = 0
            
            # Test moves in predefined directions with different step sizes
            for step_size in step_sizes:
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

def advanced_gradient_optimization(points, max_restarts=3):
    """Advanced optimization with multiple restarts and better convergence checking."""
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        min_area = compute_min_triangle_area(points)
        return -min_area  # Negative because we want to maximize
    
    best_points = points.copy()
    best_min_area = compute_min_triangle_area(points)
    
    # Multiple restarts with different perturbations
    for restart in range(max_restarts):
        try:
            # Create perturbed starting point
            np.random.seed(restart + 100)
            perturbed = points.copy()
            # Add different perturbations for each restart (smaller perturbations for speed)
            noise = np.random.normal(0, 0.003, points.shape)
            perturbed += noise
            perturbed = np.clip(perturbed, 0, 1)
            
            x0 = perturbed.flatten()
            bounds = [(0, 1) for _ in range(len(x0))]
            
            # Use L-BFGS-B with good parameters - more efficient than other methods
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_area = compute_min_triangle_area(refined_points)
                
                if refined_area > best_min_area:
                    best_min_area = refined_area
                    best_points = refined_points.copy()
                    
        except Exception:
            continue
    
    return best_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize 
    the area of the smallest triangle formed by these points.
    
    Uses a sophisticated hybrid approach combining mathematical initial configurations
    with advanced optimization techniques inspired by state-of-the-art methods.
    """
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configurations()
    
    best_points = None
    best_min_area = -1
    
    # Try each initial configuration with enhanced optimization strategies
    for i, initial_config in enumerate(initial_configs):
        # Apply enhanced local search refinement
        local_refined = enhanced_smart_local_search(initial_config, max_iter=70)
        
        # Further optimization with gradient-based method
        optimized_config = advanced_gradient_optimization(local_refined, max_restarts=3)
        
        # Evaluate the result
        min_area = compute_min_triangle_area(optimized_config)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = optimized_config.copy()
    
    # Strategy 2: Multiple restarts with better perturbation strategy
    if best_points is not None:
        for restart in range(2):  # Reduced from 3 to save time
            np.random.seed(restart + 100)
            # Use more systematic perturbations with varied magnitudes
            perturbed = best_points.copy()
            for i in range(len(perturbed)):
                # Add controlled perturbations with smaller magnitudes for efficiency
                magnitude = 0.001 + np.random.random() * 0.005  # Smaller range for faster convergence
                perturbation = np.random.uniform(-magnitude, magnitude, 2)
                perturbed[i] += perturbation
            
            perturbed = np.clip(perturbed, 0, 1)
            
            # Apply enhanced local search to the perturbed points
            local_refined = enhanced_smart_local_search(perturbed, max_iter=40)  # Reduced iterations
            local_min_area = compute_min_triangle_area(local_refined)
            
            if local_min_area > best_min_area:
                best_points = local_refined
                best_min_area = local_min_area
    
    # Strategy 3: Differential evolution with tuned parameters for time efficiency
    try:
        def objective_function(x_flat):
            points = x_flat.reshape(-1, 2)
            points = np.clip(points, 0, 1)
            min_area = compute_min_triangle_area(points)
            # Return negative for minimization (we want to maximize)
            return -min_area
        
        bounds = [(0, 1) for _ in range(2 * 13)]
        result_de = differential_evolution(
            objective_function,
            bounds,
            maxiter=10,  # Reduced iterations significantly for time efficiency
            popsize=6,   # Reduced population size for faster execution
            seed=42,
            disp=False,
            strategy='best1bin'
        )
        
        if result_de.success:
            de_points = result_de.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_min_area = compute_min_triangle_area(de_points)
            
            if de_min_area > best_min_area:
                best_points = de_points
                best_min_area = de_min_area
                
    except Exception:
        pass
    
    # Strategy 4: Final refinement with enhanced local search
    if best_points is not None:
        final_points = enhanced_smart_local_search(best_points, max_iter=30)  # Reduced iterations
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
