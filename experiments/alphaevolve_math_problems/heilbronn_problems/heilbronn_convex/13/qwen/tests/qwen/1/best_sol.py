# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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

def generate_multiple_initial_configurations():
    """Generate multiple diverse initial configurations with focus on efficiency."""
    configs = []
    n = 13
    
    # Configuration 1: Mathematical construction with better distribution
    n_polygon = 12
    radius = 0.35
    center = np.array([0.5, 0.5])
    
    points = np.zeros((n_polygon, 2))
    for i in range(n_polygon):
        angle = 2 * np.pi * i / n_polygon
        points[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
    
    # Add central point
    points = np.vstack([points, center])
    
    # Apply rotation to break symmetry with more precise rotation
    rotation_angle = np.pi / 12
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_points = points.copy()
    for i in range(n_polygon):
        rotated_points[i] = rotation_matrix @ (points[i] - center) + center
    
    # Add slight perturbations with better distribution
    np.random.seed(42)
    perturbations = np.random.uniform(-0.015, 0.015, (13, 2))
    rotated_points += perturbations
    rotated_points = np.clip(rotated_points, 0, 1)
    configs.append(rotated_points.copy())
    
    # Configuration 2: Golden spiral pattern (more efficient than complex spirals)
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
    
    # Configuration 3: Hexagonal grid pattern (better spatial distribution)
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
    
    return configs

def enhanced_local_search(points, max_iterations=50):
    """Efficient local search with focused direction set for better speed."""
    current_points = points.copy()
    best_min_area = compute_min_triangle_area(current_points)
    
    # Track improvement for early stopping
    improvement_threshold = 1e-12
    no_improvement_count = 0
    max_no_improvement = 5  # Aggressive early stopping for speed
    
    # More focused direction set for better performance
    directions = [
        [1, 0], [0, 1], [-1, 0], [0, -1],  # Cardinal
        [0.707, 0.707], [-0.707, 0.707], [-0.707, -0.707], [0.707, -0.707],  # Unit diagonals
        [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5],  # Half steps
    ]
    
    # More efficient step size progression
    step_sizes = [0.01, 0.005, 0.002, 0.001]
    
    for iteration in range(max_iterations):
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

def simulated_annealing_optimization(points, max_iterations=80):
    """Simulated annealing optimization for global search with reduced iterations."""
    current_points = points.copy()
    current_min_area = compute_min_triangle_area(current_points)
    best_points = current_points.copy()
    best_min_area = current_min_area
    
    # Annealing parameters - optimized for speed and effectiveness
    temp = 0.03
    cooling_rate = 0.995
    min_temp = 1e-6
    
    for iteration in range(max_iterations):
        # Make a small random perturbation to a random point
        point_idx = np.random.randint(0, len(current_points))
        new_points = current_points.copy()
        
        # Perturb the selected point with controlled magnitude
        delta = np.random.normal(0, 0.003, 2)  # Even smaller perturbation for efficiency
        new_points[point_idx] = np.clip(current_points[point_idx] + delta, 0, 1)
        
        # Calculate new minimum area
        new_min_area = compute_min_triangle_area(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_min_area > current_min_area:
            current_points = new_points.copy()
            current_min_area = new_min_area
        else:
            # Accept with probability based on temperature
            if np.random.random() < np.exp((new_min_area - current_min_area) / temp):
                current_points = new_points.copy()
                current_min_area = new_min_area
        
        # Update best solution
        if current_min_area > best_min_area:
            best_min_area = current_min_area
            best_points = current_points.copy()
        
        # Cool down temperature
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
    
    return best_points

def advanced_gradient_optimization(points, max_restarts=2):
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
            # Add different perturbations for each restart
            noise = np.random.normal(0, 0.002, points.shape)  # Slightly smaller noise
            perturbed += noise
            perturbed = np.clip(perturbed, 0, 1)
            
            x0 = perturbed.flatten()
            bounds = [(0, 1) for _ in range(len(x0))]
            
            # Use L-BFGS-B with good parameters - faster than other methods
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 40, 'ftol': 1e-8, 'gtol': 1e-8},  # Reduced iterations for speed
                tol=1e-8
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
    
    Uses a streamlined hybrid approach combining mathematical constructions, 
    local search, and optimization strategies for efficient computation.
    """
    np.random.seed(42)
    
    # Generate fewer but more effective initial configurations
    initial_configs = generate_multiple_initial_configurations()
    
    best_points = None
    best_min_area = -1
    
    # Try each initial configuration with optimized optimization pipeline
    for i, initial_config in enumerate(initial_configs):
        # Apply enhanced local search for quick improvement
        local_refined = enhanced_local_search(initial_config, max_iterations=40)
        
        # Apply simulated annealing optimization for global search
        sa_result = simulated_annealing_optimization(local_refined, max_iterations=80)
        
        # Apply advanced gradient optimization for further refinement
        optimized_config = advanced_gradient_optimization(sa_result, max_restarts=1)
        
        # Evaluate the result
        min_area = compute_min_triangle_area(optimized_config)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = optimized_config.copy()
    
    # Final refinement with minimal iterations to save time
    if best_points is not None:
        # Apply enhanced local search for final polishing
        final_points = enhanced_local_search(best_points, max_iterations=25)
        
        return final_points
    else:
        # Fallback to mathematical construction
        n_polygon = 12
        radius = 0.35
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
        perturbations = np.random.uniform(-0.015, 0.015, (13, 2))
        rotated_points += perturbations
        rotated_points = np.clip(rotated_points, 0, 1)
        return rotated_points


# EVOLVE-BLOCK-END
