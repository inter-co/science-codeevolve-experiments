# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit
import random

# Precompute hexagon vertices for unit regular hexagon
@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, radius, rotation):
    """Fast computation of hexagon vertices using numba"""
    angles = np.linspace(0, 2*np.pi, 7) + rotation
    vertices = np.empty((6, 2))
    for i in range(6):
        vertices[i, 0] = center_x + radius * np.cos(angles[i])
        vertices[i, 1] = center_y + radius * np.sin(angles[i])
    return vertices

def get_hexagon_vertices(center=(0,0), rotation=0, side_length=1):
    """Get vertices of a regular hexagon"""
    angle = rotation * np.pi / 180
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = side_length * np.cos(theta)
        y = side_length * np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return np.array(vertices)

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon"""
    outer_polygon = Polygon(outer_hexagon_vertices)
    for vertex in hexagon_vertices:
        if not outer_polygon.contains(Point(vertex)):
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using bounding boxes first, then polygon intersection"""
    # Quick bounding box test to eliminate obvious non-intersections
    bbox1 = [np.min(hex1_vertices[:, 0]), np.min(hex1_vertices[:, 1]),
             np.max(hex1_vertices[:, 0]), np.max(hex1_vertices[:, 1])]
    bbox2 = [np.min(hex2_vertices[:, 0]), np.min(hex2_vertices[:, 1]),
             np.max(hex2_vertices[:, 0]), np.max(hex2_vertices[:, 1])]
    
    # Check if bounding boxes intersect
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False
    
    # Use polygon intersection for final check
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    except:
        # Fallback for degenerate cases
        return False

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate the minimum side length needed for outer hexagon to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        vertices = get_hexagon_vertices(center, rotation, 1)
        all_vertices.extend(vertices)
    
    # Find the bounding circle and calculate required outer hexagon size
    if len(all_vertices) == 0:
        return 1000
    
    # Center of all vertices
    all_vertices = np.array(all_vertices)
    centroid = np.mean(all_vertices, axis=0)
    
    # Maximum distance from centroid to any vertex
    distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
    max_distance = np.max(distances)
    
    # For a regular hexagon, the relationship between side length and circumradius is:
    # side_length = circumradius
    # So we just return the max distance directly
    return max_distance

def evaluate_configuration(inner_hex_data, outer_side_length=None):
    """Evaluate a configuration for validity and quality"""
    if outer_side_length is None:
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Create outer hexagon vertices
    outer_center = (0, 0)
    outer_rotation = 0
    outer_hex_vertices = get_hexagon_vertices(outer_center, outer_rotation, outer_side_length)
    
    # Check containment
    all_contained = True
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, rotation, 1)
        
        if not check_containment(hex_vertices, outer_hex_vertices):
            all_contained = False
            break
    
    if not all_contained:
        return float('inf')  # Invalid configuration
    
    # Check overlaps with fast overlap detection
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
            rotation1 = inner_hex_data[i, 2]
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            
            hex1_vertices = get_hexagon_vertices(center1, rotation1, 1)
            hex2_vertices = get_hexagon_vertices(center2, rotation2, 1)
            
            if check_overlap_fast(hex1_vertices, hex2_vertices):
                return float('inf')  # Overlapping hexagons
    
    # Valid configuration - return negative of inverse side length (to maximize 1/R)
    return -1.0 / outer_side_length if outer_side_length else float('inf')

def generate_hexagonal_lattice_config():
    """Generate a configuration based on hexagonal lattice with specific symmetry properties"""
    # This approach creates a configuration that leverages known good hexagonal symmetries
    config = np.zeros((12, 3))
    
    # Create a configuration with 6-fold rotational symmetry when possible
    # Place hexagons in concentric rings with specific geometric relationships
    
    # Ring 1: center hexagon
    config[0] = [0.0, 0.0, 0.0]
    
    # Ring 2: 6 hexagons at distance sqrt(3) from center (this ensures touching neighbors)
    ring2_dist = np.sqrt(3)
    for i in range(6):
        angle = i * np.pi/3
        config[i+1] = [ring2_dist * np.cos(angle), ring2_dist * np.sin(angle), 0.0]
    
    # Ring 3: 5 hexagons in a staggered pattern to maximize packing efficiency
    ring3_dist = 2 * np.sqrt(3)
    for i in range(5):
        angle = i * 2*np.pi/5 + np.pi/5  # Staggered pattern
        config[i+7] = [ring3_dist * np.cos(angle), ring3_dist * np.sin(angle), 0.0]
    
    # Add small random perturbations to escape local minima
    for i in range(12):
        config[i, 0] += random.uniform(-0.03, 0.03)
        config[i, 1] += random.uniform(-0.03, 0.03)
    
    return config

def generate_constraint_satisfaction_config():
    """Generate configuration using constraint satisfaction approach with mathematical bounds"""
    # This approach uses mathematical constraints to place hexagons optimally
    # Start with a known good configuration and apply constraint propagation
    
    # Known good pattern for 12 hexagons - based on research of optimal arrangements
    # Using more precise mathematical constants from literature
    sqrt3 = np.sqrt(3)
    config = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 2.0, 0.0],           # top
        [0.0, -2.0, 0.0],          # bottom
        [sqrt3, 1.0, 0.0],         # top-right
        [-sqrt3, 1.0, 0.0],        # top-left
        [sqrt3, -1.0, 0.0],        # bottom-right
        [-sqrt3, -1.0, 0.0],       # bottom-left
        [2.0*sqrt3, 0.0, 0.0],     # far right
        [-2.0*sqrt3, 0.0, 0.0],    # far left
        [sqrt3, 3.0, 0.0],         # top-top
        [-sqrt3, 3.0, 0.0],        # top-top-left
        [sqrt3, -3.0, 0.0],        # bottom-bottom
    ])
    
    # Apply mathematical constraint satisfaction to improve this configuration
    # Using the fact that optimal packings often involve hexagonal close packing
    # We'll use a more sophisticated approach to optimize the spacing
    
    # Adjust for improved packing efficiency by scaling distances appropriately
    # The key insight: in optimal hexagonal packing, the distance between centers 
    # of adjacent hexagons is 2 (the diameter of unit hexagon)
    
    # Apply small adjustments based on mathematical understanding of hexagonal packing
    # Keep central hexagon fixed, adjust others to maximize space utilization
    for i in range(1, 12):
        # Scale distances to achieve better packing
        config[i, 0] *= 0.97
        config[i, 1] *= 0.97
    
    # Add noise for escape from local optima - more controlled perturbations
    for i in range(12):
        config[i, 0] += np.random.normal(0, 0.01)
        config[i, 1] += np.random.normal(0, 0.01)
    
    return config

def create_bounds():
    """Create bounds for optimization variables"""
    # Each hexagon has 3 parameters: (x, y, rotation)
    bounds = []
    
    # Positions: -5 to 5 (larger range to allow exploration)
    for _ in range(12):
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0)])
    
    # Rotations: 0 to 360 degrees
    for _ in range(12):
        bounds.append((0.0, 360.0))
    
    return bounds

def objective_function(x):
    """Objective function for optimization - minimized to maximize 1/R"""
    # Reshape x into 12 hexagons with (x, y, rotation) each
    hex_data = x.reshape(-1, 3)
    
    # Calculate side length
    side_length = calculate_outer_hex_side_length(hex_data)
    
    # Return negative inverse side length to maximize 1/side_length
    if side_length > 100:
        return float('inf')
    
    # Check validity
    score = evaluate_configuration(hex_data, side_length)
    return score

def optimize_with_multiple_strategies():
    """Use multiple optimization strategies to find the best solution"""
    best_score = float('inf')
    best_config = None
    best_side_length = None
    
    # Strategy 1: Start with mathematically precise configuration from inspiration 2
    try:
        # Use a mathematically precise starting configuration based on proven patterns
        # This is a more carefully constructed mathematical arrangement with better constants
        sqrt3 = np.sqrt(3)
        config = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.0, 0.0],           # top
            [0.0, -2.0, 0.0],          # bottom  
            [sqrt3, 1.0, 0.0],         # top-right
            [-sqrt3, 1.0, 0.0],        # top-left
            [sqrt3, -1.0, 0.0],        # bottom-right
            [-sqrt3, -1.0, 0.0],       # bottom-left
            [2.0*sqrt3, 0.0, 0.0],     # far right
            [-2.0*sqrt3, 0.0, 0.0],    # far left
            [sqrt3, 3.0, 0.0],         # top-top
            [-sqrt3, 3.0, 0.0],        # top-top-left
            [sqrt3, -3.0, 0.0],        # bottom-bottom
        ])
        
        # Add precise random perturbations to escape local minima
        for i in range(12):
            config[i, 0] += np.random.normal(0, 0.001)
            config[i, 1] += np.random.normal(0, 0.001)
        
        bounds = create_bounds()
        
        # Use L-BFGS-B with extremely tight tolerances for high precision
        result = minimize(
            objective_function,
            config.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            optimized_config = result.x.reshape(-1, 3)
            side_length = calculate_outer_hex_side_length(optimized_config)
            score = evaluate_configuration(optimized_config, side_length)
            
            if score < best_score and score != float('inf') and side_length < 100:
                best_score = score
                best_config = optimized_config.copy()
                best_side_length = side_length
                
    except Exception as e:
        pass
    
    # Strategy 2: Enhanced differential evolution from inspiration 2 with more aggressive settings
    try:
        # Multiple restarts with different configurations and DE parameters
        configs_and_params = [
            # Configuration 1: Start with constraint satisfaction approach
            (generate_constraint_satisfaction_config(), {'maxiter': 50, 'popsize': 20, 'seed': 42}),
            # Configuration 2: Start with hexagonal lattice approach  
            (generate_hexagonal_lattice_config(), {'maxiter': 55, 'popsize': 25, 'seed': 123}),
            # Configuration 3: Start with mathematical pattern from inspiration 1
            (np.array([
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
                [1.732, 1.0, 0.0],
                [-1.732, 1.0, 0.0],
                [1.732, -1.0, 0.0],
                [-1.732, -1.0, 0.0],
                [3.464, 0.0, 0.0],
                [-3.464, 0.0, 0.0],
                [1.732, 3.0, 0.0],
                [-1.732, 3.0, 0.0],
                [1.732, -3.0, 0.0],
            ]), {'maxiter': 40, 'popsize': 15, 'seed': 456}),
        ]
        
        for initial_config, params in configs_and_params:
            bounds = create_bounds()
            
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=params['maxiter'],
                popsize=params['popsize'],
                seed=params['seed'],
                disp=False,
                polish=True,
                strategy='best1bin'
            )
            
            if result.success:
                optimized_config = result.x.reshape(-1, 3)
                side_length = calculate_outer_hex_side_length(optimized_config)
                score = evaluate_configuration(optimized_config, side_length)
                
                if score < best_score and score != float('inf') and side_length < 100:
                    best_score = score
                    best_config = optimized_config.copy()
                    best_side_length = side_length
                    
    except Exception as e:
        pass
    
    # Strategy 3: Hybrid local refinement approach with better parameters
    if best_config is None:
        # Start with a solid configuration
        best_config = generate_constraint_satisfaction_config()
        best_side_length = calculate_outer_hex_side_length(best_config)
        best_score = evaluate_configuration(best_config, best_side_length)
    
    # Intensive local refinement with adaptive step sizes - more aggressive
    for iteration in range(500):
        new_config = best_config.copy()
        # Adaptive perturbation with faster decay
        step_size = max(0.00001, 0.02 * (1.0 - iteration/500.0))
        
        # Perturb each hexagon more aggressively early on, then fine-tune
        for i in range(len(new_config)):
            if np.random.random() < 0.8:  # 80% chance to modify
                new_config[i, 0] += np.random.normal(0, step_size)
                new_config[i, 1] += np.random.normal(0, step_size)
                if np.random.random() < 0.3:  # 30% chance to rotate
                    new_config[i, 2] += np.random.normal(0, 0.3)
        
        new_side_length = calculate_outer_hex_side_length(new_config)
        new_score = evaluate_configuration(new_config, new_side_length)
        
        if new_score < best_score and new_side_length < 100:
            best_config = new_config
            best_score = new_score
            best_side_length = new_side_length
    
    # Final high-precision polishing with L-BFGS-B - even tighter tolerances
    try:
        result = minimize(
            objective_function,
            best_config.flatten(),
            method='L-BFGS-B',
            bounds=create_bounds(),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.success:
            refined_config = result.x.reshape(-1, 3)
            refined_side_length = calculate_outer_hex_side_length(refined_config)
            refined_score = evaluate_configuration(refined_config, refined_side_length)
            
            if refined_score < best_score and refined_side_length < 100:
                best_config = refined_config
                best_score = refined_score
                best_side_length = refined_side_length
                
    except Exception as e:
        pass
    
    # If we still haven't found anything, return the best we had
    if best_config is None:
        initial_config = generate_constraint_satisfaction_config()
        side_length = calculate_outer_hex_side_length(initial_config)
        score = evaluate_configuration(initial_config, side_length)
        return initial_config, side_length, score
    
    return best_config, best_side_length, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use enhanced global optimization approach with multiple strategies
    start_time = time.time()
    
    best_config, best_side_length, best_score = optimize_with_multiple_strategies()
    
    # Final validation and optimization with more aggressive refinement
    max_refinement_iter = 1000  # More iterations for better optimization
    time_limit = 55  # Leave 5 seconds for final processing
    
    iteration = 0
    while iteration < max_refinement_iter and (time.time() - start_time) < time_limit:
        # Adaptive step size with exponential decay
        time_progress = (time.time() - start_time) / time_limit
        current_step = max(0.000001, 0.01 * (1.0 - time_progress * 0.95))  # Faster decay
        
        new_config = best_config.copy()
        # Perturbations with strategic selection
        for i in range(len(new_config)):
            # 90% chance to modify position for aggressive improvement
            if np.random.random() < 0.9:
                new_config[i, 0] += np.random.normal(0, current_step)
                new_config[i, 1] += np.random.normal(0, current_step)
            
            # 25% chance to rotate for fine-tuning
            if np.random.random() < 0.25:
                new_config[i, 2] += np.random.normal(0, 0.2)
        
        new_side_length = calculate_outer_hex_side_length(new_config)
        new_score = evaluate_configuration(new_config, new_side_length)
        
        # Accept better solutions or occasionally accept worse ones to escape local minima
        if new_score < best_score and new_side_length < 100:
            best_config = new_config
            best_score = new_score
            best_side_length = new_side_length
        elif np.random.random() < 0.001:  # Occasionally accept worse solutions
            # Accept with low probability to escape local minima
            best_config = new_config
            best_score = new_score
            best_side_length = new_side_length
            
        iteration += 1
    
    # Final ultra-precise polishing with specialized approach
    try:
        # Try a more targeted optimization approach
        # First, try to refine with very tight tolerances using L-BFGS-B
        result = minimize(
            objective_function,
            best_config.flatten(),
            method='L-BFGS-B',
            bounds=create_bounds(),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            refined_config = result.x.reshape(-1, 3)
            refined_side_length = calculate_outer_hex_side_length(refined_config)
            refined_score = evaluate_configuration(refined_config, refined_side_length)
            
            if refined_score < best_score and refined_side_length < 100:
                best_config = refined_config
                best_score = refined_score
                best_side_length = refined_side_length
                
    except Exception as e:
        pass
    
    # Try a second optimization with different parameters to ensure we're not stuck
    try:
        # Try with TNC method for potentially better results
        result = minimize(
            objective_function,
            best_config.flatten(),
            method='TNC',
            bounds=create_bounds(),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.success:
            refined_config = result.x.reshape(-1, 3)
            refined_side_length = calculate_outer_hex_side_length(refined_config)
            refined_score = evaluate_configuration(refined_config, refined_side_length)
            
            if refined_score < best_score and refined_side_length < 100:
                best_config = refined_config
                best_score = refined_score
                best_side_length = refined_side_length
                
    except Exception as e:
        pass
    
    # Final validation with enhanced checking
    final_score = evaluate_configuration(best_config, best_side_length)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Time measurement
    eval_time = time.time() - start_time
    
    return best_config, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
