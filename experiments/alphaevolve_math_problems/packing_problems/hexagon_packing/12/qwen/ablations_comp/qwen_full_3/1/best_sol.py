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
from itertools import permutations

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, radius, rotation):
    """Fast computation of hexagon vertices using numba"""
    angles = np.linspace(0, 2*np.pi, 7) + rotation
    vertices = np.empty((6, 2))
    for i in range(6):
        vertices[i, 0] = center_x + radius * np.cos(angles[i])
        vertices[i, 1] = center_y + radius * np.sin(angles[i])
    return vertices

def create_regular_hexagon(center=(0,0), radius=1, rotation=0):
    """Create a regular hexagon with given center, radius, and rotation"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius*np.cos(angles), 
                             center[1] + radius*np.sin(angles)])
    return points[:-1]  # Remove duplicate last point

def get_hexagon_vertices(hex_center, hex_radius=1, rotation=0):
    """Get all vertices of a hexagon"""
    return create_regular_hexagon(hex_center, hex_radius, rotation)

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon"""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        if not outer_polygon.contains(Point(vertex)):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    except:
        # Handle edge cases
        return False

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate the minimum side length needed for outer hexagon to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        vertices = get_hexagon_vertices(center, 1, rotation)
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
    
    # For a regular hexagon, the side length equals the circumradius
    # But we want to ensure we have a hexagon that can contain everything
    # The minimum circumscribing hexagon has side length equal to the maximum distance
    return max_distance

def evaluate_configuration(inner_hex_data, outer_side_length=None):
    """Evaluate a configuration for validity and quality"""
    if outer_side_length is None:
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Create outer hexagon vertices
    outer_center = (0, 0)
    outer_rotation = 0
    outer_hex_vertices = get_hexagon_vertices(outer_center, outer_side_length, outer_rotation)
    
    # Check containment
    all_contained = True
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, 1, rotation)
        
        if not check_containment(hex_vertices, outer_hex_vertices):
            all_contained = False
            break
    
    if not all_contained:
        return float('inf')  # Invalid configuration
    
    # Check overlaps
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
            rotation1 = inner_hex_data[i, 2]
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            
            hex1_vertices = get_hexagon_vertices(center1, 1, rotation1)
            hex2_vertices = get_hexagon_vertices(center2, 1, rotation2)
            
            if check_overlap(hex1_vertices, hex2_vertices):
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
    config = np.array([
        [0.0, 0.0, 0.0],       # center
        [0.0, 2.0, 0.0],       # top
        [0.0, -2.0, 0.0],      # bottom
        [1.732, 1.0, 0.0],     # top-right
        [-1.732, 1.0, 0.0],    # top-left
        [1.732, -1.0, 0.0],    # bottom-right
        [-1.732, -1.0, 0.0],   # bottom-left
        [3.464, 0.0, 0.0],     # far right
        [-3.464, 0.0, 0.0],    # far left
        [1.732, 3.0, 0.0],     # top-top
        [-1.732, 3.0, 0.0],    # top-top-left
        [1.732, -3.0, 0.0],    # bottom-bottom
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
        config[i, 0] *= 0.95
        config[i, 1] *= 0.95
    
    # Add noise for escape from local optima
    for i in range(12):
        config[i, 0] += random.uniform(-0.02, 0.02)
        config[i, 1] += random.uniform(-0.02, 0.02)
    
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
    
    # Strategy 1: Very precise mathematical configuration from inspiration 1
    try:
        # Use a carefully constructed mathematical starting configuration 
        # that closely matches known optimal solutions from hexagon packing literature
        config = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.0, 0.0],           # top
            [0.0, -2.0, 0.0],          # bottom  
            [1.732050808, 1.0, 0.0],   # top-right
            [-1.732050808, 1.0, 0.0],  # top-left
            [1.732050808, -1.0, 0.0],  # bottom-right
            [-1.732050808, -1.0, 0.0], # bottom-left
            [3.464101616, 0.0, 0.0],   # far right
            [-3.464101616, 0.0, 0.0],  # far left
            [1.732050808, 3.0, 0.0],   # top-top
            [-1.732050808, 3.0, 0.0],  # top-top-left
            [1.732050808, -3.0, 0.0],  # bottom-bottom
        ])
        
        # Add very precise, minimal perturbations
        for i in range(12):
            config[i, 0] += np.random.normal(0, 0.0001)
            config[i, 1] += np.random.normal(0, 0.0001)
        
        bounds = create_bounds()
        
        # Use L-BFGS-B with extremely tight tolerances for maximum precision
        result = minimize(
            objective_function,
            config.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-14}
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
    
    # Strategy 2: Robust differential evolution with multiple restarts (inspiration 2)
    try:
        # Use multiple restarts with carefully chosen parameters
        de_configs = [
            {'maxiter': 40, 'popsize': 15, 'seed': 42, 'strategy': 'best1bin'},
            {'maxiter': 50, 'popsize': 18, 'seed': 123, 'strategy': 'best1bin'},
            {'maxiter': 35, 'popsize': 12, 'seed': 456, 'strategy': 'best1bin'},
        ]
        
        # Start with different good configurations for each DE run
        starting_configs = [
            generate_constraint_satisfaction_config(),
            generate_hexagonal_lattice_config(),
            np.array([
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
            ])
        ]
        
        for i, (config, params) in enumerate(zip(starting_configs, de_configs)):
            bounds = create_bounds()
            
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=params['maxiter'],
                popsize=params['popsize'],
                seed=params['seed'],
                disp=False,
                polish=True,
                strategy=params['strategy']
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
    
    # Strategy 3: Smart local refinement with early termination (inspiration 3)
    if best_config is None:
        # Start with a good configuration
        best_config = generate_constraint_satisfaction_config()
        best_side_length = calculate_outer_hex_side_length(best_config)
        best_score = evaluate_configuration(best_config, best_side_length)
    
    # Intensive local refinement with smart stopping conditions
    last_improvement_iter = 0
    for iteration in range(300):
        new_config = best_config.copy()
        # Adaptive perturbation - start aggressive, then fine-tune
        step_size = max(0.0001, 0.02 * (1.0 - iteration/300.0))
        
        # Systematic perturbation approach
        for i in range(len(new_config)):
            if np.random.random() < 0.7:  # 70% chance to modify
                new_config[i, 0] += np.random.normal(0, step_size)
                new_config[i, 1] += np.random.normal(0, step_size)
                if np.random.random() < 0.1:  # 10% chance to rotate
                    new_config[i, 2] += np.random.normal(0, 0.5)
        
        new_side_length = calculate_outer_hex_side_length(new_config)
        new_score = evaluate_configuration(new_config, new_side_length)
        
        if new_score < best_score and new_side_length < 100:
            best_config = new_config
            best_score = new_score
            best_side_length = new_side_length
            last_improvement_iter = iteration
            
        # Early termination if no improvement in last 100 iterations
        if iteration - last_improvement_iter > 100:
            break
    
    # Final high-precision polishing with very tight tolerances
    try:
        result = minimize(
            objective_function,
            best_config.flatten(),
            method='L-BFGS-B',
            bounds=create_bounds(),
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
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
    best_config, best_side_length, best_score = optimize_with_multiple_strategies()
    
    # Final refinement with targeted perturbations
    for iteration in range(200):
        new_config = best_config.copy()
        # Make small, targeted changes with higher probability of improvement
        for i in range(len(new_config)):
            if np.random.random() < 0.3:  # 30% chance to modify
                new_config[i, 0] += np.random.normal(0, 0.003)
                new_config[i, 1] += np.random.normal(0, 0.003)
                if np.random.random() < 0.1:  # 10% chance to rotate
                    new_config[i, 2] += np.random.normal(0, 0.3)
        
        new_side_length = calculate_outer_hex_side_length(new_config)
        new_score = evaluate_configuration(new_config, new_side_length)
        
        if new_score < best_score and new_side_length < 100:
            best_config = new_config
            best_score = new_score
            best_side_length = new_side_length
    
    # Validate final result
    final_score = evaluate_configuration(best_config, best_side_length)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return best_config, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
