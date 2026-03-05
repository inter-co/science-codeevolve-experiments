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
    
    # Early exit for invalid side length
    if outer_side_length > 100:
        return float('inf')
    
    # Create outer hexagon vertices
    outer_center = (0, 0)
    outer_rotation = 0
    outer_hex_vertices = get_hexagon_vertices(outer_center, outer_side_length, outer_rotation)
    
    # Check containment with early exit and optimization
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, 1, rotation)
        
        # Quick centroid check first for fast rejection
        centroid = np.mean(hex_vertices, axis=0)
        if not Point(centroid).within(Polygon(outer_hex_vertices)):
            return float('inf')  # Not contained, skip full vertex check
            
        # Full vertex containment check for safety
        for vertex in hex_vertices:
            if not Point(vertex).within(Polygon(outer_hex_vertices)):
                return float('inf')  # Invalid configuration
    
    # Check overlaps with more efficient approach - early termination
    # Use spatial indexing for better performance on many comparisons
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
            rotation1 = inner_hex_data[i, 2]
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            
            hex1_vertices = get_hexagon_vertices(center1, 1, rotation1)
            hex2_vertices = get_hexagon_vertices(center2, 1, rotation2)
            
            # Fast distance-based check first - if too far apart, no overlap
            c1 = np.mean(hex1_vertices, axis=0)
            c2 = np.mean(hex2_vertices, axis=0)
            dist = np.linalg.norm(c1 - c2)
            
            # If distance is greater than sum of radii (2.0), no overlap
            if dist >= 2.0:
                continue
                
            # Use Shapely for precise polygon intersection
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
    
    # Strategy 1: Very precise mathematical configuration with extreme precision local optimization
    try:
        # Create a configuration based on known optimal mathematical patterns with highest precision
        # Using the exact mathematical constants from hexagonal packing theory
        config = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.0, 0.0],           # top
            [0.0, -2.0, 0.0],          # bottom  
            [1.7320508075688772, 1.0, 0.0],   # top-right (sqrt(3), 1)
            [-1.7320508075688772, 1.0, 0.0],  # top-left
            [1.7320508075688772, -1.0, 0.0],  # bottom-right
            [-1.7320508075688772, -1.0, 0.0], # bottom-left
            [3.4641016151377544, 0.0, 0.0],   # far right (2*sqrt(3), 0)
            [-3.4641016151377544, 0.0, 0.0],  # far left
            [1.7320508075688772, 3.0, 0.0],   # top-top
            [-1.7320508075688772, 3.0, 0.0],  # top-top-left
            [1.7320508075688772, -3.0, 0.0],  # bottom-bottom
        ], dtype=np.float64)
        
        # Add extremely small random perturbations to escape local minima
        for i in range(12):
            config[i, 0] += np.random.normal(0, 0.0000001)
            config[i, 1] += np.random.normal(0, 0.0000001)
        
        bounds = create_bounds()
        
        # Use L-BFGS-B with ultra-tight tolerances for maximum precision
        result = minimize(
            objective_function,
            config.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-18, 'gtol': 1e-18}
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
    
    # Strategy 2: Enhanced global search with maximum diversity and aggressive DE parameters
    try:
        # Use maximum diversity in starting configurations with even more aggressive settings
        starting_configs = []
        
        # Mathematical pattern (from inspiration 3) - very precise
        starting_configs.append(np.array([
            [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
            [1.7320508075688772, 1.0, 0.0], [-1.7320508075688772, 1.0, 0.0],
            [1.7320508075688772, -1.0, 0.0], [-1.7320508075688772, -1.0, 0.0],
            [3.4641016151377544, 0.0, 0.0], [-3.4641016151377544, 0.0, 0.0],
            [1.7320508075688772, 3.0, 0.0], [-1.7320508075688772, 3.0, 0.0],
            [1.7320508075688772, -3.0, 0.0]
        ]))
        
        # Constraint satisfaction pattern
        starting_configs.append(generate_constraint_satisfaction_config())
        
        # Hexagonal lattice pattern
        starting_configs.append(generate_hexagonal_lattice_config())
        
        # Randomized mathematical pattern
        random_config = starting_configs[0].copy()
        for i in range(len(random_config)):
            random_config[i, 0] += np.random.normal(0, 0.03)
            random_config[i, 1] += np.random.normal(0, 0.03)
        starting_configs.append(random_config)
        
        # Perturbed mathematical pattern
        perturbed_config = starting_configs[0].copy()
        for i in range(len(perturbed_config)):
            perturbed_config[i, 0] += np.random.normal(0, 0.015)
            perturbed_config[i, 1] += np.random.normal(0, 0.015)
        starting_configs.append(perturbed_config)
        
        # Even more randomized version
        random_config2 = starting_configs[0].copy()
        for i in range(len(random_config2)):
            random_config2[i, 0] += np.random.normal(0, 0.04)
            random_config2[i, 1] += np.random.normal(0, 0.04)
        starting_configs.append(random_config2)
        
        # Run differential evolution with aggressive parameters
        de_configs = [
            {'maxiter': 120, 'popsize': 50, 'seed': 42, 'strategy': 'best1bin', 'mutation': (0.95, 1.0)},
            {'maxiter': 100, 'popsize': 55, 'seed': 123, 'strategy': 'rand1bin', 'mutation': (0.9, 1.0)},
            {'maxiter': 90, 'popsize': 45, 'seed': 456, 'strategy': 'best2bin', 'mutation': (0.85, 1.0)},
        ]
        
        for i, (initial_config, params) in enumerate(zip(starting_configs, de_configs)):
            bounds = create_bounds()
            
            # Use DE with even more aggressive parameters and higher precision
            result = differential_evolution(
                objective_function,
                bounds,
                x0=initial_config.flatten(),
                maxiter=params['maxiter'],
                popsize=params['popsize'],
                seed=params['seed'],
                strategy=params['strategy'],
                mutation=params['mutation'],
                recombination=0.98,
                disp=False,
                polish=True,
                tol=1e-16
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
    
    # Strategy 3: Advanced hybrid refinement with intelligent perturbations and early stopping
    if best_config is None:
        # Start with a solid configuration
        best_config = generate_constraint_satisfaction_config()
        best_side_length = calculate_outer_hex_side_length(best_config)
        best_score = evaluate_configuration(best_config, best_side_length)
    
    # More sophisticated refinement with adaptive step sizes and better early stopping
    iteration = 0
    max_iterations = 2500
    last_improvement = 0
    improvement_count = 0
    patience_counter = 0
    
    # Track the best score seen so far for early termination
    best_seen_score = best_score
    
    while iteration < max_iterations and patience_counter < 800:
        new_config = best_config.copy()
        
        # Adaptive step size with faster decay and better scaling
        time_factor = iteration / max_iterations
        base_step_size = 0.03 * (1.0 - time_factor * 0.95)
        step_size = max(0.000001, base_step_size)
        
        # Even more aggressive perturbation strategy
        for i in range(len(new_config)):
            # 98% chance to modify position (extremely aggressive)
            if np.random.random() < 0.98:
                new_config[i, 0] += np.random.normal(0, step_size * 0.8)
                new_config[i, 1] += np.random.normal(0, step_size * 0.8)
            # 50% chance to rotate (very frequent)
            if np.random.random() < 0.50:
                new_config[i, 2] += np.random.normal(0, 0.25)
        
        new_side_length = calculate_outer_hex_side_length(new_config)
        new_score = evaluate_configuration(new_config, new_side_length)
        
        if new_score < best_score and new_side_length < 100:
            best_config = new_config
            best_score = new_score
            best_side_length = new_side_length
            last_improvement = iteration
            improvement_count += 1
            patience_counter = 0
            best_seen_score = min(best_seen_score, new_score)
        else:
            patience_counter += 1
        
        # Early termination if we're not improving significantly
        if iteration > 500 and patience_counter > 300:
            # Check if we've been stagnating for a while
            if abs(best_seen_score - best_score) < 1e-12:
                break
        
        iteration += 1
    
    # Strategy 4: Final comprehensive polishing with multiple optimization methods
    try:
        # Try multiple optimization methods with progressively tighter tolerances
        polish_configs = []
        
        # Original best config
        polish_configs.append(best_config.copy())
        
        # Slightly perturbed version with more aggressive noise
        perturbed_config = best_config.copy()
        for i in range(len(perturbed_config)):
            perturbed_config[i, 0] += np.random.normal(0, 0.001)
            perturbed_config[i, 1] += np.random.normal(0, 0.001)
        polish_configs.append(perturbed_config)
        
        # Another perturbed version with even more aggressive noise
        perturbed_config2 = best_config.copy()
        for i in range(len(perturbed_config2)):
            perturbed_config2[i, 0] += np.random.normal(0, 0.002)
            perturbed_config2[i, 1] += np.random.normal(0, 0.002)
        polish_configs.append(perturbed_config2)
        
        # Yet another version with different noise pattern
        perturbed_config3 = best_config.copy()
        for i in range(len(perturbed_config3)):
            perturbed_config3[i, 0] += np.random.normal(0, 0.0005)
            perturbed_config3[i, 1] += np.random.normal(0, 0.0005)
        polish_configs.append(perturbed_config3)
        
        # Refinement with different methods and increasing precision
        methods_to_try = [
            ('L-BFGS-B', {'maxiter': 800, 'ftol': 1e-17, 'gtol': 1e-17}),
            ('TNC', {'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-16}),
            ('SLSQP', {'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-16}),
        ]
        
        for i, (method_name, method_options) in enumerate(methods_to_try):
            try:
                # Use a slightly perturbed version for diversity
                current_config = polish_configs[min(i, len(polish_configs)-1)]
                result = minimize(
                    objective_function,
                    current_config.flatten(),
                    method=method_name,
                    bounds=create_bounds(),
                    options=method_options
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
                continue
                
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
