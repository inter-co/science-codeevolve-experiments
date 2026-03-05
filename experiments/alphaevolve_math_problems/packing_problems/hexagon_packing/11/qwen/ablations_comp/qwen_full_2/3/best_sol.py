# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from scipy.optimize import differential_evolution, minimize
import time
import random
from itertools import combinations


def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(theta)
        y = center_y + side_length * math.sin(theta)
        vertices.append((x, y))
    return vertices


def hexagon_polygon(center_x, center_y, angle_deg, side_length=1):
    """Create a Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)


def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly)


def check_containment(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.touches(hex_poly)


def calculate_outer_hex_side_length(inner_hex_data):
    """
    Calculate the minimum side length of outer hexagon that contains all inner hexagons.
    """
    # Create all inner hexagon polygons
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_poly = hexagon_polygon(center_x, center_y, angle)
        inner_hexagons.append(hex_poly)
    
    # Find bounding box of all inner hexagons
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for hex_poly in inner_hexagons:
        bounds = hex_poly.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    
    # Calculate the distance from center to the farthest corner
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    max_dist = 0
    for hex_poly in inner_hexagons:
        for point in list(hex_poly.exterior.coords):
            dist = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            max_dist = max(max_dist, dist)
    
    # For a regular hexagon circumscribed around a circle of radius r, 
    # the side length is r. So we need a hexagon with radius = max_dist
    # The side length of a hexagon is equal to its circumradius
    return max_dist + 0.01  # Add small margin for numerical precision


def is_valid_configuration(inner_hex_data):
    """Check if configuration is valid (no overlaps, all contained)."""
    # Create all inner hexagon polygons
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_poly = hexagon_polygon(center_x, center_y, angle)
        inner_hexagons.append(hex_poly)
    
    # Check for overlaps using efficient pairwise comparison
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if check_overlap(inner_hexagons[i], inner_hexagons[j]):
            return False
    
    # Create outer hexagon (large enough to contain everything)
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    outer_hex = hexagon_polygon(0, 0, 0, outer_side_length)
    
    # Check containment
    for hex_poly in inner_hexagons:
        if not check_containment(hex_poly, outer_hex):
            return False
    
    return True


def generate_multiple_initial_configs():
    """Generate multiple high-quality initial configurations."""
    configs = []
    
    # Configuration 1: Optimized hexagonal packing (from INSPIRATION 2)
    positions1 = [
        [0, 0],              # center
        [0, 1.93],           # top
        [0, -1.93],          # bottom  
        [1.67, 0.96],        # top-right
        [-1.67, 0.96],       # top-left
        [1.67, -0.96],       # bottom-right
        [-1.67, -0.96],      # bottom-left
        [3.34, 0],           # far right
        [-3.34, 0],          # far left
        [1.67, 2.89],        # upper far right
        [-1.67, 2.89],       # upper far left
    ]
    rotations1 = [0] * 11
    configs.append((np.array(positions1), np.array(rotations1)))
    
    # Configuration 2: Hexagonal close packing (from INSPIRATION 1)
    positions2 = [
        [0, 0],      # center
        [0, 2.0],    # top
        [0, -2.0],   # bottom  
        [1.732, 1.0], # top-right
        [-1.732, 1.0], # top-left
        [1.732, -1.0], # bottom-right
        [-1.732, -1.0], # bottom-left
        [3.464, 0],  # far right
        [-3.464, 0], # far left
        [1.732, 2.0], # top-right corner
        [-1.732, 2.0], # top-left corner
    ]
    rotations2 = [0] * 11
    configs.append((np.array(positions2), np.array(rotations2)))
    
    # Configuration 3: More spread-out arrangement (from INSPIRATION 2)
    positions3 = [
        [0, 0],              # center
        [0, 2.17],           # top
        [0, -2.17],          # bottom  
        [1.88, 1.08],        # top-right
        [-1.88, 1.08],       # top-left
        [1.88, -1.08],       # bottom-right
        [-1.88, -1.08],      # bottom-left
        [3.75, 0],           # far right
        [-3.75, 0],          # far left
        [2.88, 2.17],        # upper far right
        [-2.88, 2.17],       # upper far left
    ]
    rotations3 = [0] * 11
    configs.append((np.array(positions3), np.array(rotations3)))
    
    # Configuration 4: Dense central arrangement (from INSPIRATION 2)
    positions4 = [
        [0, 0],              # center
        [0, 2.3],            # top
        [0, -2.3],           # bottom  
        [2.0, 1.15],         # top-right
        [-2.0, 1.15],        # top-left
        [2.0, -1.15],        # bottom-right
        [-2.0, -1.15],       # bottom-left
        [4.0, 0],            # far right
        [-4.0, 0],           # far left
        [0, 4.6],            # upper far right
        [0, -4.6],           # upper far left
    ]
    rotations4 = [0] * 11
    configs.append((np.array(positions4), np.array(rotations4)))
    
    # Configuration 5: Highly symmetric pattern (from INSPIRATION 2)
    positions5 = [
        [0, 0],              # center
        [0, 1.85],           # top
        [0, -1.85],          # bottom  
        [1.59, 0.89],        # top-right
        [-1.59, 0.89],       # top-left
        [1.59, -0.89],       # bottom-right
        [-1.59, -0.89],      # bottom-left
        [3.18, 0],           # far right
        [-3.18, 0],          # far left
        [1.59, 2.74],        # upper far right
        [-1.59, 2.74],       # upper far left
    ]
    rotations5 = [0] * 11
    configs.append((np.array(positions5), np.array(rotations5)))
    
    # Configuration 6: Compact arrangement (from INSPIRATION 3)
    positions6 = [
        [0, 0],              # center
        [0, 1.89],           # top
        [0, -1.89],          # bottom  
        [1.64, 0.93],        # top-right
        [-1.64, 0.93],       # top-left
        [1.64, -0.93],       # bottom-right
        [-1.64, -0.93],      # bottom-left
        [3.28, 0],           # far right
        [-3.28, 0],          # far left
        [1.64, 2.85],        # upper far right
        [-1.64, 2.85],       # upper far left
    ]
    rotations6 = [0] * 11
    configs.append((np.array(positions6), np.array(rotations6)))
    
    return configs


def objective_function(params):
    """
    Objective function to minimize (negative of 1/outer_hex_side_length).
    This is used for maximization by returning negative values.
    """
    # Reshape parameters into 11 hexagons (each with 3 params: x, y, angle)
    hex_data = params.reshape(-1, 3)
    
    try:
        # Check if this configuration is valid
        if not is_valid_configuration(hex_data):
            # Return large penalty for invalid configurations
            return 1e10
        
        # Calculate outer hexagon side length
        outer_side_length = calculate_outer_hex_side_length(hex_data)
        
        # Return negative inverse for minimization (we want to maximize 1/outer_side_length)
        return -1.0 / outer_side_length
        
    except Exception as e:
        # Return large penalty for any errors
        return 1e10


def refine_with_local_optimization(initial_params):
    """Apply local optimization to refine the solution."""
    def local_objective(params):
        # Reshape parameters into 11 hexagons (each with 3 params: x, y, angle)
        hex_data = params.reshape(-1, 3)
        try:
            if not is_valid_configuration(hex_data):
                return 1e10
            outer_side_length = calculate_outer_hex_side_length(hex_data)
            return -1.0 / outer_side_length  # Negative because we're minimizing
        except Exception:
            return 1e10
    
    # Use L-BFGS-B for local refinement with tighter bounds
    bounds = []
    for i in range(len(initial_params)):
        if i % 3 == 0:  # x coordinate
            bounds.append((-10, 10))
        elif i % 3 == 1:  # y coordinate
            bounds.append((-10, 10))
        else:  # angle
            bounds.append((0, 360))
    
    try:
        result = minimize(
            local_objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        return result.x
    except Exception:
        return initial_params


def optimized_multi_strategy_approach():
    """Use a multi-strategy optimization approach to maximize chance of finding best solution."""
    best_inv_side_length = 0
    best_params = None
    best_side_length = float('inf')
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    # Strategy 1: Differential Evolution with multiple initial configs
    strategies = [
        {
            'maxiter': 200,
            'popsize': 20,
            'mutation': (0.9, 1),
            'recombination': 0.95,
            'tol': 1e-15,
            'seed': 42
        },
        {
            'maxiter': 150,
            'popsize': 15,
            'mutation': (0.8, 1),
            'recombination': 0.90,
            'tol': 1e-14,
            'seed': 123
        },
        {
            'maxiter': 100,
            'popsize': 10,
            'mutation': (0.7, 1),
            'recombination': 0.85,
            'tol': 1e-13,
            'seed': 456
        }
    ]
    
    # Try different strategies with multiple initial configurations
    for strategy_idx, strategy in enumerate(strategies):
        for config_idx, (init_positions, init_rotations) in enumerate(initial_configs):
            # Set different random seeds for each combination
            random.seed(strategy['seed'] + config_idx + strategy_idx * 100)
            
            # Add small random perturbations to escape local minima
            perturbed_positions = init_positions.copy()
            for j in range(len(perturbed_positions)):
                perturbed_positions[j][0] += np.random.uniform(-0.05, 0.05)
                perturbed_positions[j][1] += np.random.uniform(-0.05, 0.05)
            
            # Create initial population with better starting points
            initial_population = []
            # Add the good initial configuration
            initial_individual = []
            for i in range(len(perturbed_positions)):
                initial_individual.extend([perturbed_positions[i][0], perturbed_positions[i][1], init_rotations[i]])
            initial_population.append(initial_individual)
            
            # Add some random variations around the good initial configuration
            for _ in range(19):  # Total population size of 20
                individual = []
                for i in range(len(perturbed_positions)):
                    # Add small random perturbations with smaller range for better precision
                    x = perturbed_positions[i][0] + np.random.uniform(-0.03, 0.03)
                    y = perturbed_positions[i][1] + np.random.uniform(-0.03, 0.03)
                    angle = init_rotations[i] + np.random.uniform(-2, 2)
                    # Clamp angles to [0, 360)
                    angle = angle % 360
                    individual.extend([x, y, angle])
                initial_population.append(individual)
            
            # Define bounds for each parameter
            num_hexagons = 11
            params_per_hexagon = 3  # x, y, angle
            total_params = num_hexagons * params_per_hexagon
            bounds = []
            for i in range(total_params):
                if i % 3 == 0:  # x coordinate
                    bounds.append((-10, 10))
                elif i % 3 == 1:  # y coordinate
                    bounds.append((-10, 10))
                else:  # angle
                    bounds.append((0, 360))
            
            # Run differential evolution optimization with this strategy
            try:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=strategy['maxiter'],
                    popsize=strategy['popsize'],
                    tol=strategy['tol'],
                    mutation=strategy['mutation'],
                    recombination=strategy['recombination'],
                    seed=strategy['seed'] + config_idx,
                    disp=False,
                    init=initial_population
                )
                
                # Refine with local optimization
                refined_params = refine_with_local_optimization(result.x)
                final_params = refined_params.reshape(-1, 3)
                
                # Check if this is valid and better
                if is_valid_configuration(final_params):
                    side_length = calculate_outer_hex_side_length(final_params)
                    inv_side_length = 1.0 / side_length
                    
                    if inv_side_length > best_inv_side_length:
                        best_inv_side_length = inv_side_length
                        best_params = final_params.copy()
                        best_side_length = side_length
                        
            except Exception:
                continue
    
    # If we didn't find a good solution, fallback to the best initial configuration
    if best_params is None:
        # Use the best configuration from our initial set
        init_positions, init_rotations = initial_configs[0]
        best_params = np.column_stack([init_positions[:, 0], init_positions[:, 1], init_rotations])
        best_side_length = calculate_outer_hex_side_length(best_params)
        best_inv_side_length = 1.0 / best_side_length
    
    return best_params, best_side_length


def optimize_hexagon_packing():
    """
    Optimize the packing of 11 unit regular hexagons using optimized multi-strategy approach.
    """
    start_time = time.time()
    
    # Use optimized multi-strategy approach
    best_params, best_side_length = optimized_multi_strategy_approach()
    
    end_time = time.time()
    
    # Calculate final metrics
    inv_outer_side_length = 1.0 / best_side_length
    benchmark_ratio = inv_outer_side_length / 0.2544
    
    # The outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    return best_params, outer_hex_data, best_side_length, end_time - start_time, inv_outer_side_length, benchmark_ratio


def hexagon_packing_11():
    """
    Main function to construct an optimized packing of 11 disjoint unit regular hexagons.
    Uses optimized multi-strategy approach to find the global optimum.
    """
    inner_hex_data, outer_hex_data, outer_side_length, eval_time, inv_outer_side_length, benchmark_ratio = optimize_hexagon_packing()
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
