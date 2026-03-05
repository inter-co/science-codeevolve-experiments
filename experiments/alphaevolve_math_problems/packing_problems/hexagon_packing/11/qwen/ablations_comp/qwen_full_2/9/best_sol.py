# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import math
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
import itertools
import time

# Use more precise mathematical constants
SQRT_3 = np.sqrt(3)
SQRT_3_OVER_2 = SQRT_3 / 2.0
SQRT_3_HALF = SQRT_3 / 2.0


def hexagon_vertices(center_x, center_y, side_length=1, angle_deg=0):
    """Generate vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)


def contains_hexagon_shapely(hex_center, hex_angle, outer_hex_vertices):
    """Check if a hexagon is fully contained within outer hexagon using Shapely."""
    hex_vertices = hexagon_vertices(hex_center[0], hex_center[1], 1, hex_angle)
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    
    # Check if inner hexagon is completely inside outer hexagon
    return outer_polygon.contains(inner_polygon) or outer_polygon.covers(inner_polygon)


def compute_outer_hexagon_radius(inner_hex_centers, inner_hex_angles):
    """Compute minimum radius needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i, (center, angle) in enumerate(zip(inner_hex_centers, inner_hex_angles)):
        hex_vertices = hexagon_vertices(center[0], center[1], 1, angle)
        all_vertices.extend(hex_vertices)
    
    if len(all_vertices) == 0:
        return 1.0
    
    # Find center of all vertices (use median for better numerical stability with outliers)
    all_vertices = np.array(all_vertices)
    center_x = np.median(all_vertices[:, 0])
    center_y = np.median(all_vertices[:, 1])
    
    # Compute max distance from center to any vertex
    distances = np.sqrt((all_vertices[:, 0] - center_x)**2 + (all_vertices[:, 1] - center_y)**2)
    
    # Add small margin to ensure containment - use 1.005 instead of 1.01 for better precision
    return np.max(distances) * 1.005


def check_collision_shapely(hex1_center, hex1_angle, hex2_center, hex2_angle):
    """Check if two hexagons collide using Shapely for accurate collision detection."""
    hex1_vertices = hexagon_vertices(hex1_center[0], hex1_center[1], 1, hex1_angle)
    hex2_vertices = hexagon_vertices(hex2_center[0], hex2_center[1], 1, hex2_angle)
    
    polygon1 = Polygon(hex1_vertices)
    polygon2 = Polygon(hex2_vertices)
    
    # Check if polygons intersect
    return polygon1.intersects(polygon2)


def objective_function(params):
    """Objective function to minimize (negative of 1/outer_hex_side_length)."""
    # params: [x1, y1, angle1, x2, y2, angle2, ..., x11, y11, angle11]
    n = 11
    inner_positions = []
    inner_angles = []
    
    for i in range(n):
        inner_positions.append([params[3*i], params[3*i+1]])
        inner_angles.append(params[3*i+2])
    
    # Check collisions using shapely for accuracy
    for i, j in combinations(range(n), 2):
        if check_collision_shapely(inner_positions[i], inner_angles[i], 
                                 inner_positions[j], inner_angles[j]):
            return 1e35  # Extremely large penalty for collisions
    
    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(inner_positions, inner_angles)
    
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, outer_radius, 0)
    
    # Check containment of all inner hexagons using shapely
    for i in range(n):
        if not contains_hexagon_shapely(inner_positions[i], inner_angles[i], outer_vertices):
            return 1e35  # Extremely large penalty for containment violation
    
    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -1.0 / outer_radius if outer_radius > 0 else 1e35


def generate_multiple_initial_configs():
    """Generate multiple high-quality initial configurations based on mathematical research."""
    configs = []
    
    # Configuration 1: Most symmetric honeycomb pattern (from INSPIRATION 2)
    config1 = [
        [0, 0, 0],          # center
        [0, 2.0, 0],        # top
        [0, -2.0, 0],       # bottom  
        [1.732, 1.0, 0],    # top-right
        [-1.732, 1.0, 0],   # top-left
        [1.732, -1.0, 0],   # bottom-right
        [-1.732, -1.0, 0],  # bottom-left
        [3.464, 0, 0],      # far right
        [-3.464, 0, 0],     # far left
        [1.732, 3.0, 0],    # upper far right
        [-1.732, 3.0, 0],   # upper far left
    ]
    configs.append(config1)
    
    # Configuration 2: Tighter packing pattern (from INSPIRATION 2)
    config2 = [
        [0, 0, 0],          # center
        [0, 1.93, 0],       # top
        [0, -1.93, 0],      # bottom  
        [1.67, 0.96, 0],    # top-right
        [-1.67, 0.96, 0],   # top-left
        [1.67, -0.96, 0],   # bottom-right
        [-1.67, -0.96, 0],  # bottom-left
        [3.34, 0, 0],       # far right
        [-3.34, 0, 0],      # far left
        [1.67, 2.89, 0],    # upper far right
        [-1.67, 2.89, 0],   # upper far left
    ]
    configs.append(config2)
    
    # Configuration 3: More spread-out arrangement (from INSPIRATION 3)
    config3 = [
        [0, 0, 0],          # center
        [0, 2.1, 0],        # top
        [0, -2.1, 0],       # bottom  
        [1.8, 1.05, 0],     # top-right
        [-1.8, 1.05, 0],    # top-left
        [1.8, -1.05, 0],    # bottom-right
        [-1.8, -1.05, 0],   # bottom-left
        [3.6, 0, 0],        # far right
        [-3.6, 0, 0],       # far left
        [1.8, 3.15, 0],     # upper far right
        [-1.8, 3.15, 0],    # upper far left
    ]
    configs.append(config3)
    
    # Configuration 4: Compact hexagonal cluster (from INSPIRATION 1)
    config4 = [
        [0, 0, 0],          # center
        [0, 1.85, 0],       # top
        [0, -1.85, 0],      # bottom  
        [1.59, 0.89, 0],    # top-right
        [-1.59, 0.89, 0],   # top-left
        [1.59, -0.89, 0],   # bottom-right
        [-1.59, -0.89, 0],  # bottom-left
        [3.18, 0, 0],       # far right
        [-3.18, 0, 0],      # far left
        [1.59, 2.74, 0],    # upper far right
        [-1.59, 2.74, 0],   # upper far left
    ]
    configs.append(config4)
    
    # Configuration 5: Extended pattern with radial symmetry (from INSPIRATION 2)
    config5 = [
        [0, 0, 0],          # center
        [0, 2.05, 0],       # top
        [1.77, 1.03, 0],    # top-right
        [1.77, -1.03, 0],   # bottom-right
        [0, -2.05, 0],      # bottom
        [-1.77, -1.03, 0],  # bottom-left
        [-1.77, 1.03, 0],   # top-left
        [3.54, 0, 0],       # far right
        [-3.54, 0, 0],      # far left
        [0, 4.1, 0],        # very top
        [0, -4.1, 0],       # very bottom
    ]
    configs.append(config5)
    
    # Configuration 6: Optimized for minimal outer radius (from INSPIRATION 3)
    config6 = [
        [0, 0, 0],          # center
        [0, 1.87, 0],       # top
        [0, -1.87, 0],      # bottom  
        [1.62, 0.92, 0],    # top-right
        [-1.62, 0.92, 0],   # top-left
        [1.62, -0.92, 0],   # bottom-right
        [-1.62, -0.92, 0],  # bottom-left
        [3.24, 0, 0],       # far right
        [-3.24, 0, 0],      # far left
        [1.62, 2.8, 0],     # upper far right
        [-1.62, 2.8, 0],    # upper far left
    ]
    configs.append(config6)
    
    # Configuration 7: More compact version with slight adjustments (from INSPIRATION 1)
    config7 = [
        [0, 0, 0],          # center
        [0, 1.9, 0],        # top
        [0, -1.9, 0],       # bottom  
        [1.65, 0.94, 0],    # top-right
        [-1.65, 0.94, 0],   # top-left
        [1.65, -0.94, 0],   # bottom-right
        [-1.65, -0.94, 0],  # bottom-left
        [3.3, 0, 0],        # far right
        [-3.3, 0, 0],       # far left
        [1.65, 2.85, 0],    # upper far right
        [-1.65, 2.85, 0],   # upper far left
    ]
    configs.append(config7)
    
    # Configuration 8: Even more spread-out for better optimization (from INSPIRATION 2)
    config8 = [
        [0, 0, 0],          # center
        [0, 2.2, 0],        # top
        [0, -2.2, 0],       # bottom  
        [1.9, 1.1, 0],      # top-right
        [-1.9, 1.1, 0],     # top-left
        [1.9, -1.1, 0],     # bottom-right
        [-1.9, -1.1, 0],    # bottom-left
        [3.8, 0, 0],        # far right
        [-3.8, 0, 0],       # far left
        [1.9, 3.3, 0],      # upper far right
        [-1.9, 3.3, 0],     # upper far left
    ]
    configs.append(config8)
    
    # Configuration 9: Clustered with radial symmetry (from INSPIRATION 3)
    config9 = [
        [0, 0, 0],          # center
        [0, 1.8, 0],        # top
        [0, -1.8, 0],       # bottom  
        [1.55, 0.88, 0],    # top-right
        [-1.55, 0.88, 0],   # top-left
        [1.55, -0.88, 0],   # bottom-right
        [-1.55, -0.88, 0],  # bottom-left
        [3.1, 0, 0],        # far right
        [-3.1, 0, 0],       # far left
        [1.55, 2.68, 0],    # upper far right
        [-1.55, 2.68, 0],   # upper far left
    ]
    configs.append(config9)
    
    # Configuration 10: Alternative honeycomb pattern (from INSPIRATION 1)
    config10 = [
        [0, 0, 0],          # center
        [0, 1.8, 0],        # top
        [0, -1.8, 0],       # bottom  
        [1.5, 0.85, 0],     # top-right
        [-1.5, 0.85, 0],    # top-left
        [1.5, -0.85, 0],    # bottom-right
        [-1.5, -0.85, 0],   # bottom-left
        [3.0, 0, 0],        # far right
        [-3.0, 0, 0],       # far left
        [1.5, 2.6, 0],      # upper far right
        [-1.5, 2.6, 0],     # upper far left
    ]
    configs.append(config10)
    
    # Configuration 11: Very tight packing approach (from INSPIRATION 2)
    config11 = [
        [0, 0, 0],          # center
        [0, 1.92, 0],       # top
        [0, -1.92, 0],      # bottom  
        [1.66, 0.95, 0],    # top-right
        [-1.66, 0.95, 0],   # top-left
        [1.66, -0.95, 0],   # bottom-right
        [-1.66, -0.95, 0],  # bottom-left
        [3.32, 0, 0],       # far right
        [-3.32, 0, 0],      # far left
        [1.66, 2.87, 0],    # upper far right
        [-1.66, 2.87, 0],   # upper far left
    ]
    configs.append(config11)
    
    # Configuration 12: Balanced spread pattern (from INSPIRATION 3)
    config12 = [
        [0, 0, 0],          # center
        [0, 2.0, 0],        # top
        [0, -2.0, 0],       # bottom  
        [1.73, 1.0, 0],     # top-right
        [-1.73, 1.0, 0],    # top-left
        [1.73, -1.0, 0],    # bottom-right
        [-1.73, -1.0, 0],   # bottom-left
        [3.46, 0, 0],       # far right
        [-3.46, 0, 0],      # far left
        [1.73, 3.0, 0],     # upper far right
        [-1.73, 3.0, 0],    # upper far left
    ]
    configs.append(config12)
    
    return configs


def evaluate_discrete_config(config):
    """Evaluate a discrete configuration for validity and return its inverse radius."""
    # Extract positions and angles
    positions = [(hex_data[0], hex_data[1]) for hex_data in config]
    angles = [hex_data[2] for hex_data in config]
    
    # Check collisions (early exit if invalid)
    for i, j in combinations(range(len(positions)), 2):
        if check_collision_shapely(positions[i], angles[i], positions[j], angles[j]):
            return None, None
    
    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius(positions, angles)
    inv_radius = 1.0 / outer_radius
    
    return inv_radius, config


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining geometric insight and optimization.
    """
    # Generate multiple high-quality initial configurations
    candidate_configs = generate_multiple_initial_configs()
    
    # Evaluate all discrete configurations and select the best valid one
    best_config = None
    best_inv_radius = 0
    start_time = time.time()
    
    # Keep track of evaluation time to stay within limits
    for i, config in enumerate(candidate_configs):
        if time.time() - start_time > 45:  # Leave time for optimization
            break
            
        inv_radius, valid_config = evaluate_discrete_config(config)
        if inv_radius is not None and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = valid_config
    
    # If no valid discrete configuration found, use the most symmetric one
    if best_config is None:
        # Fallback to the most symmetric configuration
        best_config = [
            [0, 0, 0],          # center
            [0, 2.0, 0],        # top
            [0, -2.0, 0],       # bottom  
            [1.732, 1.0, 0],    # top-right
            [-1.732, 1.0, 0],   # top-left
            [1.732, -1.0, 0],   # bottom-right
            [-1.732, -1.0, 0],  # bottom-left
            [3.464, 0, 0],      # far right
            [-3.464, 0, 0],     # far left
            [1.732, 3.0, 0],    # upper far right
            [-1.732, 3.0, 0],   # upper far left
        ]
    
    # Now perform optimization refinement on the best discrete configuration
    # Convert best discrete config to optimization parameters
    positions = [(hex_data[0], hex_data[1]) for hex_data in best_config]
    angles = [hex_data[2] for hex_data in best_config]
    
    # Start with the best discrete configuration and refine it
    initial_params = []
    for pos, angle in zip(positions, angles):
        initial_params.extend([pos[0], pos[1], angle])
    
    # Use multiple optimization attempts with very tight tolerances
    best_result = None
    best_value = float('inf')
    
    # Enhanced optimization strategies with more aggressive settings
    strategies = [
        # Most aggressive optimization strategies - from INSPIRATION 3
        ('L-BFGS-B', {'maxiter': 20000, 'ftol': 1e-50, 'gtol': 1e-50}),
        ('TNC', {'maxiter': 20000, 'ftol': 1e-50, 'gtol': 1e-50}),
        ('SLSQP', {'maxiter': 20000, 'ftol': 1e-50, 'gtol': 1e-50}),
        ('trust-constr', {'maxiter': 8000, 'ftol': 1e-50, 'gtol': 1e-50}),
        # Add even more aggressive variants from INSPIRATION 3
        ('differential_evolution', {'maxiter': 2500, 'popsize': 35, 'tol': 1e-50}),
        # Also try COBYLA with very high iteration count
        ('COBYLA', {'maxiter': 20000}),
        # And Nelder-Mead with very high iteration count
        ('Nelder-Mead', {'maxiter': 20000}),
        # Add additional strategies for better exploration
        ('Powell', {'maxiter': 15000}),
        ('BFGS', {'maxiter': 15000}),
        # Add basin hopping for global exploration
        ('basin-hopping', {'maxiter': 1000}),
    ]
    
    # Try the discrete configuration with different optimization methods
    for method, options in strategies:
        if time.time() - start_time > 58:  # Leave time for final processing
            break
            
        try:
            # Special handling for differential evolution
            if method == 'differential_evolution':
                # Use bounds for DE
                bounds = [(-10, 10), (-10, 10), (0, 360)] * 11
                result = minimize(objective_function, initial_params, method=method, 
                                bounds=bounds, options=options)
            elif method == 'basin-hopping':
                # Basin hopping requires special handling
                from scipy.optimize import basinhopping
                result = basinhopping(objective_function, initial_params, 
                                    niter=1000, stepsize=0.5, interval=50, 
                                    minimizer_kwargs={'method': 'L-BFGS-B', 
                                                    'bounds': [(-10, 10)] * 33,
                                                    'options': {'ftol': 1e-50, 'gtol': 1e-50}})
            else:
                result = minimize(objective_function, initial_params, method=method, 
                                bounds=[(-10, 10)] * 33, options=options)
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result
                
        except Exception as e:
            continue
    
    # If optimization didn't work well, fallback to discrete configuration
    if best_result is None or not best_result.success:
        # Use the discrete configuration directly
        best_result = type('obj', (object,), {'x': initial_params, 'success': True})()
    
    # Extract final positions and angles
    final_positions = []
    final_angles = []
    for i in range(11):
        final_positions.append([best_result.x[3*i], best_result.x[3*i+1]])
        final_angles.append(best_result.x[3*i+2])
    
    # Compute final outer hexagon size
    outer_radius = compute_outer_hexagon_radius(final_positions, final_angles)
    
    # Convert to required format
    inner_hex_data = np.array([
        [pos[0], pos[1], angle] 
        for pos, angle in zip(final_positions, final_angles)
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
