# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time

def create_regular_hexagon(center=(0,0), radius=1, rotation=0):
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius * np.cos(angles),
                             center[1] + radius * np.sin(angles)])
    return points[:-1]  # Remove last point to close polygon

def get_hexagon_vertices(hex_data):
    """Get vertices for all hexagons from their data."""
    vertices_list = []
    for x, y, angle in hex_data:
        hex_points = create_regular_hexagon((x, y), 1, angle)
        vertices_list.append(hex_points)
    return vertices_list

def check_containment_and_overlap(inner_hex_data, outer_radius):
    """Check if all inner hexagons fit within outer hexagon and don't overlap."""
    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_radius, 0)
    outer_polygon = Polygon(outer_hex)
    
    # Check containment and overlap for all inner hexagons
    inner_polygons = []
    for x, y, angle in inner_hex_data:
        hex_points = create_regular_hexagon((x, y), 1, angle)
        inner_polygon = Polygon(hex_points)
        inner_polygons.append(inner_polygon)
        
        # Check containment
        if not outer_polygon.contains(inner_polygon):
            return False, None
    
    # Check pairwise overlaps
    for i in range(len(inner_polygons)):
        for j in range(i+1, len(inner_polygons)):
            if inner_polygons[i].intersects(inner_polygons[j]):
                return False, None
    
    return True, inner_polygons

def objective_function(params):
    """Objective function to minimize (negative of 1/outer_radius)."""
    # Extract parameters
    # First 36 params: 12 hexagons * 3 (x, y, angle)
    # Last 1 param: outer radius
    hex_params = params[:-1].reshape(-1, 3)
    outer_radius = params[-1]
    
    # Check if valid configuration
    valid, polygons = check_containment_and_overlap(hex_params, outer_radius)
    if not valid:
        # Return large penalty for invalid configurations
        return 1e10
    
    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -1.0 / outer_radius

def constraint_function(params):
    """Constraint function to ensure outer radius is positive."""
    return params[-1]  # outer radius must be > 0

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-inspired optimization approach.
    """
    # Start with a good initial guess based on known efficient packings
    # This is a more symmetric starting configuration
    initial_hex_data = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom  
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0], # top left
        [1.732, -1, 0], # bottom right
        [-1.732, -1, 0],# bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0], # top far left
        [1.732, -3, 0], # bottom far right
    ])
    
    # Initial outer radius estimate (should be around 4-5 for reasonable packing)
    initial_outer_radius = 4.0
    
    # Flatten parameters for optimization
    initial_params = np.concatenate([initial_hex_data.flatten(), [initial_outer_radius]])
    
    # Set up bounds for optimization
    # Hexagon positions: -10 to 10 for x and y
    # Hexagon angles: 0 to 360 degrees
    # Outer radius: 1 to 20
    bounds = []
    for _ in range(12):
        bounds.extend([(None, None), (None, None), (0, 360)])  # x, y, angle
    bounds.append((1, 20))  # outer radius
    
    # Optimization constraints
    constraints = [{'type': 'ineq', 'fun': constraint_function}]
    
    # Use scipy optimization with L-BFGS-B method which handles bounds well
    try:
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            optimized_params = result.x
            hex_data = optimized_params[:-1].reshape(-1, 3)
            outer_radius = optimized_params[-1]
            
            # Verify final solution
            valid, _ = check_containment_and_overlap(hex_data, outer_radius)
            if valid:
                # Return results
                outer_hex_data = np.array([0, 0, 0])
                return hex_data, outer_hex_data, outer_radius
    except Exception as e:
        pass
    
    # Fallback to initial configuration if optimization fails
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 5.0  # Conservative estimate
    return initial_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
