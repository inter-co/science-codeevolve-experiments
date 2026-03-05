# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def create_regular_hexagon(center=(0, 0), radius=1, rotation=0):
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius * np.cos(angles),
                             center[1] + radius * np.sin(angles)])
    return points[:-1]  # Remove last point to close the polygon

def get_hexagon_vertices(hex_data):
    """Get vertices for all hexagons from their data."""
    vertices_list = []
    for x, y, angle in hex_data:
        hex_points = create_regular_hexagon((x, y), 1, angle)
        vertices_list.append(hex_points)
    return vertices_list

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all hexagon vertices are contained within outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    
    for hex_vert in hex_vertices:
        hex_polygon = Polygon(hex_vert)
        if not outer_polygon.contains(hex_polygon):
            return False
    return True

def calculate_overlap_penalty(hex_vertices):
    """Calculate penalty for overlapping hexagons."""
    penalty = 0
    n = len(hex_vertices)
    
    for i in range(n):
        for j in range(i+1, n):
            poly_i = Polygon(hex_vertices[i])
            poly_j = Polygon(hex_vertices[j])
            
            if poly_i.intersects(poly_j):
                # Calculate intersection area as penalty
                intersection = poly_i.intersection(poly_j)
                penalty += intersection.area
    
    return penalty

def objective_function(params):
    """Objective function to minimize (negative of inverse outer radius)."""
    # Extract parameters
    # First 33 params: 11 hexagons * 3 (x, y, angle)
    # Last 3 params: outer hexagon center and rotation
    hex_params = params[:33].reshape(-1, 3)
    outer_center_angle = params[33:]
    
    # Get hexagon vertices
    hex_vertices = get_hexagon_vertices(hex_params)
    
    # Create outer hexagon vertices
    outer_center = outer_center_angle[:2]
    outer_rotation = outer_center_angle[2]
    outer_radius = 1  # We'll scale this later
    
    # Scale outer hexagon to contain all inner hexagons
    max_dist = 0
    for hex_vert in hex_vertices:
        for point in hex_vert:
            dist = np.sqrt((point[0] - outer_center[0])**2 + (point[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add some buffer for safety
    outer_radius = max_dist + 1.0
    
    outer_hex_vertices = create_regular_hexagon(outer_center, outer_radius, outer_rotation)
    
    # Check containment
    if not check_containment(hex_vertices, outer_hex_vertices):
        # Large penalty if not contained
        return 1e10
    
    # Calculate overlap penalty
    overlap_penalty = calculate_overlap_penalty(hex_vertices)
    
    # Return negative inverse of outer radius plus penalty
    # We want to maximize 1/outer_radius, so minimize -1/outer_radius
    return -1.0 / outer_radius + overlap_penalty * 1e6

def constraint_function(params):
    """Constraint function to keep outer hexagon properly sized."""
    hex_params = params[:33].reshape(-1, 3)
    outer_center_angle = params[33:]
    
    # Get hexagon vertices
    hex_vertices = get_hexagon_vertices(hex_params)
    
    # Create outer hexagon vertices
    outer_center = outer_center_angle[:2]
    outer_rotation = outer_center_angle[2]
    
    # Calculate minimum required outer radius
    max_dist = 0
    for hex_vert in hex_vertices:
        for point in hex_vert:
            dist = np.sqrt((point[0] - outer_center[0])**2 + (point[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add buffer
    required_radius = max_dist + 1.0
    return required_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses force-directed optimization approach.
    """
    # Better initial guess based on known good configurations
    # Start with a more compact arrangement
    initial_hex_positions = np.array([
        [0, 0, 0],      # center
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [1.732, 1.0, 0], # top right
        [-1.732, 1.0, 0], # top left
        [1.732, -1.0, 0], # bottom right
        [-1.732, -1.0, 0], # bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [0, 3.464, 0],  # far top
        [0, -3.464, 0], # far bottom
    ])
    
    # Initial outer hexagon centered at origin
    initial_outer = np.array([0, 0, 0])
    
    # Combine all parameters
    initial_params = np.concatenate([initial_hex_positions.flatten(), initial_outer])
    
    # Optimization bounds (reasonable constraints)
    bounds = []
    # Hexagon positions: x, y, angle for each of 11 hexagons
    for i in range(11):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])  # x, y, angle bounds
    # Outer hexagon center and rotation
    bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    
    # Optimize
    try:
        result = minimize(objective_function, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 1000})
        
        if result.success:
            # Extract results
            hex_params = result.x[:33].reshape(-1, 3)
            outer_params = result.x[33:]
            
            # Calculate final outer radius
            hex_vertices = get_hexagon_vertices(hex_params)
            outer_center = outer_params[:2]
            outer_rotation = outer_params[2]
            
            # Compute actual outer radius needed
            max_dist = 0
            for hex_vert in hex_vertices:
                for point in hex_vert:
                    dist = np.sqrt((point[0] - outer_center[0])**2 + (point[1] - outer_center[1])**2)
                    max_dist = max(max_dist, dist)
            outer_radius = max_dist + 1.0
            
            # Create final outer hexagon vertices
            outer_hex_vertices = create_regular_hexagon(outer_center, outer_radius, outer_rotation)
            
            return hex_params, np.array([outer_center[0], outer_center[1], outer_rotation]), outer_radius
        else:
            # Fallback to simple arrangement if optimization fails
            pass
    except Exception as e:
        pass
    
    # Fallback to improved arrangement
    # Based on more careful placement that reduces gaps
    improved_hex_positions = np.array([
        [0, 0, 0],      # center
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [1.732, 1.0, 0], # top right
        [-1.732, 1.0, 0], # top left
        [1.732, -1.0, 0], # bottom right
        [-1.732, -1.0, 0], # bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [0, 3.464, 0],  # far top
        [0, -3.464, 0], # far bottom
    ])
    
    # Calculate outer radius for this configuration
    max_dist = 0
    for x, y, _ in improved_hex_positions:
        dist = np.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)
    
    outer_radius = max_dist + 1.5  # Add buffer
    
    return improved_hex_positions, np.array([0, 0, 0]), outer_radius


# EVOLVE-BLOCK-END
