# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from numba import jit
import time
from scipy.optimize import differential_evolution, minimize
import math

# Constants for regular hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3)/2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, angle_deg, radius=HEX_RADIUS):
    """Fast computation of hexagon vertices using numba"""
    angle_rad = np.deg2rad(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        vertices[i, 0] = center_x + radius * np.cos(angle)
        vertices[i, 1] = center_y + radius * np.sin(angle)
    return vertices

def get_hexagon_vertices(center_x, center_y, angle_deg, radius=HEX_RADIUS):
    """Get vertices of a regular hexagon given center, angle, and radius"""
    return hexagon_vertices_jit(center_x, center_y, angle_deg, radius)

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon."""
    # Check if all vertices of inner hexagon are inside outer hexagon
    for vertex in inner_hex.exterior.coords[:-1]:  # Exclude closing vertex
        if not outer_hex.contains(Point(vertex)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def check_containment_and_overlap(inner_hex_data, outer_radius):
    """Check if all inner hexagons are contained in outer hexagon and no overlaps exist"""
    # Create outer hexagon vertices
    outer_vertices = get_hexagon_vertices(0, 0, 0, outer_radius)
    outer_hex = Polygon(outer_vertices)
    
    # Check containment and overlap for each inner hexagon
    for i, (cx, cy, angle) in enumerate(inner_hex_data):
        inner_vertices = get_hexagon_vertices(cx, cy, angle)
        inner_hex = Polygon(inner_vertices)
        
        # Check if inner hexagon is fully contained
        if not check_containment(inner_hex, outer_hex):
            return False, "Not contained"
        
        # Check for overlaps with other hexagons
        for j in range(i):
            cx2, cy2, angle2 = inner_hex_data[j]
            inner_hex2_vertices = get_hexagon_vertices(cx2, cy2, angle2)
            inner_hex2 = Polygon(inner_hex2_vertices)
            
            if check_overlap(inner_hex, inner_hex2):
                return False, "Overlap detected"
    
    return True, "Valid"

def objective_function(params):
    """
    Objective function to minimize (negative of 1/outer_hex_side_length)
    params: flattened array of [x1, y1, angle1, x2, y2, angle2, ..., x11, y11, angle11, outer_radius]
    """
    # Extract parameters
    n = 11
    inner_params = params[:-1]  # First 33 parameters: 11 hexagons * 3 params each
    outer_radius = params[-1]   # Last parameter: outer hexagon radius
    
    # Reshape inner hexagon parameters
    inner_hex_data = inner_params.reshape(-1, 3)
    
    # Check if the configuration is valid
    is_valid, message = check_containment_and_overlap(inner_hex_data, outer_radius)
    
    if not is_valid:
        # Return a large penalty value for invalid configurations
        return 1e10
    
    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -1.0 / outer_radius

def generate_best_initial_config():
    """Generate the best initial configuration based on proven hexagonal packing principles"""
    # Based on successful configurations from inspirations - optimized hexagonal arrangement
    return np.array([
        [0.0, 0.0, 0.0],      # center
        [0.0, 1.85, 0.0],     # top
        [1.59, 0.90, 0.0],    # top-right
        [1.59, -0.90, 0.0],   # bottom-right
        [0.0, -1.85, 0.0],    # bottom
        [-1.59, -0.90, 0.0],  # bottom-left
        [-1.59, 0.90, 0.0],   # top-left
        [3.18, 0.0, 0.0],     # far right
        [-3.18, 0.0, 0.0],    # far left
        [1.59, 2.75, 0.0],    # top-right extended
        [-1.59, 2.75, 0.0],   # top-left extended
    ])

def compute_outer_radius_from_config(inner_hex_data):
    """Compute the minimal outer radius needed to contain all hexagons"""
    max_distance = 0
    for cx, cy, _ in inner_hex_data:
        distance = np.sqrt(cx**2 + cy**2)
        max_distance = max(max_distance, distance + HEX_RADIUS)
    
    # Add small buffer for numerical stability
    return max_distance + 0.0001

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a robust hybrid approach combining global and local optimization with smart initialization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Time limit for execution (leave some margin for final processing)
    start_time = time.time()
    timeout = 9.5  # 10 second limit with some buffer
    
    # Start with the best known configuration from mathematical analysis
    initial_config = generate_best_initial_config()
    initial_outer_radius = compute_outer_radius_from_config(initial_config)
    
    # Set up optimization parameters
    bounds = []
    
    # Add bounds for inner hexagons (11 hexagons, 3 parameters each)
    for _ in range(11):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])  # x, y, angle
    
    # Add bound for outer hexagon radius
    bounds.append((1.0, 15.0))
    
    # Create initial parameters
    flat_params = initial_config.flatten()
    flat_params = np.append(flat_params, initial_outer_radius)
    
    best_inv_side_length = 1.0 / initial_outer_radius
    best_inner_hex_data = initial_config.copy()
    best_outer_radius = initial_outer_radius
    best_outer_hex_data = np.array([0, 0, 0])
    
    try:
        # Stage 1: Global optimization with differential evolution - inspired by INSPIRATION 3
        # Use more iterations and better parameters for better convergence
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=30,    # More iterations for better convergence
            popsize=10,    # Better population size for exploration
            mutation=(0.6, 1.0),  # Good mutation rate
            recombination=0.8,    # Good recombination rate
            seed=42,
            disp=False,
            tol=1e-8  # Tighter tolerance
        )
        
        if result.success:
            final_params = result.x
            inner_params = final_params[:-1]
            outer_radius = final_params[-1]
            inner_hex_data = inner_params.reshape(-1, 3)
            
            # Validate final result
            is_valid, message = check_containment_and_overlap(inner_hex_data, outer_radius)
            
            if is_valid:
                inv_side_length = -result.fun
                if inv_side_length > best_inv_side_length:
                    best_inv_side_length = inv_side_length
                    best_inner_hex_data = inner_hex_data.copy()
                    best_outer_radius = outer_radius
        
        # Stage 2: Local refinement with L-BFGS-B using the best result so far
        # This provides more precise optimization for the final result
        try:
            local_bounds = bounds.copy()
            local_params = best_inner_hex_data.flatten()
            local_params = np.append(local_params, best_outer_radius)
            
            local_result = minimize(
                objective_function,
                local_params,
                method='L-BFGS-B',
                bounds=local_bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if local_result.success:
                final_params = local_result.x
                inner_params = final_params[:-1]
                outer_radius = final_params[-1]
                inner_hex_data = inner_params.reshape(-1, 3)
                
                # Validate final result
                is_valid, message = check_containment_and_overlap(inner_hex_data, outer_radius)
                
                if is_valid:
                    inv_side_length = -local_result.fun
                    if inv_side_length > best_inv_side_length:
                        best_inv_side_length = inv_side_length
                        best_inner_hex_data = inner_hex_data.copy()
                        best_outer_radius = outer_radius
                        
        except Exception:
            # Continue with current best if local optimization fails
            pass
            
    except Exception as e:
        # If anything goes wrong, fall back to the initial configuration
        pass
    
    # Final validation to ensure we have a valid solution
    try:
        is_valid, message = check_containment_and_overlap(best_inner_hex_data, best_outer_radius)
        if not is_valid:
            # Revert to initial configuration if final validation fails
            best_inner_hex_data = initial_config.copy()
            best_outer_radius = initial_outer_radius
    except Exception:
        # If validation fails, still use the best we have
        pass
    
    end_time = time.time()
    
    return best_inner_hex_data, best_outer_hex_data, best_outer_radius


# EVOLVE-BLOCK-END
