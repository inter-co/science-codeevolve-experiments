# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time


def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon with given center and rotation."""
    angle = rotation * np.pi / 180
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return Polygon(vertices)


def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon."""
    return outer_hexagon.contains(hexagon)


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)


def evaluate_packing(params):
    """Evaluate a packing configuration and return negative inverse outer radius (for minimization)."""
    # Extract parameters
    inner_positions = params[:22].reshape(-1, 2)  # 11 hexagons * 2 coords
    inner_rotations = params[22:33]  # 11 rotations
    outer_center = params[33:35]  # outer hex center
    outer_radius = params[35]  # outer hex radius
    
    # Create outer hexagon (unit size)
    outer_hex = create_unit_hexagon((0, 0), 0)
    
    # Scale and translate outer hexagon to desired size and position
    scaled_outer_vertices = []
    for x, y in outer_hex.exterior.coords[:-1]:  # Exclude duplicate last point
        scaled_x = outer_center[0] + x * outer_radius
        scaled_y = outer_center[1] + y * outer_radius
        scaled_outer_vertices.append((scaled_x, scaled_y))
    scaled_outer_hex = Polygon(scaled_outer_vertices)
    
    # Create inner hexagons
    inner_hexagons = []
    for i, (pos, rot) in enumerate(zip(inner_positions, inner_rotations)):
        hexagon = create_unit_hexagon(pos, rot)
        inner_hexagons.append(hexagon)
    
    # Check containment
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, scaled_outer_hex):
            return 1e10  # Large penalty for containment violation
    
    # Check overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i + 1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return 1e10  # Large penalty for overlap
    
    # Return negative inverse of outer radius (since we want to maximize 1/R)
    return -1.0 / outer_radius


def calculate_min_outer_radius_for_positions(positions):
    """Calculate minimum outer hexagon radius needed for given positions."""
    max_dist = 0
    for x, y in positions:
        # Distance from center to vertex of hexagon at (x,y) is sqrt(x²+y²)+1
        dist = np.sqrt(x*x + y*y) + 1.0
        max_dist = max(max_dist, dist)
    return max_dist


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining intelligent initialization and aggressive optimization.
    """
    # Use the most precise initial configuration from INSPIRATION 1
    # These positions have been carefully chosen for maximum packing efficiency
    initial_positions = [
        (0.0, 0.0),        # center
        (0.0, 1.75),       # top
        (0.0, -1.75),      # bottom
        (1.52, 0.87),      # top-right
        (-1.52, 0.87),     # top-left
        (1.52, -0.87),     # bottom-right
        (-1.52, -0.87),    # bottom-left
        (3.04, 0.0),       # far right
        (-3.04, 0.0),      # far left
        (1.52, 2.61),      # upper right
        (-1.52, 2.61),     # upper left
    ]
    
    # Initialize rotations (all 0 for simplicity initially)
    initial_rotations = [0] * 11
    
    # Estimate initial outer radius more carefully
    initial_outer_radius = calculate_min_outer_radius_for_positions(initial_positions)
    
    # Initial outer hexagon parameters (centered)
    initial_outer_center = [0, 0]
    
    # Flatten parameters for optimization
    initial_params = (
        np.array(initial_positions).flatten().tolist() +
        initial_rotations +
        initial_outer_center +
        [initial_outer_radius]
    )
    
    # Define bounds for optimization - more focused bounds for better convergence
    bounds = []
    # Positions: x,y coordinates for 11 hexagons - tighter bounds for better convergence
    for _ in range(11):
        bounds.extend([(-12, 12), (-12, 12)])  # slightly wider bounds for exploration
    # Rotations: 0-360 degrees
    bounds.extend([(0, 360)] * 11)
    # Outer center
    bounds.extend([(-10, 10), (-10, 10)])
    # Outer radius: positive values, reasonable upper bound
    bounds.append((1.0, 15.0))
    
    start_time = time.time()
    
    # Phase 1: Very aggressive global optimization with high precision settings
    try:
        result = differential_evolution(
            evaluate_packing,
            bounds,
            maxiter=800,      # More iterations for better optimization
            popsize=150,      # Larger population for better exploration
            mutation=(0.999, 1),  # Near maximum mutation for good exploration
            recombination=0.999,  # Near maximum recombination
            tol=1e-18,        # Extremely tight tolerance for maximum precision
            seed=42,
            disp=False,
            callback=lambda x, convergence: time.time() - start_time > 55
        )
        
        # Phase 2: Local refinement with multiple strategies for ultimate precision
        if result.success:
            final_params = result.x
            
            # Try trust-constr optimization first for maximum precision
            try:
                refined_bounds = []
                for i in range(11):
                    x, y = final_params[2*i:2*i+2]
                    refined_bounds.extend([
                        (max(-12, x-0.01), min(12, x+0.01)), 
                        (max(-12, y-0.01), min(12, y+0.01))
                    ])
                refined_bounds.extend([(0, 360)] * 11)  # rotations
                refined_bounds.extend([(-10, 10), (-10, 10)])  # outer center
                refined_bounds.append((max(1.0, final_params[-1]*0.9999), min(15.0, final_params[-1]*1.0001)))  # radius
                
                local_result = minimize(
                    evaluate_packing,
                    final_params,
                    method='trust-constr',
                    bounds=refined_bounds,
                    options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18},
                    callback=lambda x: time.time() - start_time > 55
                )
                
                if local_result.success:
                    final_params = local_result.x
                else:
                    # Fallback to L-BFGS-B if trust-constr fails
                    local_result = minimize(
                        evaluate_packing,
                        final_params,
                        method='L-BFGS-B',
                        bounds=refined_bounds,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15},
                        callback=lambda x: time.time() - start_time > 55
                    )
                    
                    if local_result.success:
                        final_params = local_result.x
                        
            except:
                pass
                
        else:
            final_params = np.array(initial_params)
            
    except Exception as e:
        # Fallback to initial configuration
        final_params = np.array(initial_params)
    
    # Extract results
    inner_positions = final_params[:22].reshape(-1, 2)
    inner_rotations = final_params[22:33]
    outer_center = final_params[33:35]
    outer_radius = final_params[35]
    
    # Format output
    inner_hex_data = np.column_stack([
        inner_positions[:, 0],
        inner_positions[:, 1],
        inner_rotations
    ])
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0])
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
