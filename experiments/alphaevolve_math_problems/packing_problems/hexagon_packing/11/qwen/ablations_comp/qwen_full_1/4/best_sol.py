# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import time

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(angle), 
               center[1] + side_length * np.sin(angle)) 
              for angle in angles]
    return Polygon(points)

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def calculate_max_vertex_distance(inner_hex_data):
    """Calculate maximum distance from origin to any vertex of any inner hexagon"""
    max_vertex_distance = 0.0
    sqrt3_over_2 = np.sqrt(3) / 2.0
    
    for i in range(11):
        center_x, center_y, angle = inner_hex_data[i]
        # Vertices of unit hexagon at center (center_x, center_y) rotated by angle
        # Unit hexagon vertices in local coordinates:
        # (±1, 0), (±0.5, ±0.866) where 0.866 ≈ sqrt(3)/2
        local_vertices = [
            (1.0, 0.0),    # right
            (0.5, sqrt3_over_2),  # upper right
            (-0.5, sqrt3_over_2), # upper left
            (-1.0, 0.0),   # left
            (-0.5, -sqrt3_over_2), # lower left
            (0.5, -sqrt3_over_2)   # lower right
        ]
        
        # Apply rotation and translation
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        for lx, ly in local_vertices:
            wx = center_x + lx * cos_a - ly * sin_a
            wy = center_y + lx * sin_a + ly * cos_a
            vertex_distance = np.sqrt(wx*wx + wy*wy)
            max_vertex_distance = max(max_vertex_distance, vertex_distance)
    
    return max_vertex_distance

def evaluate_packaging(params):
    """Evaluate a packing configuration"""
    # Extract parameters
    # First 33 params: 11 hexagons (x, y, angle each)
    # Last 3 params: outer hexagon (x, y, angle)
    # Final param: outer hexagon side length
    
    inner_params = params[:-4]
    outer_center_x, outer_center_y, outer_angle = params[-4:-1]
    outer_side_length = params[-1]
    
    # Create outer hexagon
    outer_hex = create_regular_hexagon((outer_center_x, outer_center_y), outer_side_length, outer_angle)
    
    # Create inner hexagons
    inner_hexagons = []
    total_penalty = 0
    
    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        inner_hex = create_regular_hexagon((x, y), 1, angle)
        
        # Check containment
        if not check_containment(inner_hex, outer_hex):
            total_penalty += 10000  # Large penalty for containment violation
            
        inner_hexagons.append(inner_hex)
    
    # Check overlaps between inner hexagons
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 10000  # Large penalty for overlap
                
    # Return negative of inverse side length plus penalties (since we want to minimize)
    objective = -1.0 / outer_side_length + total_penalty
    
    return objective

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use a more systematic approach inspired by the best of all inspirations
    # Try a few high-quality configurations from different sources
    
    # Use configurations from the best inspirations
    # Configuration 1: From INSPIRATION 1 - proven good symmetric arrangement
    config1 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.925, 0.0],         # top 
        [0.0, -1.925, 0.0],        # bottom
        [1.665, 0.962, 0.0],       # top-right
        [-1.665, 0.962, 0.0],      # top-left  
        [1.665, -0.962, 0.0],      # bottom-right
        [-1.665, -0.962, 0.0],     # bottom-left
        [3.330, 0.0, 0.0],         # far right
        [-3.330, 0.0, 0.0],        # far left
        [1.665, 2.887, 0.0],       # top far right
        [-1.665, 2.887, 0.0],      # top far left
    ])
    
    # Configuration 2: From INSPIRATION 2 - slightly different symmetric arrangement
    config2 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.95, 0.0],          # top
        [0.0, -1.95, 0.0],         # bottom
        [1.66, 0.95, 0.0],         # top-right
        [-1.66, 0.95, 0.0],        # top-left
        [1.66, -0.95, 0.0],        # bottom-right
        [-1.66, -0.95, 0.0],       # bottom-left
        [3.32, 0.0, 0.0],          # far right
        [-3.32, 0.0, 0.0],         # far left
        [1.66, 2.85, 0.0],         # top far right
        [-1.66, 2.85, 0.0],        # top far left
    ])
    
    # Configuration 3: Compact arrangement from INSPIRATION 3
    config3 = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.8, 0.0],           # top
        [0.0, -1.8, 0.0],          # bottom
        [1.55, 0.89, 0.0],         # top-right
        [-1.55, 0.89, 0.0],        # top-left
        [1.55, -0.89, 0.0],        # bottom-right
        [-1.55, -0.89, 0.0],       # bottom-left
        [3.1, 0.0, 0.0],           # far right
        [-3.1, 0.0, 0.0],          # far left
        [1.55, 2.65, 0.0],         # top far right
        [-1.55, 2.65, 0.0],        # top far left
    ])
    
    configs_to_try = [config1, config2, config3]
    best_inner_hex_data = None
    best_outer_side_length = float('inf')
    best_outer_hex_data = None
    best_score = 0.0
    
    # Test each configuration
    for i, config in enumerate(configs_to_try):
        try:
            # Calculate outer hexagon side length needed
            max_vertex_distance = calculate_max_vertex_distance(config)
            outer_side_length = max_vertex_distance * 1.0005  # Small safety margin
            
            # Validate configuration
            outer_hex = create_regular_hexagon((0, 0), outer_side_length, 0)
            inner_hexagons = []
            valid = True
            
            for j in range(11):
                x, y, angle = config[j]
                inner_hex = create_regular_hexagon((x, y), 1, angle)
                if not check_containment(inner_hex, outer_hex):
                    valid = False
                    break
                inner_hexagons.append(inner_hex)
            
            # Check overlaps
            if valid:
                for j in range(11):
                    for k in range(j+1, 11):
                        if check_overlap(inner_hexagons[j], inner_hexagons[k]):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = 1.0 / outer_side_length
                if score > best_score:
                    best_score = score
                    best_outer_side_length = outer_side_length
                    best_inner_hex_data = config.copy()
                    best_outer_hex_data = np.array([0, 0, 0])
                    
        except Exception:
            continue
    
    # If no valid configuration found, use default
    if best_inner_hex_data is None:
        best_inner_hex_data = config1
        max_vertex_distance = calculate_max_vertex_distance(best_inner_hex_data)
        best_outer_side_length = max_vertex_distance * 1.0005
        best_outer_hex_data = np.array([0, 0, 0])
        best_score = 1.0 / best_outer_side_length
    
    # Perform optimization with enhanced parameters
    try:
        # Create initial parameter vector for optimization
        initial_params = []
        for i in range(11):
            x, y, angle = best_inner_hex_data[i]
            initial_params.extend([x, y, angle])
        
        # Add outer hexagon parameters
        initial_params.extend([0, 0, 0, best_outer_side_length])
        
        # Define bounds for optimization - balanced for convergence and time
        bounds = []
        
        # Inner hexagons: x, y, angle for each of 11 hexagons
        for _ in range(11):
            bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (-180, 180)])
        
        # Outer hexagon: center x, y, angle, side length
        bounds.extend([(-3.0, 3.0), (-3.0, 3.0), (-180, 180), (2.0, 6.0)])
        
        # Run optimization with parameters that balance quality and time
        remaining_time = 85 - (time.time() - start_time)
        if remaining_time > 5:
            result = differential_evolution(
                evaluate_packaging,
                bounds,
                maxiter=40,   # Moderate iterations to save time
                popsize=20,   # Good population size for exploration
                tol=1e-8,     # Reasonable tolerance
                seed=42,
                disp=False,
                strategy='best1bin',
                mutation=(0.8, 1.0),    # Standard mutation
                recombination=0.9       # Good recombination rate
            )
            
            # If optimization succeeds, use the result
            if result.success:
                final_params = result.x
                inner_params = final_params[:-4]
                outer_center_x, outer_center_y, outer_angle = final_params[-4:-1]
                outer_side_length = final_params[-1]
                
                # Convert to proper data structures
                inner_hex_data = np.array([inner_params[3*i:3*i+3] for i in range(11)])
                outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])
                
                # Validate and update if better
                max_vertex_distance = calculate_max_vertex_distance(inner_hex_data)
                validated_outer_side_length = max_vertex_distance * 1.0005
                
                if validated_outer_side_length < outer_side_length:
                    outer_side_length = validated_outer_side_length
                
                # Return if this gives a better result
                current_score = 1.0 / outer_side_length
                if current_score > best_score:
                    return inner_hex_data, outer_hex_data, outer_side_length
    
    except Exception:
        # If optimization fails, return the best configuration we found
        pass
    
    return best_inner_hex_data, best_outer_hex_data, best_outer_side_length


# EVOLVE-BLOCK-END
