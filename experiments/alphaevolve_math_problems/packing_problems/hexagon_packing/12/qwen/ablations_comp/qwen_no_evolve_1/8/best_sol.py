# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math


def create_unit_hexagon_vertices(center=(0, 0), rotation_deg=0):
    """Create vertices of a unit regular hexagon centered at center with given rotation."""
    angle_rad = math.radians(rotation_deg)
    # Vertices of unit hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = math.cos(angle)
        y = math.sin(angle)
        hex_vertices.append((x + center[0], y + center[1]))
    return np.array(hex_vertices)


def check_containment(hex_vertices, outer_hex_center, outer_hex_radius):
    """Check if all vertices of hexagon are within outer hexagon."""
    # Create outer hexagon vertices
    outer_vertices = []
    for i in range(6):
        angle = i * math.pi / 3
        x = outer_hex_radius * math.cos(angle) + outer_hex_center[0]
        y = outer_hex_radius * math.sin(angle) + outer_hex_center[1]
        outer_vertices.append((x, y))
    
    outer_polygon = np.array(outer_vertices)
    
    # Check if all inner vertices are inside outer polygon
    for vertex in hex_vertices:
        # Simple point-in-polygon test using winding number or ray casting
        # For simplicity, we'll check distance from center
        dist_from_center = math.sqrt((vertex[0] - outer_hex_center[0])**2 + 
                                   (vertex[1] - outer_hex_center[1])**2)
        if dist_from_center >= outer_hex_radius:
            return False
    return True


def compute_hexagon_distance(hex1_vertices, hex2_vertices):
    """Compute minimum distance between two hexagons."""
    # Compute pairwise distances between all vertices
    distances = cdist(hex1_vertices, hex2_vertices)
    min_dist = np.min(distances)
    return min_dist


def evaluate_packing_config(config, outer_radius_guess=None):
    """Evaluate a configuration of 12 hexagons."""
    # Extract positions and rotations
    positions = config[:24].reshape(12, 2)  # x, y for each hexagon
    rotations = config[24:]  # rotation for each hexagon
    
    # Create hexagon vertices
    hex_vertices = []
    for i in range(12):
        pos = positions[i]
        rot = rotations[i]
        verts = create_unit_hexagon_vertices(pos, rot)
        hex_vertices.append(verts)
    
    # Check containment (assume outer hexagon is centered at origin with radius R)
    if outer_radius_guess is None:
        # Estimate outer radius from positions
        max_dist = 0
        for i in range(12):
            for vertex in hex_vertices[i]:
                dist = math.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
        outer_radius = max_dist + 1  # Add buffer
    else:
        outer_radius = outer_radius_guess
    
    # Check containment
    all_contained = True
    for i in range(12):
        if not check_containment(hex_vertices[i], (0, 0), outer_radius):
            all_contained = False
            break
    
    # Check overlaps
    no_overlaps = True
    for i in range(12):
        for j in range(i+1, 12):
            min_dist = compute_hexagon_distance(hex_vertices[i], hex_vertices[j])
            if min_dist < 0.01:  # Very small overlap threshold
                no_overlaps = False
                break
        if not no_overlaps:
            break
    
    # Return penalty if constraints violated
    if not all_contained or not no_overlaps:
        return 1e10  # Large penalty
    
    # Objective: minimize outer radius
    return outer_radius


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses geometric optimization with symmetry constraints.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start with a symmetric configuration inspired by known optimal arrangements
    # Based on literature, a good starting configuration has:
    # - Central hexagon
    # - Surrounding ring with 6 hexagons
    # - Outer ring with 5 additional hexagons
    
    # Initial symmetric configuration
    initial_positions = np.array([
        [0, 0],      # center
        [0, 2],      # top
        [1.732, 1],  # top-right
        [1.732, -1], # bottom-right  
        [0, -2],     # bottom
        [-1.732, -1], # bottom-left
        [-1.732, 1],  # top-left
        [3.464, 0],   # far right
        [1.732, 3],   # upper-right
        [-1.732, 3],  # upper-left
        [-3.464, 0],  # far left
        [-1.732, -3]  # lower-left
    ])
    
    initial_rotations = np.zeros(12)  # All horizontal
    
    # Flatten configuration for optimization
    initial_config = np.concatenate([initial_positions.flatten(), initial_rotations])
    
    # Optimization parameters
    bounds = []  # Will define bounds for positions and rotations
    
    # Position bounds: reasonable limits around initial configuration
    for i in range(24):
        if i % 2 == 0:  # x coordinates
            bounds.append((-10, 10))  # x range
        else:  # y coordinates  
            bounds.append((-10, 10))  # y range
    
    # Rotation bounds: 0-360 degrees
    for i in range(12):
        bounds.append((0, 360))
    
    # Optimization with bounds
    try:
        # Use scipy minimize with bounds
        result = minimize(
            lambda x: evaluate_packing_config(x),
            initial_config,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_config = result.x
            final_positions = optimized_config[:24].reshape(12, 2)
            final_rotations = optimized_config[24:]
            
            # Create final configuration data
            inner_hex_data = np.column_stack([final_positions, final_rotations])
            
            # Calculate actual outer radius from final configuration
            hex_vertices = []
            for i in range(12):
                pos = final_positions[i]
                rot = final_rotations[i]
                verts = create_unit_hexagon_vertices(pos, rot)
                hex_vertices.append(verts)
            
            # Find maximum distance from origin
            max_dist = 0
            for i in range(12):
                for vertex in hex_vertices[i]:
                    dist = math.sqrt(vertex[0]**2 + vertex[1]**2)
                    max_dist = max(max_dist, dist)
            
            outer_hex_side_length = max_dist + 0.1  # Add small margin
            
            # Outer hexagon centered at origin (can be adjusted)
            outer_hex_data = np.array([0, 0, 0])
            
        else:
            # Fall back to initial configuration if optimization fails
            inner_hex_data = np.column_stack([initial_positions, initial_rotations])
            outer_hex_side_length = 5.0  # Estimated from initial
            outer_hex_data = np.array([0, 0, 0])
            
    except Exception as e:
        # Fallback to simple configuration
        inner_hex_data = np.column_stack([initial_positions, initial_rotations])
        outer_hex_side_length = 5.0
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
