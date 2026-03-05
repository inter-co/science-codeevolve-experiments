# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
from itertools import combinations

# Precompute hexagon vertices for unit regular hexagon
def get_hexagon_vertices(center=(0,0), rotation=0, side_length=1):
    """Get vertices of a regular hexagon"""
    angle = rotation * np.pi / 180
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = side_length * np.cos(theta)
        y = side_length * np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return np.array(vertices)

def compute_outer_hexagon_side_length(inner_hex_data):
    """Compute the minimum side length needed for outer hexagon to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        vertices = get_hexagon_vertices(center, rotation, 1)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1000000
        
    all_vertices = np.array(all_vertices)
    
    # Find the bounding circle radius - distance from origin to farthest point
    distances = np.linalg.norm(all_vertices, axis=1)
    max_distance = np.max(distances)
    
    # For a regular hexagon, the circumradius equals the side length
    # So we need side_length >= max_distance
    return max_distance

def hexagon_distance(hex1_center, hex1_rotation, hex2_center, hex2_rotation):
    """Compute minimum distance between two hexagons"""
    hex1_vertices = get_hexagon_vertices(hex1_center, hex1_rotation, 1)
    hex2_vertices = get_hexagon_vertices(hex2_center, hex2_rotation, 1)
    
    # Compute minimum distance between any pair of vertices
    distances = cdist(hex1_vertices, hex2_vertices)
    min_distance = np.min(distances)
    
    # Distance between hexagons is min_distance - 2 (since each has radius 1)
    return min_distance - 2.0

def check_overlap_and_containment(inner_hex_data):
    """More efficient overlap checking using distance-based approach"""
    # Check pairwise distances to detect overlaps
    num_hexagons = len(inner_hex_data)
    total_penalty = 0.0
    
    # Check overlaps between all pairs of hexagons
    for i, j in combinations(range(num_hexagons), 2):
        center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation1 = inner_hex_data[i, 2]
        center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
        rotation2 = inner_hex_data[j, 2]
        
        distance = hexagon_distance(center1, rotation1, center2, rotation2)
        
        # Overlap occurs when distance < 0
        if distance < 0:
            total_penalty += 10000.0 * abs(distance)
    
    # Check containment - all hexagons should be within a bounding circle
    outer_radius = compute_outer_hexagon_side_length(inner_hex_data)
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        # Distance from origin to center
        dist_to_center = np.sqrt(center[0]**2 + center[1]**2)
        # Add penalty if center is too far from origin
        if dist_to_center + 1.0 > outer_radius + 0.1:  # Allow small margin
            total_penalty += 1000.0 * (dist_to_center + 1.0 - outer_radius)
    
    return total_penalty

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a mathematical approach with symmetry consideration and geometric optimization.
    """
    
    # Mathematical approach: Start with known optimal configurations
    # Based on hexagonal lattice packing principles and symmetry
    
    sqrt3 = np.sqrt(3)
    sqrt3_over_2 = sqrt3 / 2.0
    
    # Configuration based on hexagonal lattice structure
    # Using a 3x4 rectangular arrangement in hexagonal coordinates
    base_config = np.array([
        # Row 1
        [0.0, 0.0, 0.0],           # center
        [sqrt3, 0.0, 0.0],         # right
        [sqrt3/2.0, 1.5, 0.0],     # top-right
        [-sqrt3/2.0, 1.5, 0.0],    # top-left
        
        # Row 2  
        [0.0, 3.0, 0.0],           # top center
        [sqrt3, 3.0, 0.0],         # top right
        [sqrt3/2.0, 4.5, 0.0],     # top-top-right
        [-sqrt3/2.0, 4.5, 0.0],    # top-top-left
        
        # Row 3
        [0.0, 6.0, 0.0],           # bottom center
        [sqrt3, 6.0, 0.0],         # bottom right
        [sqrt3/2.0, 7.5, 0.0],     # bottom-bottom-right
        [-sqrt3/2.0, 7.5, 0.0],    # bottom-bottom-left
    ], dtype=np.float64)
    
    # Adjust to get better packing
    # Apply symmetry reduction to reduce degrees of freedom
    adjusted_config = base_config.copy()
    
    # Scale down to improve packing efficiency
    scale_factor = 0.95
    adjusted_config[:, 0] *= scale_factor
    adjusted_config[:, 1] *= scale_factor
    
    # Create symmetric arrangement with rotations
    # Use rotational symmetry to improve packing
    final_config = np.zeros((12, 3))
    
    # Place centers in a pattern that respects hexagonal symmetry
    positions = [
        (0.0, 0.0),      # center
        (0.0, 2.0),      # top
        (0.0, -2.0),     # bottom
        (sqrt3, 1.0),    # top-right
        (-sqrt3, 1.0),   # top-left
        (sqrt3, -1.0),   # bottom-right
        (-sqrt3, -1.0),  # bottom-left
        (2*sqrt3, 0.0),  # far right
        (-2*sqrt3, 0.0), # far left
        (sqrt3, 3.0),    # upper right
        (-sqrt3, 3.0),   # upper left
        (sqrt3, -3.0),   # lower right
    ]
    
    # Apply scaling to get more compact arrangement
    positions = [(p[0]*0.9, p[1]*0.9) for p in positions]
    
    # Set up final configuration with rotations
    for i in range(12):
        final_config[i] = [positions[i][0], positions[i][1], 0.0]
    
    # Refine using a local optimization approach that maintains geometric properties
    def objective(params):
        # Reshape parameters
        inner_hex_data = params.reshape(-1, 3)
        
        # Compute outer hexagon side length
        outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
        
        # Compute penalty for overlaps and containment violations
        penalty = check_overlap_and_containment(inner_hex_data)
        
        # Objective: maximize 1/outer_side_length (minimize -1/outer_side_length)
        # Add penalty for invalid configurations
        if penalty > 1e-3:
            return 1e10  # Invalid configuration
        if outer_side_length > 1000:
            return 1e10
            
        return -1.0 / outer_side_length
    
    # Use a more targeted optimization approach
    initial_params = final_config.flatten()
    
    # Define bounds more carefully
    bounds = []
    for i in range(12):
        # x coordinate bounds (reasonable range based on initial configuration)
        bounds.extend([
            (-6.0, 6.0),      # x coordinate
            (-6.0, 6.0),      # y coordinate  
            (0, 360)          # rotation (0-360 degrees)
        ])
    
    # Optimization with a more targeted approach
    try:
        # First, try a simple gradient-based optimization with bounds
        result = minimize(
            fun=objective,
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-10},
            tol=1e-10
        )
        
        if result.success:
            refined_params = result.x.reshape(-1, 3)
            refined_side_length = compute_outer_hexagon_side_length(refined_params)
            refined_penalty = check_overlap_and_containment(refined_params)
            
            if refined_penalty < 1e-3 and refined_side_length < 1000:
                return refined_params, np.array([0, 0, 0]), refined_side_length
    except Exception:
        pass
    
    # If optimization failed, return the carefully constructed configuration
    return final_config, np.array([0, 0, 0]), compute_outer_hexagon_side_length(final_config)


# EVOLVE-BLOCK-END
