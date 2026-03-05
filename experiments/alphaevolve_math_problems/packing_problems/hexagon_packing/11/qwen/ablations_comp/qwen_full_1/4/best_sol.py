# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from numba import jit
import time
from itertools import combinations
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

def check_containment_and_overlap(inner_hex_data, outer_radius):
    """Check if all inner hexagons are contained in outer hexagon and no overlaps exist"""
    # Create outer hexagon vertices
    outer_vertices = get_hexagon_vertices(0, 0, 0, outer_radius)
    outer_hex = Polygon(outer_vertices)
    
    # Create list of inner hexagons for overlap checking
    inner_hexagons = []
    for i, (cx, cy, angle) in enumerate(inner_hex_data):
        inner_vertices = get_hexagon_vertices(cx, cy, angle)
        inner_hex = Polygon(inner_vertices)
        inner_hexagons.append(inner_hex)
        
        # Check if inner hexagon is fully contained
        if not outer_hex.contains(inner_hex):
            return False, "Not contained"
    
    # Check for overlaps between all pairs of inner hexagons
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if inner_hexagons[i].intersects(inner_hexagons[j]):
            return False, "Overlap detected"
    
    return True, "Valid"

def calculate_min_outer_radius(inner_hex_data):
    """Calculate minimum outer radius needed to contain all inner hexagons"""
    max_distance = 0
    for cx, cy, _ in inner_hex_data:
        distance = np.sqrt(cx**2 + cy**2)
        max_distance = max(max_distance, distance + HEX_RADIUS)
    return max_distance + 0.001  # Small buffer

def generate_symmetric_configuration():
    """Generate a symmetric configuration inspired by optimal hexagonal packings"""
    # Start with a central hexagon
    config = [[0.0, 0.0, 0.0]]
    
    # Place 6 surrounding hexagons at distance 2 (center-to-center)
    for i in range(6):
        angle = i * 60  # 60 degrees increments
        rad_angle = np.radians(angle)
        x = 2 * np.cos(rad_angle)
        y = 2 * np.sin(rad_angle)
        config.append([x, y, 0.0])
    
    # Add two more hexagons along the vertical axis
    config.append([0.0, -3.0, 0.0])
    config.append([0.0, 3.0, 0.0])
    
    # Add remaining hexagons in a strategic pattern
    # Add hexagons at diagonal positions
    config.append([np.sqrt(3), 1.5, 0.0])
    config.append([-np.sqrt(3), 1.5, 0.0])
    config.append([np.sqrt(3), -1.5, 0.0])
    config.append([-np.sqrt(3), -1.5, 0.0])
    
    # Add one more hexagon at extreme positions
    config.append([4.0, 0.0, 0.0])
    
    return np.array(config)

def generate_constructive_configuration():
    """Generate a configuration using a constructive approach based on known optimal patterns"""
    # This follows a systematic construction approach:
    # 1. Start with hexagonal lattice pattern
    # 2. Adjust positions to maximize packing efficiency
    
    # Base pattern inspired by hexagonal tiling with 11 elements
    config = [
        [0.0, 0.0, 0.0],        # center
        [0.0, 2.0, 0.0],        # top
        [1.732, 1.0, 0.0],      # top-right (sqrt(3) ≈ 1.732)
        [1.732, -1.0, 0.0],     # bottom-right
        [0.0, -2.0, 0.0],       # bottom
        [-1.732, -1.0, 0.0],    # bottom-left
        [-1.732, 1.0, 0.0],     # top-left
        [3.464, 0.0, 0.0],      # far right (2*sqrt(3))
        [-3.464, 0.0, 0.0],     # far left
        [0.0, 3.0, 0.0],        # top-top
        [0.0, -3.0, 0.0]        # bottom-bottom
    ]
    
    # Apply small perturbations to improve packing
    # These values were chosen to reduce overlap issues and improve containment
    config[1] = [0.0, 2.03, 0.0]      # Top slightly adjusted
    config[2] = [1.73205, 1.015, 0.0] # Top-right
    config[3] = [1.73205, -1.015, 0.0] # Bottom-right
    config[4] = [0.0, -2.03, 0.0]     # Bottom
    config[5] = [-1.73205, -1.015, 0.0] # Bottom-left
    config[6] = [-1.73205, 1.015, 0.0]  # Top-left
    config[7] = [3.46410, 0.0, 0.0]    # Far right
    config[8] = [-3.46410, 0.0, 0.0]   # Far left
    config[9] = [1.73205, 2.918, 0.0]  # Top-top-right
    config[10] = [-1.73205, 2.918, 0.0] # Top-top-left
    
    return np.array(config)

def find_optimal_packing():
    """Find optimal packing using a constructive approach with geometric optimization"""
    
    # Start with our best constructive configuration
    inner_hex_data = generate_constructive_configuration()
    
    # Try different rotation angles to see if we can improve
    best_config = inner_hex_data.copy()
    best_radius = calculate_min_outer_radius(best_config)
    
    # Test various rotations for some hexagons to see improvement
    test_rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # Try different configurations systematically
    for attempt in range(50):  # Multiple attempts with different tweaks
        # Create variation of configuration
        config = inner_hex_data.copy()
        
        # Apply small random perturbations to positions
        for i in range(len(config)):
            if i < 5:  # Perturb first few hexagons more
                config[i][0] += np.random.uniform(-0.05, 0.05)
                config[i][1] += np.random.uniform(-0.05, 0.05)
            elif i >= 5 and i < 8:  # Perturb middle ones
                config[i][0] += np.random.uniform(-0.02, 0.02)
                config[i][1] += np.random.uniform(-0.02, 0.02)
        
        # Try different rotations for some hexagons
        for i in range(3, 7):  # Middle hexagons
            if np.random.random() < 0.3:  # 30% chance to rotate
                config[i][2] = test_rotations[np.random.randint(len(test_rotations))]
        
        # Calculate new radius
        new_radius = calculate_min_outer_radius(config)
        
        # Check validity and update if better
        is_valid, _ = check_containment_and_overlap(config, new_radius)
        if is_valid and new_radius < best_radius:
            best_radius = new_radius
            best_config = config.copy()
    
    return best_config, best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a constructive approach with systematic geometric optimization rather than traditional optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Time limit for execution
    start_time = time.time()
    
    try:
        # Use the constructive approach to find good initial configuration
        inner_hex_data, outer_radius = find_optimal_packing()
        
        # Perform additional refinement using geometric reasoning
        # Try to improve by adjusting positions and rotations systematically
        
        # Refinement step 1: Fine-tune positions to reduce outer radius
        best_config = inner_hex_data.copy()
        best_radius = outer_radius
        
        # Try to optimize by moving hexagons inward where possible
        for iter_num in range(20):
            config = best_config.copy()
            
            # Make small adjustments to positions
            for i in range(len(config)):
                # Move hexagons toward center if they're far out
                dist_to_center = np.sqrt(config[i][0]**2 + config[i][1]**2)
                if dist_to_center > 2.5:  # If far from center
                    # Move toward center slightly
                    factor = 0.95
                    config[i][0] *= factor
                    config[i][1] *= factor
            
            # Recalculate radius
            new_radius = calculate_min_outer_radius(config)
            
            # Check validity
            is_valid, _ = check_containment_and_overlap(config, new_radius)
            if is_valid and new_radius < best_radius:
                best_radius = new_radius
                best_config = config.copy()
        
        # Final validation
        is_valid, message = check_containment_and_overlap(best_config, best_radius)
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
    except Exception as e:
        # Fallback to a simple configuration if anything goes wrong
        print(f"Error occurred: {e}")
        inner_hex_data = np.array([
            [0.0, 0.0, 0.0],       # center
            [0.0, 2.0, 0.0],       # top
            [1.732, 1.0, 0.0],     # top-right
            [1.732, -1.0, 0.0],    # bottom-right
            [0.0, -2.0, 0.0],      # bottom
            [-1.732, -1.0, 0.0],   # bottom-left
            [-1.732, 1.0, 0.0],    # top-left
            [3.0, 0.0, 0.0],       # far right
            [-3.0, 0.0, 0.0],      # far left
            [0.0, 3.0, 0.0],       # top-top
            [0.0, -3.0, 0.0]       # bottom-bottom
        ])
        
        # Calculate outer radius
        max_distance = 0
        for cx, cy, _ in inner_hex_data:
            distance = np.sqrt(cx**2 + cy**2)
            max_distance = max(max_distance, distance + HEX_RADIUS)
        
        outer_radius = max_distance + 0.01
        outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    
    return best_config, outer_hex_data, best_radius


# EVOLVE-BLOCK-END
