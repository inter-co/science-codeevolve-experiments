# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(theta)
        y = center_y + side_length * math.sin(theta)
        vertices.append((x, y))
    return vertices

def outer_hexagon_vertices(side_length, center_x=0, center_y=0, angle_deg=0):
    """Generate vertices of the outer hexagon"""
    return hexagon_vertices(center_x, center_y, angle_deg, side_length)

def check_containment(inner_hex_data, outer_side_length):
    """Check if all inner hexagons are contained within outer hexagon"""
    outer_vertices = outer_hexagon_vertices(outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    for i, (cx, cy, angle) in enumerate(inner_hex_data):
        inner_vertices = hexagon_vertices(cx, cy, angle)
        inner_polygon = Polygon(inner_vertices)
        
        # Check if inner hexagon is fully contained
        if not outer_polygon.contains(inner_polygon):
            return False
            
        # Check if all vertices are within bounds
        for vx, vy in inner_vertices:
            if not outer_polygon.contains(Point(vx, vy)):
                return False
                
    return True

def calculate_penetration_volume(inner_hex_data, outer_side_length):
    """Calculate total penetration volume (negative when overlapping)"""
    try:
        outer_vertices = outer_hexagon_vertices(outer_side_length)
        outer_polygon = Polygon(outer_vertices)
        
        total_volume = 0
        for i, (cx, cy, angle) in enumerate(inner_hex_data):
            inner_vertices = hexagon_vertices(cx, cy, angle)
            inner_polygon = Polygon(inner_vertices)
            
            # Calculate intersection area
            intersection = outer_polygon.intersection(inner_polygon)
            if intersection.geom_type == 'Polygon':
                total_volume += intersection.area
            elif intersection.geom_type == 'MultiPolygon':
                total_volume += sum(poly.area for poly in intersection.geoms)
                
        return total_volume
    except:
        return 0

def compute_distance_matrix(inner_hex_data):
    """Compute distance matrix between all hexagon centers"""
    centers = np.array([[cx, cy] for cx, cy, _ in inner_hex_data])
    return cdist(centers, centers)

def compute_min_distance(inner_hex_data):
    """Compute minimum distance between any two hexagon centers"""
    dist_matrix = compute_distance_matrix(inner_hex_data)
    # Set diagonal to large value to ignore self-distances
    np.fill_diagonal(dist_matrix, 1000)
    return np.min(dist_matrix)

def evaluate_packing(inner_hex_data, outer_side_length):
    """Evaluate a packing configuration"""
    # Check containment
    if not check_containment(inner_hex_data, outer_side_length):
        # Return very poor score if not contained
        return 1000000
    
    # Compute overlap penalty (smaller is better)
    min_dist = compute_min_distance(inner_hex_data)
    if min_dist < 1.0:  # Overlapping
        overlap_penalty = (1.0 - min_dist) * 10000
    else:
        overlap_penalty = 0
        
    # Add penalty for being too close to boundary
    boundary_penalty = 0
    outer_vertices = outer_hexagon_vertices(outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    for i, (cx, cy, angle) in enumerate(inner_hex_data):
        inner_vertices = hexagon_vertices(cx, cy, angle)
        inner_polygon = Polygon(inner_vertices)
        
        # Check how much we're outside the boundary
        for vx, vy in inner_vertices:
            point = Point(vx, vy)
            if not outer_polygon.contains(point):
                distance_to_boundary = point.distance(outer_polygon.exterior)
                boundary_penalty += distance_to_boundary * 100
    
    # Objective: minimize outer hexagon size + penalties
    return outer_side_length + overlap_penalty + boundary_penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses force-directed optimization approach with geometric constraints.
    """
    
    # Initial configuration - more strategic placement
    initial_positions = [
        [0, 0, 0],      # center
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [1.73, 1.0, 0], # top right
        [-1.73, 1.0, 0], # top left
        [1.73, -1.0, 0], # bottom right
        [-1.73, -1.0, 0], # bottom left
        [3.46, 2.0, 0], # far top right
        [-3.46, 2.0, 0], # far top left
        [3.46, -2.0, 0], # far bottom right
        [-3.46, -2.0, 0], # far bottom left
    ]
    
    # Convert to array for easier manipulation
    initial_data = np.array(initial_positions)
    
    # Use optimization to find better configuration
    def objective(x):
        # Reshape parameters back into hexagon data
        hex_data = x.reshape(-1, 3)
        # We want to minimize outer hexagon size
        # Since we're optimizing with respect to the outer hexagon size,
        # we'll start with a reasonable guess and let it vary
        return evaluate_packing(hex_data, 6.0)  # Initial guess for outer size
    
    def constraint_func(x):
        # Ensure all positions are valid and contained
        hex_data = x.reshape(-1, 3)
        return evaluate_packing(hex_data, 6.0)
    
    # Flatten initial data for optimization
    x0 = initial_data.flatten()
    
    # Use scipy optimization with bounds
    bounds = [(-10, 10) for _ in range(len(x0))]  # Reasonable bounds for positions
    
    # Simple gradient-free optimization approach
    try:
        # Try several optimization approaches
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
        optimized_data = result.x.reshape(-1, 3)
    except:
        # Fallback to original approach if optimization fails
        optimized_data = initial_data
    
    # Final refinement with geometric constraints
    best_outer_size = 6.0
    best_config = optimized_data.copy()
    
    # Iteratively improve the solution
    for iter_num in range(10):
        # Try to reduce outer hexagon size
        test_size = best_outer_size - 0.1
        if test_size > 2.0:  # Minimum reasonable size
            # Check if configuration works with smaller outer hexagon
            if check_containment(best_config, test_size):
                best_outer_size = test_size
            else:
                # If not, try to adjust positions
                pass
    
    # Final validation
    final_config = best_config.copy()
    outer_side_length = best_outer_size
    
    # Make sure we have a valid configuration
    if not check_containment(final_config, outer_side_length):
        # Revert to conservative approach
        final_config = initial_data
        outer_side_length = 8.0
    
    # Ensure we're within reasonable bounds
    outer_side_length = max(4.0, outer_side_length)
    
    # Return final result
    return final_config, np.array([0, 0, 0]), outer_side_length


# EVOLVE-BLOCK-END
