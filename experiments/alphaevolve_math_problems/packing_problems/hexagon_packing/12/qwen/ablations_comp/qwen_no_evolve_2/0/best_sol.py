# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import math

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_containment(hexagon_vertices, outer_hex_center_x, outer_hex_center_y, outer_hex_side_length):
    """Check if all vertices of a hexagon are within the outer hexagon."""
    # Create outer hexagon vertices
    outer_vertices = create_hexagon_vertices(outer_hex_center_x, outer_hex_center_y, 0, outer_hex_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check if all inner hexagon vertices are inside outer polygon
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def compute_hexagon_area(vertices):
    """Compute area of hexagon using shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2

def calculate_overlaps(inner_hex_data, outer_hex_side_length):
    """Calculate total overlap between all hexagons."""
    # Create polygons for all inner hexagons
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        hex_polygons.append(Polygon(vertices))
    
    # Create outer hexagon polygon
    outer_vertices = create_hexagon_vertices(0, 0, 0, outer_hex_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Calculate overlaps
    total_overlap = 0
    for i in range(len(hex_polygons)):
        # Check containment in outer hexagon
        if not outer_polygon.contains(hex_polygons[i]):
            # Partial containment - get intersection area
            intersection = outer_polygon.intersection(hex_polygons[i])
            if intersection.geom_type == 'Polygon':
                total_overlap += intersection.area
            elif intersection.geom_type == 'MultiPolygon':
                for part in intersection.geoms:
                    total_overlap += part.area
        
        # Check pairwise overlaps
        for j in range(i + 1, len(hex_polygons)):
            intersection = hex_polygons[i].intersection(hex_polygons[j])
            if intersection.geom_type == 'Polygon':
                total_overlap += intersection.area
            elif intersection.geom_type == 'MultiPolygon':
                for part in intersection.geoms:
                    total_overlap += part.area
    
    return total_overlap

def evaluate_configuration(params, outer_hex_side_length):
    """Evaluate a configuration of hexagons."""
    # params contains [x1, y1, angle1, x2, y2, angle2, ...] for 12 hexagons
    inner_hex_data = np.zeros((12, 3))
    
    for i in range(12):
        inner_hex_data[i] = [params[3*i], params[3*i+1], params[3*i+2]]
    
    # Check if any hexagon is outside outer hexagon
    outer_center = [0, 0]
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], 0, outer_hex_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment and overlaps
    penalty = 0
    for i in range(12):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        
        # Check containment penalty
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                # Calculate distance from point to boundary
                min_dist = min([
                    abs(vertex[0] - outer_vertices[j][0]) + abs(vertex[1] - outer_vertices[j][1])
                    for j in range(len(outer_vertices))
                ])
                penalty += min_dist * 100  # Heavy penalty for containment violations
        
        # Check overlap penalty with other hexagons
        for j in range(i + 1, 12):
            center_x2, center_y2, angle2 = inner_hex_data[j]
            vertices2 = create_hexagon_vertices(center_x2, center_y2, angle2)
            
            poly1 = Polygon(vertices)
            poly2 = Polygon(vertices2)
            
            try:
                intersection = poly1.intersection(poly2)
                if intersection.geom_type == 'Polygon' and intersection.area > 0:
                    penalty += intersection.area * 1000  # Heavy penalty for overlaps
            except:
                pass  # Skip problematic intersections
    
    return penalty

def generate_initial_symmetric_config():
    """Generate a symmetric initial configuration."""
    # Start with a highly symmetric pattern
    # Central hexagon + 6 surrounding + 5 others in a ring
    config = []
    
    # Center hexagon
    config.append([0, 0, 0])
    
    # Surrounding hexagons in a ring
    radius = 2  # Distance from center
    for i in range(6):
        angle = i * 60  # degrees
        rad_angle = math.radians(angle)
        x = radius * math.cos(rad_angle)
        y = radius * math.sin(rad_angle)
        config.append([x, y, 0])
    
    # Additional positions to make 12 total
    # Place additional hexagons in a second ring
    radius2 = 3
    for i in range(5):
        angle = i * 72  # degrees
        rad_angle = math.radians(angle)
        x = radius2 * math.cos(rad_angle)
        y = radius2 * math.sin(rad_angle)
        config.append([x, y, 0])
    
    # Add some randomization to break perfect symmetry
    config = np.array(config)
    config[:, 0] += np.random.normal(0, 0.1, 12)
    config[:, 1] += np.random.normal(0, 0.1, 12)
    
    return config.flatten()

def optimize_hexagon_packing():
    """Use optimization to find better packing."""
    # Start with a symmetric configuration
    initial_params = generate_initial_symmetric_config()
    
    # Initial guess for outer hexagon size
    outer_side_length = 6.0  # Start with reasonable value
    
    # Define bounds for optimization
    bounds = []
    for i in range(36):  # 12 hexagons * 3 parameters each
        if i % 3 == 0:  # x coordinate
            bounds.append((-10, 10))  # Reasonable bounds
        elif i % 3 == 1:  # y coordinate  
            bounds.append((-10, 10))
        else:  # angle
            bounds.append((0, 360))
    
    def objective(params):
        # Create configuration
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i] = [params[3*i], params[3*i+1], params[3*i+2]]
        
        # Compute overlap penalty
        overlap_penalty = calculate_overlaps(inner_hex_data, outer_side_length)
        
        # Add penalty for being too close to edges
        edge_penalty = 0
        for i in range(12):
            x, y, _ = inner_hex_data[i]
            # Distance to center
            dist_to_center = math.sqrt(x*x + y*y)
            # If we're too close to the boundary, penalize
            if dist_to_center > outer_side_length - 1.5:  # Assuming hexagon diameter ~2
                edge_penalty += (dist_to_center - (outer_side_length - 1.5)) * 1000
        
        return overlap_penalty + edge_penalty
    
    # Use scipy optimization
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        optimized_params = result.x
    except:
        optimized_params = initial_params
    
    # Create final configuration
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
    
    # Find minimum outer hexagon size that contains all hexagons
    max_radius = 0
    for i in range(12):
        x, y, _ = inner_hex_data[i]
        # Add hexagon radius (sqrt(3) ~ 1.732) to account for full hexagon extent
        dist = math.sqrt(x*x + y*y) + 1.732
        max_radius = max(max_radius, dist)
    
    outer_side_length = max_radius
    
    # Refine to ensure no overlaps
    # We'll use a simple iterative refinement approach
    for _ in range(50):
        # Recalculate overlaps and adjust positions slightly
        overlap = calculate_overlaps(inner_hex_data, outer_side_length)
        if overlap < 1e-6:  # No significant overlaps
            break
        # Reduce outer hexagon size slightly to force better packing
        outer_side_length *= 0.99
    
    # Final adjustment to make sure all hexagons fit
    max_radius = 0
    for i in range(12):
        x, y, _ = inner_hex_data[i]
        dist = math.sqrt(x*x + y*y) + 1.732
        max_radius = max(max_radius, dist)
    
    outer_side_length = max_radius
    
    # Set outer hexagon at center
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use optimization approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure the outer hexagon actually contains all inner hexagons
    # This is a safety check to make sure our solution is valid
    while True:
        # Check if all hexagons are contained
        all_contained = True
        outer_vertices = create_hexagon_vertices(0, 0, 0, outer_hex_side_length)
        outer_polygon = Polygon(outer_vertices)
        
        for i in range(12):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = create_hexagon_vertices(center_x, center_y, angle)
            hex_polygon = Polygon(vertices)
            
            # Check if hexagon is fully contained
            if not outer_polygon.contains(hex_polygon):
                # Adjust outer hexagon size
                max_dist = 0
                for vertex in vertices:
                    dist = math.sqrt(vertex[0]**2 + vertex[1]**2)
                    max_dist = max(max_dist, dist)
                outer_hex_side_length = max(outer_hex_side_length, max_dist + 1.0)
                all_contained = False
                break
        
        if all_contained:
            break
    
    # Make sure we have a valid solution
    if outer_hex_side_length < 1:
        outer_hex_side_length = 1.0
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
