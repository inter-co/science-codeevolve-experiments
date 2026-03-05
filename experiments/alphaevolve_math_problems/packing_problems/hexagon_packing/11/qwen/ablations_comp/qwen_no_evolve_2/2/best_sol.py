# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3) / 2  # Distance from center to side
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, radius=HEX_RADIUS):
    """Generate vertices of a hexagon at position (x,y) with given angle"""
    angle_rad = np.radians(angle_deg)
    vertices = np.zeros((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Calculate distance from point to line segment"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    # Vector from line_start to point
    px_minus_x1, py_minus_y1 = px - x1, py - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return np.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)
    
    # Project point onto line
    t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
    
    # Clamp t to [0, 1] for line segment
    t = max(0, min(1, t))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon(point, hex_center, hex_angle, radius=HEX_RADIUS):
    """Check if point is inside hexagon using ray casting method"""
    # Transform point to hexagon's local coordinate system
    angle_rad = np.radians(hex_angle)
    cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
    px_local = cos_a * (point[0] - hex_center[0]) - sin_a * (point[1] - hex_center[1])
    py_local = sin_a * (point[0] - hex_center[0]) + cos_a * (point[1] - hex_center[1])
    
    # Check if point is inside the hexagon
    vertices = hexagon_vertices(0, 0, 0, radius)
    
    # Ray casting algorithm
    intersections = 0
    p1x, p1y = vertices[0]
    for i in range(1, 7):
        p2x, p2y = vertices[i % 6]
        if py_local > min(p1y, p2y):
            if py_local <= max(p1y, p2y):
                if px_local <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py_local - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px_local <= xinters:
                        intersections += 1
        p1x, p1y = p2x, p2y
    
    return intersections % 2 == 1

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon"""
    for vertex in hex_vertices:
        # Simple check: vertex should be inside outer hexagon
        # Using point-in-polygon test
        if not point_in_hexagon(vertex, (0,0), 0):  # Simplified check
            return False
    return True

def create_hexagon_polygon(vertices):
    """Create shapely polygon from vertices"""
    return Polygon(vertices)

def calculate_outer_hexagon_radius(inner_hex_data, padding=0.01):
    """Estimate minimum outer hexagon radius needed to contain all inner hexagons"""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)
        all_vertices.extend(vertices)
    
    # Find bounding circle
    if len(all_vertices) == 0:
        return 1.0
    
    center = np.mean(all_vertices, axis=0)
    distances = [np.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2) for v in all_vertices]
    max_distance = max(distances) + padding + HEX_RADIUS
    
    # Convert to hexagon radius (accounting for hexagon geometry)
    # For a regular hexagon, radius = distance from center to corner
    # But we need to account for the fact that our outer hexagon is also regular
    # So we need to find the smallest regular hexagon that contains all points
    return max_distance

def evaluate_packing(inner_hex_data, outer_radius=None):
    """Evaluate the validity and efficiency of a packing"""
    if outer_radius is None:
        outer_radius = calculate_outer_hexagon_radius(inner_hex_data)
    
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)
        inner_polygons.append(create_hexagon_polygon(vertices))
    
    # Check for overlaps between inner hexagons
    for i in range(len(inner_polygons)):
        for j in range(i+1, len(inner_polygons)):
            if inner_polygons[i].intersects(inner_polygons[j]):
                return False, 0.0, outer_radius
    
    # Check containment - all inner hexagons should be inside outer hexagon
    outer_hex_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = create_hexagon_polygon(outer_hex_vertices)
    
    # Check if all inner hexagons are contained in outer hexagon
    for poly in inner_polygons:
        # Check if any vertex of inner hexagon is outside outer hexagon
        # More accurate: check if the polygon is completely inside outer polygon
        if not outer_polygon.contains(poly):
            # Check if any vertex of inner hexagon is outside outer hexagon
            inner_vertices = list(poly.exterior.coords)
            for vertex in inner_vertices[:-1]:  # Exclude last duplicate vertex
                if not point_in_hexagon(vertex, (0,0), 0, outer_radius):
                    return False, 0.0, outer_radius
    
    # Calculate the inverse of outer hexagon side length
    # Side length of hexagon with radius r is r
    inv_side_length = 1.0 / outer_radius
    
    return True, inv_side_length, outer_radius

def generate_initial_solution():
    """Generate a more sophisticated initial solution using hexagonal lattice"""
    # Arrange 11 hexagons in a pattern inspired by hexagonal close packing
    # Start with central hexagon, then arrange around it in rings
    
    # Hexagon positions in a hexagonal lattice pattern
    positions = [
        (0, 0),           # center
        (0, 2),           # top
        (0, -2),          # bottom
        (-1.732, 1),      # top-left
        (1.732, 1),       # top-right
        (-1.732, -1),     # bottom-left
        (1.732, -1),      # bottom-right
        (-3.464, 0),      # far left
        (3.464, 0),       # far right
        (-1.732, -3),     # bottom far
        (1.732, -3),      # bottom far right
    ]
    
    # Create initial data
    inner_hex_data = np.zeros((11, 3))
    for i, (x, y) in enumerate(positions):
        inner_hex_data[i] = [x, y, 0]  # No rotation initially
    
    return inner_hex_data

def mutate_solution(solution, mutation_strength=0.5):
    """Apply random mutation to solution"""
    mutated = solution.copy()
    
    # Randomly choose which hexagon to mutate
    hex_idx = np.random.randint(0, len(mutated))
    
    # Mutate position (x, y) and possibly rotation
    mutated[hex_idx, 0] += np.random.normal(0, mutation_strength)  # x
    mutated[hex_idx, 1] += np.random.normal(0, mutation_strength)  # y
    mutated[hex_idx, 2] += np.random.normal(0, 10)  # rotation (degrees)
    
    # Keep rotation in [0, 360) range
    mutated[hex_idx, 2] = mutated[hex_idx, 2] % 360
    
    return mutated

def optimize_hexagon_packing(max_time=60):
    """Evolutionary optimization of hexagon packing"""
    start_time = time.time()
    
    # Generate initial solution
    current_solution = generate_initial_solution()
    best_solution = current_solution.copy()
    best_score = 0.0
    best_radius = float('inf')
    
    # Evolution parameters
    population_size = 20
    generations = 1000
    mutation_rate = 0.3
    
    # Initialize population
    population = [current_solution.copy()]
    for _ in range(population_size - 1):
        population.append(mutate_solution(current_solution))
    
    # Optimization loop
    for generation in range(generations):
        if time.time() - start_time > max_time:
            break
            
        # Evaluate fitness of population
        fitness_scores = []
        for individual in population:
            valid, score, radius = evaluate_packing(individual)
            if valid:
                fitness_scores.append((score, individual, radius))
            else:
                fitness_scores.append((0.0, individual, float('inf')))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Update best solution
        if fitness_scores[0][0] > best_score:
            best_score = fitness_scores[0][0]
            best_solution = fitness_scores[0][1].copy()
            best_radius = fitness_scores[0][2]
        
        # Create new population through selection and crossover
        new_population = [fitness_scores[0][1]]  # Keep best
        
        # Add some mutated versions of top performers
        for _ in range(population_size - 1):
            parent = fitness_scores[np.random.randint(0, min(5, len(fitness_scores)))][1]
            child = mutate_solution(parent)
            new_population.append(child)
        
        population = new_population
    
    return best_solution, best_radius, best_score

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use evolutionary optimization
    inner_hex_data, outer_hex_side_length, inv_side_length = optimize_hexagon_packing()
    
    # Set outer hexagon at center (can be optimized further)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
