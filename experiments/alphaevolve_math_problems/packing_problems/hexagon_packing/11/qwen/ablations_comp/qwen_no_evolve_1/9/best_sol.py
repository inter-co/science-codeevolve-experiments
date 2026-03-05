# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import math
from typing import Tuple, List

def create_regular_hexagon(center: Tuple[float, float], radius: float, rotation: float = 0) -> Polygon:
    """Create a regular hexagon with given center, radius, and rotation."""
    points = []
    for i in range(6):
        angle = rotation + i * np.pi / 3
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_hexagon_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if a hexagon is fully contained within the outer hexagon."""
    return outer_hex.contains(hexagon) or outer_hex.intersects(hexagon)

def calculate_hexagon_area(radius: float) -> float:
    """Calculate area of regular hexagon with given radius."""
    return (3 * np.sqrt(3) / 2) * radius * radius

def get_hexagon_vertices(center: Tuple[float, float], radius: float, rotation: float = 0) -> List[Tuple[float, float]]:
    """Get vertices of a regular hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation + i * np.pi / 3
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

def compute_outer_hexagon_radius(inner_hexagons: List[Tuple[Tuple[float, float], float]], min_distance: float = 0.01) -> float:
    """Compute minimum radius needed for outer hexagon to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (center, rotation) in inner_hexagons:
        vertices = get_hexagon_vertices(center, 1.0, rotation)
        all_vertices.extend(vertices)
    
    # Find the bounding box
    if not all_vertices:
        return 1.0
    
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Compute distance from center to farthest vertex
    max_dist = 0
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    for x, y in all_vertices:
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add some buffer for the outer hexagon
    return max_dist + 1.0

def is_collision_free(hexagons: List[Tuple[Tuple[float, float], float]]) -> bool:
    """Check if all hexagons are collision-free."""
    polygons = []
    for (center, rotation) in hexagons:
        hex_poly = create_regular_hexagon(center, 1.0, rotation)
        polygons.append(hex_poly)
    
    # Check pairwise intersections
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                return False
    return True

def generate_initial_config() -> List[Tuple[Tuple[float, float], float]]:
    """Generate an initial configuration of 11 hexagons."""
    # Start with a more sophisticated layout
    configs = [
        # Center hexagon
        ((0, 0), 0),
        # Surrounding hexagons in 2 layers
        ((2, 0), 0),
        ((-2, 0), 0),
        ((1, np.sqrt(3)), 0),
        ((-1, np.sqrt(3)), 0),
        ((1, -np.sqrt(3)), 0),
        ((-1, -np.sqrt(3)), 0),
        ((3, np.sqrt(3)), 0),
        ((-3, np.sqrt(3)), 0),
        ((3, -np.sqrt(3)), 0),
        ((-3, -np.sqrt(3)), 0),
    ]
    
    # Randomize rotations slightly
    for i in range(len(configs)):
        configs[i] = (configs[i][0], random.uniform(0, 360))
    
    return configs

def simulated_annealing_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Use simulated annealing to find optimal hexagon packing."""
    # Initial configuration
    current_config = generate_initial_config()
    best_config = current_config[:]
    
    # Initialize parameters
    temp = 1000.0
    cooling_rate = 0.999
    min_temp = 1e-8
    iterations = 0
    max_iterations = 50000
    
    # Best objective value found so far
    best_radius = compute_outer_hexagon_radius(best_config)
    best_inv_radius = 1.0 / best_radius
    
    # Main annealing loop
    while temp > min_temp and iterations < max_iterations:
        # Create neighbor solution by perturbing one hexagon
        neighbor_config = current_config[:]
        
        # Choose a random hexagon to perturb (excluding center)
        idx = random.randint(1, len(neighbor_config) - 1)
        
        # Perturb position and rotation
        center, rotation = neighbor_config[idx]
        new_center = (
            center[0] + random.uniform(-0.5, 0.5),
            center[1] + random.uniform(-0.5, 0.5)
        )
        new_rotation = rotation + random.uniform(-30, 30)
        
        neighbor_config[idx] = (new_center, new_rotation)
        
        # Check if neighbor is valid (collision-free)
        if is_collision_free(neighbor_config):
            # Calculate radius for neighbor
            neighbor_radius = compute_outer_hexagon_radius(neighbor_config)
            neighbor_inv_radius = 1.0 / neighbor_radius
            
            # Accept or reject based on Metropolis criterion
            if neighbor_inv_radius > best_inv_radius or \
               random.random() < math.exp((neighbor_inv_radius - best_inv_radius) / temp):
                current_config = neighbor_config[:]
                if neighbor_inv_radius > best_inv_radius:
                    best_config = neighbor_config[:]
                    best_inv_radius = neighbor_inv_radius
                    best_radius = neighbor_radius
        
        temp *= cooling_rate
        iterations += 1
    
    # Convert best configuration to desired output format
    inner_hex_data = np.zeros((11, 3))
    for i, (center, rotation) in enumerate(best_config):
        inner_hex_data[i] = [center[0], center[1], rotation]
    
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    outer_hex_side_length = best_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-based optimization approach with simulated annealing.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use simulated annealing to find better configuration
    return simulated_annealing_hexagon_packing()


# EVOLVE-BLOCK-END
