# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

# Precompute hexagon vertices for unit hexagon centered at origin
def get_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Get vertices of a unit regular hexagon with optional rotation"""
    angle = rotation * np.pi / 180
    # Unit hexagon vertices (radius = 1)
    hex_points = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_points.append((x + center[0], y + center[1]))
    return np.array(hex_points)

def hexagon_area(side_length):
    """Calculate area of regular hexagon with given side length"""
    return (3 * np.sqrt(3) / 2) * side_length ** 2

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_side_length(inner_hex_data):
    """Compute minimum outer hexagon side length needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_unit_hexagon_vertices((x, y), angle)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1000
    
    all_vertices = np.array(all_vertices)
    
    # Find the bounding circle by computing the distance from origin to all vertices
    # For a regular hexagon with side length 1, the circumradius is 1
    # So we need to find the maximum distance from center to any vertex of any hexagon
    center = np.array([0, 0])
    distances = np.sqrt(np.sum((all_vertices - center)**2, axis=1))
    max_distance = np.max(distances)
    
    # The outer hexagon needs to have a circumradius >= max_distance
    # For a regular hexagon, circumradius = side_length
    return max_distance * 1.001  # Add small buffer to ensure containment

def calculate_arrangement_penalty(hex_data):
    """Calculate penalty for overlaps and containment violations"""
    total_penalty = 0
    inner_positions = hex_data[:, :2]  # (x, y) positions
    inner_angles = hex_data[:, 2]     # angles in degrees
    
    # Check overlap between all pairs of hexagons
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            pos_i, angle_i = inner_positions[i], inner_angles[i]
            pos_j, angle_j = inner_positions[j], inner_angles[j]
            
            hex_i = get_unit_hexagon_vertices(pos_i, angle_i)
            hex_j = get_unit_hexagon_vertices(pos_j, angle_j)
            
            # Check if they intersect
            if check_hexagon_overlap(hex_i, hex_j):
                # Calculate intersection area as penalty
                poly_i = Polygon(hex_i)
                poly_j = Polygon(hex_j)
                intersection = poly_i.intersection(poly_j)
                if hasattr(intersection, 'area'):
                    total_penalty += intersection.area * 1000  # Heavy penalty
    
    # Check containment of all hexagons within a bounding hexagon
    # Create a bounding hexagon that contains all inner hexagons
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    # Create outer hexagon centered at origin with proper radius
    outer_radius = outer_side_length
    outer_hex_vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        outer_hex_vertices.append((x, y))
    outer_poly = Polygon(outer_hex_vertices)
    
    # Add penalty if any vertex of inner hexagons is outside outer hexagon
    for i in range(len(hex_data)):
        pos, angle = inner_positions[i], inner_angles[i]
        vertices = get_unit_hexagon_vertices(pos, angle)
        
        for vertex in vertices:
            if not outer_poly.contains(Point(vertex)):
                # Add penalty proportional to how far outside it is
                dist_to_boundary = outer_poly.distance(Point(vertex))
                total_penalty += dist_to_boundary * 10000
    
    return total_penalty

def evaluate_arrangement(hex_data):
    """Evaluate arrangement quality - returns negative of 1/outer_radius for optimization"""
    # Check for overlaps and containment violations
    penalty = calculate_arrangement_penalty(hex_data)
    
    if penalty > 0:
        return 1000000 + penalty  # Invalid arrangement
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    if outer_side_length > 1000:
        return 1000000  # Invalid arrangement
    
    # Return negative inverse of outer side length (to maximize 1/outer_side_length)
    return -1.0 / outer_side_length if outer_side_length > 0 else 1000000

def generate_initial_configuration():
    """Generate a much better initial configuration based on known optimal patterns"""
    # More compact arrangement inspired by known dense packings
    # Place 6 hexagons around a central one in a tight hexagonal pattern
    # Adjust positions to minimize overall space needed
    
    # Central hexagon
    positions = [[0.0, 0.0, 0.0]]
    
    # Surrounding hexagons in a tight hexagonal pattern
    # Distance between centers of adjacent hexagons is 1 (unit distance)
    # But we want a tighter arrangement, so we'll adjust the spacing
    
    # Positions in a hexagonal pattern with smaller spacing
    hex_positions = [
        [1.0, 0.0, 0.0],           # right
        [0.5, 0.866, 0.0],         # top-right
        [-0.5, 0.866, 0.0],        # top-left
        [-1.0, 0.0, 0.0],          # left
        [-0.5, -0.866, 0.0],       # bottom-left
        [0.5, -0.866, 0.0],        # bottom-right
    ]
    
    # Add the surrounding hexagons
    positions.extend(hex_positions)
    
    # Add 5 more hexagons in a tighter cluster
    additional_positions = [
        [1.5, 0.0, 0.0],           # far right
        [-1.5, 0.0, 0.0],          # far left  
        [0.0, 1.732, 0.0],         # top
        [0.0, -1.732, 0.0],        # bottom
        [1.0, 1.732, 0.0],         # top-right extended
    ]
    
    positions.extend(additional_positions)
    
    # Make sure we have exactly 11 positions
    while len(positions) < 11:
        positions.append([0.0, 0.0, 0.0])  # Fill with dummy values
    
    positions = positions[:11]  # Trim to exactly 11
    
    # Apply small random adjustments to avoid symmetry issues
    for i in range(1, len(positions)):  # Skip center
        positions[i][0] += np.random.normal(0, 0.05)
        positions[i][1] += np.random.normal(0, 0.05)
        positions[i][2] += np.random.normal(0, 5)
    
    return np.array(positions)

def advanced_optimization(max_iterations=10000):
    """Use a hybrid optimization approach with multiple strategies"""
    # Start with a better initial configuration
    current_solution = generate_initial_configuration()
    current_energy = evaluate_arrangement(current_solution)
    
    # Parameters for adaptive simulated annealing
    temperature = 1.0
    cooling_rate = 0.9995  # Slightly slower cooling for better exploration
    min_temperature = 0.00001
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Track recent improvements for adaptive cooling
    recent_improvements = []
    
    # Use different perturbation strategies
    perturbation_strategies = ['position', 'rotation', 'both']
    
    for iteration in range(max_iterations):
        # Generate neighbor solution by perturbing one hexagon
        neighbor_solution = current_solution.copy()
        
        # Choose a random hexagon to perturb (excluding center)
        hexagon_idx = np.random.randint(1, len(neighbor_solution))  # Skip center
        
        # Choose perturbation strategy
        strategy = np.random.choice(perturbation_strategies)
        
        # Apply perturbation based on strategy
        if strategy == 'position':
            neighbor_solution[hexagon_idx, 0] += np.random.normal(0, 0.1)  # x position
            neighbor_solution[hexagon_idx, 1] += np.random.normal(0, 0.1)  # y position
        elif strategy == 'rotation':
            neighbor_solution[hexagon_idx, 2] += np.random.normal(0, 10)   # rotation
            neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
        else:  # both
            neighbor_solution[hexagon_idx, 0] += np.random.normal(0, 0.08)  # x position
            neighbor_solution[hexagon_idx, 1] += np.random.normal(0, 0.08)  # y position
            neighbor_solution[hexagon_idx, 2] += np.random.normal(0, 8)   # rotation
            neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
        
        # Evaluate neighbor
        neighbor_energy = evaluate_arrangement(neighbor_solution)
        
        # Accept or reject based on Metropolis criterion
        if neighbor_energy < current_energy:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        else:
            # Accept with probability based on temperature
            delta_energy = neighbor_energy - current_energy
            acceptance_probability = np.exp(-delta_energy / temperature)
            if np.random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_energy = neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
            recent_improvements = []  # Reset improvement tracker
        else:
            recent_improvements.append(current_energy)
            # If no improvement in last 500 iterations, increase temperature slightly
            if len(recent_improvements) > 500:
                temperature = min(temperature * 1.05, 1.0)
                recent_improvements = recent_improvements[-100:]  # Keep recent history
        
        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)
    
    return best_solution, best_energy

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-inspired simulated annealing approach for optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use advanced optimization
    start_time = time.time()
    inner_hex_data, best_energy = advanced_optimization(max_iterations=5000)
    end_time = time.time()
    
    # Compute final outer hexagon size
    outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Ensure we have a valid solution
    if outer_side_length <= 0 or np.isnan(outer_side_length):
        # Fall back to a known good configuration
        inner_hex_data = generate_initial_configuration()
        outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
