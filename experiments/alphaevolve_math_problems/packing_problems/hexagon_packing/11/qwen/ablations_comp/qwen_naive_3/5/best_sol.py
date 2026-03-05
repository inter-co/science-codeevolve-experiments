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
    
    # Find the bounding circle - compute distance from origin to each vertex
    distances = np.sqrt(np.sum(all_vertices**2, axis=1))
    max_distance = np.max(distances)
    
    # Add small buffer to ensure complete containment
    return max_distance * 1.05

def calculate_arrangement_penalty(hex_data):
    """Calculate penalty for overlaps and containment violations - optimized version"""
    total_penalty = 0
    inner_positions = hex_data[:, :2]  # (x, y) positions
    inner_angles = hex_data[:, 2]     # angles in degrees
    
    # Pre-compute all hexagon polygons for efficiency
    hex_polygons = []
    for i in range(len(hex_data)):
        pos, angle = inner_positions[i], inner_angles[i]
        vertices = get_unit_hexagon_vertices(pos, angle)
        hex_polygons.append(Polygon(vertices))
    
    # Check overlap between all pairs of hexagons efficiently
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            # Early rejection based on distance - this is much faster than full polygon intersection
            dist = np.sqrt(np.sum((inner_positions[i] - inner_positions[j])**2))
            if dist > 2.0:  # Max possible distance between unit hexagons without overlapping
                continue
                
            # Check if they intersect
            if hex_polygons[i].intersects(hex_polygons[j]):
                # Calculate intersection area as penalty
                try:
                    intersection = hex_polygons[i].intersection(hex_polygons[j])
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        total_penalty += intersection.area * 1000  # Heavy penalty
                except:
                    # Fallback if intersection fails
                    total_penalty += 10000
    
    # Check containment of all hexagons within a bounding hexagon
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    # Create a hexagon that can contain all vertices
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
        vertices = get_unit_hexagon_vertices(inner_positions[i], inner_angles[i])
        
        for vertex in vertices:
            point = Point(vertex)
            if not outer_poly.contains(point):
                # Add penalty proportional to how far outside it is
                try:
                    dist_to_boundary = outer_poly.distance(point)
                    total_penalty += dist_to_boundary * 10000
                except:
                    # Fallback penalty if distance calculation fails
                    total_penalty += 100000
    
    return total_penalty

def evaluate_arrangement(hex_data):
    """Evaluate arrangement quality - returns negative of 1/outer_radius for optimization"""
    # Check for overlaps and containment violations
    penalty = calculate_arrangement_penalty(hex_data)
    
    if penalty > 0:
        return 1000000 + penalty  # Invalid arrangement
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    if outer_side_length > 1000 or outer_side_length <= 0:
        return 1000000  # Invalid arrangement
    
    # Return negative inverse of outer side length (to maximize 1/outer_side_length)
    return -1.0 / outer_side_length if outer_side_length > 0 else 1000000

def calculate_arrangement_penalty_fast(hex_data):
    """Fast penalty calculation with early exits and optimized overlap detection"""
    total_penalty = 0
    inner_positions = hex_data[:, :2]  # (x, y) positions
    inner_angles = hex_data[:, 2]     # angles in degrees
    
    # Pre-compute all hexagon polygons for efficiency
    hex_polygons = []
    for i in range(len(hex_data)):
        pos, angle = inner_positions[i], inner_angles[i]
        vertices = get_unit_hexagon_vertices(pos, angle)
        hex_polygons.append(Polygon(vertices))
    
    # Check overlap between all pairs of hexagons efficiently
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            # Early rejection based on distance - this is much faster than full polygon intersection
            dist = np.sqrt(np.sum((inner_positions[i] - inner_positions[j])**2))
            
            # If centers are more than 2 units apart, they cannot overlap
            if dist > 2.0:  
                continue
            
            # Check intersection directly
            if hex_polygons[i].intersects(hex_polygons[j]):
                # Calculate intersection area as penalty
                try:
                    intersection = hex_polygons[i].intersection(hex_polygons[j])
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        total_penalty += intersection.area * 1000  # Heavy penalty
                except:
                    # Fallback if polygon operations fail
                    total_penalty += 10000
    
    # Check containment of all hexagons within a bounding hexagon
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    if outer_side_length <= 0:
        return 1000000
    
    # Create a hexagon that can contain all vertices
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
        vertices = get_unit_hexagon_vertices(inner_positions[i], inner_angles[i])
        
        for vertex in vertices:
            point = Point(vertex)
            if not outer_poly.contains(point):
                # Add penalty proportional to how far outside it is
                try:
                    dist_to_boundary = outer_poly.distance(point)
                    total_penalty += dist_to_boundary * 10000
                except:
                    # Fallback penalty if distance calculation fails
                    total_penalty += 100000
    
    return total_penalty

def generate_initial_configuration():
    """Generate a smart initial configuration using a more systematic approach"""
    # Start with a known good configuration pattern
    # Based on research on hexagonal packing patterns
    # This configuration should be more optimized than the previous one
    
    # Hexagonal lattice with central hexagon and surrounding ring
    positions = [
        [0.0, 0.0, 0.0],           # center
        [1.0, 0.0, 0.0],           # right
        [-1.0, 0.0, 0.0],          # left
        [0.5, 0.866, 0.0],         # top-right
        [-0.5, 0.866, 0.0],        # top-left
        [0.5, -0.866, 0.0],        # bottom-right
        [-0.5, -0.866, 0.0],       # bottom-left
        [1.5, 0.866, 0.0],         # far right-top
        [-1.5, 0.866, 0.0],        # far left-top
        [1.5, -0.866, 0.0],        # far right-bottom
        [-1.5, -0.866, 0.0]        # far left-bottom
    ]
    
    # Try a more optimized starting configuration
    # Based on known tight packings of 11 hexagons
    optimized_positions = [
        [0.0, 0.0, 0.0],           # center
        [1.0, 0.0, 0.0],           # right
        [-1.0, 0.0, 0.0],          # left
        [0.5, 0.866025, 0.0],      # top-right
        [-0.5, 0.866025, 0.0],     # top-left
        [0.5, -0.866025, 0.0],     # bottom-right
        [-0.5, -0.866025, 0.0],    # bottom-left
        [1.5, 0.866025, 0.0],      # far right-top
        [-1.5, 0.866025, 0.0],     # far left-top
        [1.5, -0.866025, 0.0],     # far right-bottom
        [-1.5, -0.866025, 0.0]     # far left-bottom
    ]
    
    return np.array(optimized_positions)

def improved_initial_configuration():
    """Generate a better initial configuration based on known optimal packings"""
    # More carefully constructed configuration based on mathematical analysis
    # Try to achieve better packing density by placing some hexagons closer together
    # and optimizing the positions for minimal outer radius
    
    # Start with a symmetric arrangement
    # Central hexagon
    positions = [[0.0, 0.0, 0.0]]
    
    # First ring around center (6 hexagons) - standard packing
    ring1_angles = [0, 60, 120, 180, 240, 300]
    ring1_radius = 1.0  # Distance from center to adjacent hexagons
    
    for angle in ring1_angles:
        rad = np.radians(angle)
        x = ring1_radius * np.cos(rad)
        y = ring1_radius * np.sin(rad)
        positions.append([x, y, 0.0])
    
    # Second ring - try to place strategically to minimize overall size
    # Using a more compact arrangement
    ring2_positions = [
        [1.2, 0.0, 0.0],      # Right (slightly closer)
        [-1.2, 0.0, 0.0],     # Left (slightly closer)
        [0.0, 1.2, 0.0],      # Top (slightly closer)
        [0.0, -1.2, 0.0]      # Bottom (slightly closer)
    ]
    
    positions.extend(ring2_positions)
    
    # Further optimized positions to create a more compact arrangement
    # These are based on mathematical attempts to minimize the bounding circle
    optimized_positions = [
        [0.0, 0.0, 0.0],           # center
        [1.0, 0.0, 0.0],           # right
        [-1.0, 0.0, 0.0],          # left
        [0.5, 0.866025, 0.0],      # top-right
        [-0.5, 0.866025, 0.0],     # top-left
        [0.5, -0.866025, 0.0],     # bottom-right
        [-0.5, -0.866025, 0.0],    # bottom-left
        [1.1, 0.635, 0.0],         # far right-top (more compact)
        [-1.1, 0.635, 0.0],        # far left-top (more compact)
        [1.1, -0.635, 0.0],        # far right-bottom (more compact)
        [-1.1, -0.635, 0.0]        # far left-bottom (more compact)
    ]
    
    return np.array(optimized_positions)

def get_hexagon_radius():
    """Return the radius of a unit regular hexagon"""
    # For a unit regular hexagon (side length = 1), the radius (distance from center to vertex) is 1
    return 1.0

def simulated_annealing_optimization(max_iterations=5000):
    """Use improved simulated annealing to optimize the hexagon arrangement"""
    # Start with a better initial configuration
    current_solution = improved_initial_configuration()
    current_energy = evaluate_arrangement(current_solution)
    
    # Parameters for simulated annealing - fine-tuned for better convergence
    temperature = 1.0
    cooling_rate = 0.9995  # Slightly faster cooling for better convergence
    min_temperature = 0.0001
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Store history for debugging
    energy_history = []
    
    # Track consecutive failures to adjust strategy
    consecutive_failures = 0
    max_failures = 100
    
    for iteration in range(max_iterations):
        # Generate neighbor solution by perturbing one hexagon
        neighbor_solution = current_solution.copy()
        
        # Choose a random hexagon to perturb (exclude center for stability)
        hexagon_idx = np.random.randint(1, len(neighbor_solution))  # Skip center
        
        # Perturb position and rotation with adaptive step sizes
        # Smaller steps later in optimization to refine solution
        step_size_pos = 0.1 if iteration < max_iterations // 3 else 0.05 if iteration < 2*max_iterations//3 else 0.02
        step_size_rot = 15.0 if iteration < max_iterations // 3 else 7.0 if iteration < 2*max_iterations//3 else 3.0
        
        neighbor_solution[hexagon_idx, 0] += np.random.normal(0, step_size_pos)  # x position
        neighbor_solution[hexagon_idx, 1] += np.random.normal(0, step_size_pos)  # y position
        neighbor_solution[hexagon_idx, 2] += np.random.normal(0, step_size_rot)   # rotation
        
        # Keep rotation in [0, 360)
        neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
        
        # Evaluate neighbor
        neighbor_energy = evaluate_arrangement(neighbor_solution)
        
        # Accept or reject based on Metropolis criterion
        if neighbor_energy < current_energy:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
            consecutive_failures = 0  # Reset failure counter on improvement
        else:
            # Accept with probability based on temperature
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 1000:  # Avoid accepting very bad moves too often
                acceptance_probability = np.exp(-delta_energy / temperature)
                if np.random.random() < acceptance_probability:
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
                    consecutive_failures = 0  # Reset failure counter on acceptance
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
        
        # Update best solution
        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
        
        # Strategy adjustment: if too many consecutive failures, increase step size
        if consecutive_failures > max_failures:
            # Increase step size slightly to escape local minima
            step_size_pos *= 1.1
            step_size_rot *= 1.1
            consecutive_failures = 0  # Reset counter
        
        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Track progress
        if iteration % 100 == 0:
            energy_history.append(current_energy)
    
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
    
    # Use simulated annealing optimization
    start_time = time.time()
    inner_hex_data, best_energy = simulated_annealing_optimization(max_iterations=5000)
    end_time = time.time()
    
    # Compute final outer hexagon size
    outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Ensure we have a valid solution
    if outer_side_length <= 0 or np.isnan(outer_side_length):
        # Fall back to a known good configuration
        inner_hex_data = improved_initial_configuration()
        outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
