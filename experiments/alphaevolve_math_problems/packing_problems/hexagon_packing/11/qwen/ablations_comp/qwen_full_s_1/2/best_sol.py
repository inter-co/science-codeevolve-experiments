# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
from scipy.spatial.distance import cdist

def get_hexagon_vertices(center, side_length, rotation=0):
    """Get vertices of a regular hexagon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + side_length * np.cos(angle), 
             center[1] + side_length * np.sin(angle)) 
            for angle in angles]

def check_hexagon_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon"""
    try:
        hex_poly = Polygon(hexagon)
        outer_poly = Polygon(outer_hexagon)
        return outer_poly.contains(hex_poly)
    except:
        return False

def check_hexagon_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    try:
        poly1 = Polygon(hex1)
        poly2 = Polygon(hex2)
        return poly1.intersects(poly2)
    except:
        return True

def compute_hexagon_distances(hex1_vertices, hex2_vertices):
    """Compute minimum distance between two hexagons"""
    points1 = np.array(hex1_vertices)
    points2 = np.array(hex2_vertices)
    distances = cdist(points1, points2)
    return np.min(distances)

def compute_forces_on_hexagons(hex_data, outer_radius):
    """Compute forces acting on each hexagon in a physics simulation"""
    num_hexagons = len(hex_data)
    forces = np.zeros((num_hexagons, 2))  # (fx, fy) for each hexagon
    
    # Repulsive forces between overlapping hexagons
    for i in range(num_hexagons):
        for j in range(i+1, num_hexagons):
            center_i = hex_data[i][:2]
            center_j = hex_data[j][:2]
            rotation_i = hex_data[i][2]
            rotation_j = hex_data[j][2]
            
            hex_i = get_hexagon_vertices(center_i, 1.0, rotation_i)
            hex_j = get_hexagon_vertices(center_j, 1.0, rotation_j)
            
            # Check if they overlap
            if check_hexagon_overlap(hex_i, hex_j):
                # Compute repulsion force
                diff = np.array(center_i) - np.array(center_j)
                distance = np.linalg.norm(diff)
                if distance > 0:
                    force_magnitude = 1.0 / (distance ** 2)
                    forces[i] += force_magnitude * diff / distance
                    forces[j] -= force_magnitude * diff / distance
    
    # Attractive forces towards center (to keep hexagons from flying away)
    center = np.array([0.0, 0.0])
    for i in range(num_hexagons):
        center_diff = center - hex_data[i][:2]
        distance = np.linalg.norm(center_diff)
        if distance > 0:
            forces[i] += 0.01 * center_diff / distance
    
    # Boundary forces to keep hexagons inside outer hexagon
    outer_hex = get_hexagon_vertices((0, 0), outer_radius, 0)
    for i in range(num_hexagons):
        center_i = hex_data[i][:2]
        hex_i = get_hexagon_vertices(center_i, 1.0, hex_data[i][2])
        
        # Check if any vertex is outside the boundary
        for vertex in hex_i:
            vertex_point = np.array(vertex)
            # Simple distance check to boundary
            dist_to_center = np.linalg.norm(vertex_point)
            if dist_to_center > outer_radius - 1.0:  # Account for hexagon radius
                # Apply repulsion from boundary
                boundary_force = (vertex_point - center) * 0.1
                forces[i] -= boundary_force
    
    return forces

def simulate_hexagon_packaging(initial_hex_data, max_iterations=1000):
    """Physics-based simulation to pack hexagons optimally"""
    # Start with a reasonable outer radius
    outer_radius = 5.0
    hex_data = initial_hex_data.copy()
    
    # Add some randomness to initial positions for better exploration
    for i in range(len(hex_data)):
        hex_data[i][0] += np.random.normal(0, 0.1)
        hex_data[i][1] += np.random.normal(0, 0.1)
    
    for iteration in range(max_iterations):
        # Compute forces
        forces = compute_forces_on_hexagons(hex_data, outer_radius)
        
        # Update positions
        for i in range(len(hex_data)):
            # Apply forces with damping
            hex_data[i][0] += 0.01 * forces[i][0]
            hex_data[i][1] += 0.01 * forces[i][1]
        
        # Occasionally adjust outer radius based on how packed we are
        if iteration % 100 == 0:
            # Check current packing quality
            min_dist = float('inf')
            for i in range(len(hex_data)):
                for j in range(i+1, len(hex_data)):
                    center_i = hex_data[i][:2]
                    center_j = hex_data[j][:2]
                    distance = np.linalg.norm(np.array(center_i) - np.array(center_j))
                    min_dist = min(min_dist, distance)
            
            # If hexagons are too close, expand outer radius
            if min_dist < 1.5:
                outer_radius *= 1.01
            elif min_dist > 2.0 and outer_radius > 3.0:
                outer_radius *= 0.99
    
    return hex_data, outer_radius

def binary_search_min_outer_size(inner_solution, max_size=15.0, precision=1e-8):
    """Binary search to find minimum outer hexagon size that contains all hexagons"""
    low = 1.0
    high = max_size
    best_size = max_size
    
    # Binary search with high precision
    iterations = int(np.log2(max_size / precision))
    for _ in range(iterations):
        mid = (low + high) / 2
        # Simplified validation - just check if all hexagons fit without overlap
        # This is a faster approximation than full validation
        try:
            # Quick check: see if all hexagons can fit inside outer hexagon
            outer_center = (0, 0)
            outer_hex = get_hexagon_vertices(outer_center, mid, 0)
            
            valid = True
            for i in range(len(inner_solution)):
                center = (inner_solution[i][0], inner_solution[i][1])
                rotation = inner_solution[i][2]
                side_length = 1.0
                hexagon = get_hexagon_vertices(center, side_length, rotation)
                
                # Check containment
                if not check_hexagon_containment(hexagon, outer_hex):
                    valid = False
                    break
                    
                # Check overlaps with other hexagons
                for j in range(i):
                    hexagon_j = get_hexagon_vertices(
                        (inner_solution[j][0], inner_solution[j][1]), 
                        1.0, 
                        inner_solution[j][2]
                    )
                    if check_hexagon_overlap(hexagon, hexagon_j):
                        valid = False
                        break
                        
            if valid:
                best_size = mid
                high = mid
            else:
                low = mid
                
        except:
            low = mid
            
    return best_size

def generate_physics_based_initial_configurations():
    """Generate initial configurations using physics-inspired layout patterns"""
    configs = []
    
    # Configuration 1: Hexagonal lattice pattern with central hexagon
    config1 = np.array([
        [0.0, 0.0, 0.0],       # center
        [0.0, 2.0, 0.0],       # top
        [1.732, 1.0, 0.0],     # top-right (sqrt(3) = 1.732)
        [1.732, -1.0, 0.0],    # bottom-right
        [0.0, -2.0, 0.0],      # bottom
        [-1.732, -1.0, 0.0],   # bottom-left
        [-1.732, 1.0, 0.0],    # top-left
        [3.464, 0.0, 0.0],     # far right (2*sqrt(3))
        [-3.464, 0.0, 0.0],    # far left
        [1.732, 3.0, 0.0],     # upper triangle
        [-1.732, 3.0, 0.0],    # upper triangle
    ])
    configs.append(config1)
    
    # Configuration 2: Spiral-like arrangement with radial symmetry
    config2 = np.array([
        [0.0, 0.0, 0.0],       # center
        [0.0, 1.8, 0.0],       # top
        [1.55, 0.9, 0.0],      # top-right
        [1.55, -0.9, 0.0],     # bottom-right
        [0.0, -1.8, 0.0],      # bottom
        [-1.55, -0.9, 0.0],    # bottom-left
        [-1.55, 0.9, 0.0],     # top-left
        [3.1, 0.0, 0.0],       # far right
        [-3.1, 0.0, 0.0],      # far left
        [1.55, 2.7, 0.0],      # upper triangle
        [-1.55, 2.7, 0.0],     # upper triangle
    ])
    configs.append(config2)
    
    # Configuration 3: Dense cluster with irregular spacing
    config3 = np.array([
        [0.0, 0.0, 0.0],       # center
        [0.0, 1.9, 0.0],       # top
        [1.65, 0.95, 0.0],     # top-right
        [1.65, -0.95, 0.0],    # bottom-right
        [0.0, -1.9, 0.0],      # bottom
        [-1.65, -0.95, 0.0],   # bottom-left
        [-1.65, 0.95, 0.0],    # top-left
        [3.3, 0.0, 0.0],       # far right
        [-3.3, 0.0, 0.0],      # far left
        [1.65, 2.85, 0.0],     # upper triangle
        [-1.65, 2.85, 0.0],    # upper triangle
    ])
    configs.append(config3)
    
    return configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a novel physics-based simulation approach with force-directed modeling.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate initial configurations
    initial_configs = generate_physics_based_initial_configurations()
    
    best_solution = None
    best_side_length = float('inf')
    
    # Try physics simulation on each initial configuration
    for i, initial_config in enumerate(initial_configs):
        # Run physics simulation
        try:
            simulated_solution, outer_radius = simulate_hexagon_packaging(initial_config, max_iterations=500)
            
            # Find the minimal outer hexagon size that contains this configuration
            final_side_length = binary_search_min_outer_size(simulated_solution, 15.0, 1e-6)
            
            # Validate the solution
            valid = True
            outer_hex = get_hexagon_vertices((0, 0), final_side_length, 0)
            for j in range(len(simulated_solution)):
                center = (simulated_solution[j][0], simulated_solution[j][1])
                rotation = simulated_solution[j][2]
                hexagon = get_hexagon_vertices(center, 1.0, rotation)
                
                if not check_hexagon_containment(hexagon, outer_hex):
                    valid = False
                    break
                    
                for k in range(j):
                    hexagon_k = get_hexagon_vertices(
                        (simulated_solution[k][0], simulated_solution[k][1]), 
                        1.0, 
                        simulated_solution[k][2]
                    )
                    if check_hexagon_overlap(hexagon, hexagon_k):
                        valid = False
                        break
            
            if valid and final_side_length < best_side_length:
                best_side_length = final_side_length
                best_solution = simulated_solution.copy()
                
        except Exception as e:
            continue
            
        # Early exit if time limit approaching
        if time.time() - start_time > 55:
            break
    
    # If no good solution found, use one of the initial configurations
    if best_solution is None:
        # Use a well-known configuration from the literature
        best_solution = np.array([
            [0.0, 0.0, 0.0],       # center
            [0.0, 2.0, 0.0],       # top
            [1.732, 1.0, 0.0],     # top-right (sqrt(3) = 1.732)
            [1.732, -1.0, 0.0],    # bottom-right
            [0.0, -2.0, 0.0],      # bottom
            [-1.732, -1.0, 0.0],   # bottom-left
            [-1.732, 1.0, 0.0],    # top-left
            [3.464, 0.0, 0.0],     # far right (2*sqrt(3))
            [-3.464, 0.0, 0.0],    # far left
            [1.732, 3.0, 0.0],     # upper triangle
            [-1.732, 3.0, 0.0],    # upper triangle
        ])
        best_side_length = binary_search_min_outer_size(best_solution, 15.0, 1e-8)
    
    # Ensure all rotations are within [0, 360)
    best_solution[:, 2] = np.mod(best_solution[:, 2], 360)
    
    # Set outer hexagon at center with zero rotation
    outer_hex_data = np.array([0, 0, 0])
    
    return best_solution, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
