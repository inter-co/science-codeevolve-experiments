# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from scipy.spatial.distance import cdist
import random

def create_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Create vertices of a regular hexagon"""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices = []
    for angle in angles[:-1]:  # exclude last point to close polygon
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap"""
    polygon1 = Polygon(hex1_vertices)
    polygon2 = Polygon(hex2_vertices)
    return polygon1.intersects(polygon2)

def compute_outer_hex_side_length(inner_hex_data):
    """Compute minimum outer hexagon side length that contains all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        pos = inner_hex_data[i][:2]
        # Distance from center to hexagon center
        dist = np.sqrt(pos[0]**2 + pos[1]**2)
        # Add hexagon radius (sqrt(3) for unit hexagon)
        max_dist = max(max_dist, dist + np.sqrt(3))
    
    # For a regular hexagon with circumradius R, side length = R
    # We want outer hexagon to contain all inner hexagons
    return max_dist

def validate_packing(inner_hex_data, outer_side_length):
    """Validate that the packing is valid (no overlaps, all contained)"""
    # Create outer hexagon vertices
    outer_vertices = create_hexagon_vertices(0, 0, outer_side_length, 0)
    
    # Check each inner hexagon
    for i in range(len(inner_hex_data)):
        pos = inner_hex_data[i][:2]
        rotation = inner_hex_data[i][2]
        hex_vertices = create_hexagon_vertices(pos[0], pos[1], 1, rotation)
        
        if not check_hexagon_containment(hex_vertices, outer_vertices):
            return False
            
        # Check overlaps with other hexagons
        for j in range(i+1, len(inner_hex_data)):
            other_pos = inner_hex_data[j][:2]
            other_rotation = inner_hex_data[j][2]
            other_vertices = create_hexagon_vertices(other_pos[0], other_pos[1], 1, other_rotation)
            if check_hexagon_overlap(hex_vertices, other_vertices):
                return False
                
    return True

def compute_penalty(inner_hex_data, outer_side_length):
    """Compute penalty for invalid configurations"""
    penalty = 0
    outer_vertices = create_hexagon_vertices(0, 0, outer_side_length, 0)
    
    # Check containment penalties
    for i in range(len(inner_hex_data)):
        pos = inner_hex_data[i][:2]
        rotation = inner_hex_data[i][2]
        hex_vertices = create_hexagon_vertices(pos[0], pos[1], 1, rotation)
        if not check_hexagon_containment(hex_vertices, outer_vertices):
            penalty += 10000
    
    # Check overlap penalties
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            pos1 = inner_hex_data[i][:2]
            rotation1 = inner_hex_data[i][2]
            pos2 = inner_hex_data[j][:2]
            rotation2 = inner_hex_data[j][2]
            
            hex1_vertices = create_hexagon_vertices(pos1[0], pos1[1], 1, rotation1)
            hex2_vertices = create_hexagon_vertices(pos2[0], pos2[1], 1, rotation2)
            
            if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                penalty += 10000
    
    return penalty

def objective_function(params):
    """Objective function to maximize 1/outer_hex_side_length (minimize negative of it)"""
    # Reshape params into 12 hexagons with (x, y, angle)
    hex_data = params.reshape(-1, 3)
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hex_side_length(hex_data)
    
    # Check if this configuration is valid
    penalty = compute_penalty(hex_data, outer_side_length)
    
    if penalty > 0:
        # Return large penalty for invalid configurations
        return 1e10
    
    # Return negative inverse side length (we want to maximize 1/R, so minimize -1/R)
    return -1.0 / outer_side_length

def generate_symmetric_configurations():
    """Generate several symmetric initial configurations"""
    configs = []
    
    # Configuration 1: Highly symmetric pattern with 6-fold rotational symmetry
    config1 = np.array([
        [0.0, 0.0, 0.0],          # center
        [0.0, 1.915, 0.0],        # top
        [0.0, -1.915, 0.0],       # bottom  
        [1.658, 0.957, 0.0],      # top-right
        [-1.658, 0.957, 0.0],     # top-left
        [1.658, -0.957, 0.0],     # bottom-right
        [-1.658, -0.957, 0.0],    # bottom-left
        [3.316, 0.0, 0.0],        # far right
        [-3.316, 0.0, 0.0],       # far left
        [1.658, 2.871, 0.0],      # top far-right
        [-1.658, 2.871, 0.0],     # top far-left
        [0.0, -3.830, 0.0],       # far bottom
    ])
    configs.append(config1)
    
    # Configuration 2: Optimized version with better radial distribution
    config2 = np.array([
        [0.0, 0.0, 0.0],          # center
        [0.0, 1.8, 0.0],          # top
        [0.0, -1.8, 0.0],         # bottom  
        [1.55, 0.87, 0.0],        # top-right
        [-1.55, 0.87, 0.0],       # top-left
        [1.55, -0.87, 0.0],       # bottom-right
        [-1.55, -0.87, 0.0],      # bottom-left
        [3.1, 0.0, 0.0],          # far right
        [-3.1, 0.0, 0.0],         # far left
        [1.55, 2.6, 0.0],         # top far-right
        [-1.55, 2.6, 0.0],        # top far-left
        [0.0, -3.6, 0.0],         # far bottom
    ])
    configs.append(config2)
    
    # Configuration 3: Concentric ring arrangement (more compact)
    config3 = np.array([
        [0.0, 0.0, 0.0],          # center
        [0.0, 1.5, 0.0],          # top
        [0.0, -1.5, 0.0],         # bottom  
        [1.3, 0.75, 0.0],         # top-right
        [-1.3, 0.75, 0.0],        # top-left
        [1.3, -0.75, 0.0],        # bottom-right
        [-1.3, -0.75, 0.0],       # bottom-left
        [2.6, 0.0, 0.0],          # far right
        [-2.6, 0.0, 0.0],         # far left
        [1.3, 2.25, 0.0],         # top far-right
        [-1.3, 2.25, 0.0],        # top far-left
        [0.0, -3.0, 0.0],         # far bottom
    ])
    configs.append(config3)
    
    # Configuration 4: Spiral arrangement with radial spacing
    config4 = np.array([
        [0.0, 0.0, 0.0],          # center
        [0.0, 1.732, 0.0],        # top
        [0.0, -1.732, 0.0],       # bottom  
        [1.5, 0.866, 0.0],        # top-right
        [-1.5, 0.866, 0.0],       # top-left
        [1.5, -0.866, 0.0],       # bottom-right
        [-1.5, -0.866, 0.0],      # bottom-left
        [3.0, 0.0, 0.0],          # far right
        [-3.0, 0.0, 0.0],         # far left
        [1.5, 2.598, 0.0],        # top far-right
        [-1.5, 2.598, 0.0],       # top far-left
        [0.0, -3.464, 0.0],       # far bottom
    ])
    configs.append(config4)
    
    return configs

def force_directed_placement_with_rotation(initial_config, max_iter=1000, learning_rate=0.01):
    """
    Apply enhanced force-directed placement with rotation optimization to optimize hexagon positions and orientations
    """
    # Convert to numpy array for easier manipulation
    positions = np.array([[h[0], h[1]] for h in initial_config])
    rotations = np.array([h[2] for h in initial_config])
    
    # Calculate initial outer radius
    outer_radius = compute_outer_hex_side_length(initial_config)
    
    for iteration in range(max_iter):
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Compute repulsion forces between overlapping hexagons
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                
                # Distance between centers
                dist_vec = pos_j - pos_i
                distance = np.linalg.norm(dist_vec)
                
                # If hexagons are too close, apply repulsion force
                if distance < 2.0:  # 2 units is the minimum distance for non-overlapping unit hexagons
                    repulsion_force = 1.0 / (distance + 0.1) * dist_vec / (distance + 0.01)
                    forces[i] += repulsion_force
                    forces[j] -= repulsion_force
        
        # Compute attraction forces to keep hexagons together and maintain structure
        center_of_mass = np.mean(positions, axis=0)
        for i in range(len(positions)):
            # Attract to center of mass
            attraction_force = 0.05 * (center_of_mass - positions[i])
            forces[i] += attraction_force
            
            # Keep within bounds
            pos = positions[i]
            dist_to_center = np.linalg.norm(pos)
            if dist_to_center > outer_radius - 1.732:  # sqrt(3) for hexagon radius
                push_force = -0.1 * pos / (dist_to_center + 0.1)
                forces[i] += push_force
        
        # Apply forces to update positions
        positions += learning_rate * forces
        
        # Correct overlaps directly
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                dist_vec = pos_j - pos_i
                distance = np.linalg.norm(dist_vec)
                
                # If they're overlapping, push them apart
                if distance < 2.0:
                    separation = (2.0 - distance) * dist_vec / (distance + 0.01)
                    positions[i] -= 0.5 * separation
                    positions[j] += 0.5 * separation
        
        # Keep positions within outer hexagon boundary
        for i in range(len(positions)):
            pos = positions[i]
            dist_to_center = np.linalg.norm(pos)
            if dist_to_center > outer_radius - 1.732:
                if dist_to_center > 0:
                    positions[i] = pos * (outer_radius - 1.732) / dist_to_center
        
        # Adjust outer radius after each iteration
        outer_radius = compute_outer_hex_side_length(np.column_stack([positions, rotations]))
    
    # Update the data with optimized positions
    optimized_data = []
    for i in range(len(initial_config)):
        optimized_data.append([positions[i][0], positions[i][1], rotations[i]])
    
    return np.array(optimized_data)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining symmetric initial configurations, force-directed optimization, and multi-start optimization.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start timing
    start_time = time.time()
    
    # Generate symmetric configurations
    initial_configs = generate_symmetric_configurations()
    
    best_config = None
    best_side_length = float('inf')
    best_inv_ratio = 0
    
    # Try each configuration with force-directed placement
    for i, config in enumerate(initial_configs):
        try:
            # Apply force-directed placement for optimization
            optimized_config = force_directed_placement_with_rotation(config.copy(), max_iter=500, learning_rate=0.01)
            
            # Calculate side length
            side_length = compute_outer_hex_side_length(optimized_config)
            
            # Validate the configuration
            if validate_packing(optimized_config, side_length):
                inv_ratio = 1.0 / side_length
                if inv_ratio > best_inv_ratio:
                    best_inv_ratio = inv_ratio
                    best_config = optimized_config.copy()
                    best_side_length = side_length
                    
        except Exception as e:
            continue
    
    # If no good configuration found, try guided random search approach (like inspiration 1)
    if best_config is None:
        # Start with a known good configuration
        base_config = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 1.915, 0.0],         # top
            [0.0, -1.915, 0.0],        # bottom  
            [1.658, 0.957, 0.0],       # top-right
            [-1.658, 0.957, 0.0],      # top-left
            [1.658, -0.957, 0.0],      # bottom-right
            [-1.658, -0.957, 0.0],     # bottom-left
            [3.316, 0.0, 0.0],         # far right
            [-3.316, 0.0, 0.0],        # far left
            [1.658, 2.871, 0.0],       # top far-right
            [-1.658, 2.871, 0.0],      # top far-left
            [0.0, -3.830, 0.0],        # far bottom
        ])
        
        best_config = base_config.copy()
        best_side_length = compute_outer_hex_side_length(best_config)
        best_inv_ratio = 1.0 / best_side_length
        
        # Perform guided random search to improve
        for iteration in range(2000):
            # Create a variant of the base configuration
            config = base_config.copy()
            
            # Make small random adjustments to positions
            for i in range(1, 12):  # Skip center
                config[i][0] += random.uniform(-0.03, 0.03)
                config[i][1] += random.uniform(-0.03, 0.03)
            
            # Try to adjust some positions more aggressively
            if random.random() < 0.2:  # 20% chance of more aggressive adjustment
                corner_indices = [3, 4, 5, 6]  # corner positions
                idx = random.choice(corner_indices)
                config[idx][0] *= 0.995  # Slight inward adjustment
                config[idx][1] *= 0.995
            
            # Compute and validate
            side_length = compute_outer_hex_side_length(config)
            if validate_packing(config, side_length):
                inv_side_length = 1.0 / side_length
                if inv_side_length > best_inv_ratio:
                    best_inv_ratio = inv_side_length
                    best_config = config.copy()
                    best_side_length = side_length
    
    # Final validation
    outer_hex_side_length = compute_outer_hex_side_length(best_config)
    
    # Centered at origin
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return best_config, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
