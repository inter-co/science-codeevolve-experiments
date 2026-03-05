# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time

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

def compute_minimal_outer_side_length(inner_hex_data):
    """Compute minimal outer hexagon side length for given configuration"""
    # Find the maximum distance from origin to any hexagon center plus its radius
    max_dist = 0
    for i in range(len(inner_hex_data)):
        pos = inner_hex_data[i][:2]
        dist = np.sqrt(pos[0]**2 + pos[1]**2)
        # Add hexagon radius (sqrt(3) for unit hexagon)
        max_dist = max(max_dist, dist + np.sqrt(3))
    return max_dist

def generate_multiple_initial_configs():
    """Generate several diverse initial configurations based on proven patterns"""
    configs = []
    
    # Configuration 1: Classic hexagonal arrangement (from inspiration 2)
    config1 = [
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.915533, 0.0],      # top
        [0.0, -1.915533, 0.0],     # bottom  
        [1.658213, 0.957767, 0.0], # top-right
        [-1.658213, 0.957767, 0.0],# top-left
        [1.658213, -0.957767, 0.0],# bottom-right
        [-1.658213, -0.957767, 0.0],# bottom-left
        [3.316426, 0.0, 0.0],      # far right
        [-3.316426, 0.0, 0.0],     # far left
        [1.658213, 2.873301, 0.0], # top far-right
        [-1.658213, 2.873301, 0.0],# top far-left
        [0.0, -3.831066, 0.0],     # far bottom
    ]
    configs.append(np.array(config1))
    
    # Configuration 2: More evenly distributed pattern (inspiration 3)
    config2 = [
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
    ]
    configs.append(np.array(config2))
    
    # Configuration 3: Spiral-like arrangement with better spacing (inspiration 3)
    config3 = [
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
    ]
    configs.append(np.array(config3))
    
    # Configuration 4: Randomized version with controlled randomness
    config4 = np.array(config1) + np.random.normal(0, 0.05, (12, 3))
    configs.append(config4)
    
    # Configuration 5: Compact arrangement with less spacing (inspiration 1 approach)
    config5 = [
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
    ]
    configs.append(np.array(config5))
    
    return configs

def force_directed_placement(initial_config, max_iter=2000, learning_rate=0.02):
    """
    Apply enhanced force-directed placement to optimize hexagon positions while maintaining constraints
    Inspired by inspiration 1 and 3 with improved physics modeling
    """
    # Convert to numpy array for easier manipulation
    positions = np.array([[h[0], h[1]] for h in initial_config])
    rotations = np.array([h[2] for h in initial_config])
    
    # Calculate initial outer radius
    outer_radius = compute_outer_hex_side_length(initial_config)
    
    for iteration in range(max_iter):
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Compute repulsion forces between overlapping hexagons (stronger force at closer distances)
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                
                # Distance between centers
                dist_vec = pos_j - pos_i
                distance = np.linalg.norm(dist_vec)
                
                # If hexagons are too close, apply repulsion force
                if distance < 2.0:  # 2 units is the minimum distance for non-overlapping unit hexagons
                    # Use stronger repulsion with better physical model
                    repulsion_magnitude = 2.0 / (distance * distance + 0.01)
                    repulsion_force = repulsion_magnitude * dist_vec / (distance + 0.01)
                    forces[i] += repulsion_force
                    forces[j] -= repulsion_force
        
        # Compute attraction forces to keep hexagons together and maintain structure
        center_of_mass = np.mean(positions, axis=0)
        for i in range(len(positions)):
            # Attract to center of mass with stronger force
            attraction_force = 0.05 * (center_of_mass - positions[i])
            forces[i] += attraction_force
            
            # Keep within bounds with stronger constraint
            pos = positions[i]
            dist_to_center = np.linalg.norm(pos)
            if dist_to_center > outer_radius - 1.732:  # sqrt(3) for hexagon radius
                # Push back towards center with stronger force when far from center
                push_force = -0.2 * pos / (dist_to_center + 0.1)
                forces[i] += push_force
        
        # Apply forces to update positions
        positions += learning_rate * forces
        
        # Correct overlaps directly with better handling
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                dist_vec = pos_j - pos_i
                distance = np.linalg.norm(dist_vec)
                
                # If they're overlapping, push them apart directly with better separation
                if distance < 2.0:
                    separation = (2.0 - distance) * dist_vec / (distance + 0.001)
                    positions[i] -= 0.3 * separation
                    positions[j] += 0.3 * separation
        
        # Keep positions within outer hexagon boundary with improved projection
        for i in range(len(positions)):
            pos = positions[i]
            dist_to_center = np.linalg.norm(pos)
            if dist_to_center > outer_radius - 1.732:
                # Project back onto boundary with more careful handling
                if dist_to_center > 0:
                    positions[i] = pos * (outer_radius - 1.732) / dist_to_center
        
        # Adjust outer radius after each iteration to reflect current configuration
        outer_radius = compute_outer_hex_side_length(np.column_stack([positions, rotations]))
    
    # Update the data with optimized positions
    optimized_data = []
    for i in range(len(initial_config)):
        optimized_data.append([positions[i][0], positions[i][1], rotations[i]])
    
    return np.array(optimized_data)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining multiple initial configurations with force-directed optimization.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configs()
    
    best_config = None
    best_side_length = float('inf')
    best_inv_ratio = 0
    
    # Try each configuration and optimize it
    for i, config in enumerate(initial_configs):
        try:
            # Apply force-directed placement for optimization
            optimized_config = force_directed_placement(config.copy(), max_iter=1500, learning_rate=0.01)
            
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
    
    # If no good configuration found, fallback to the best known configuration
    if best_config is None:
        # Use the configuration from inspiration 1 as fallback
        inner_hex_data = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 1.915533, 0.0],      # top
            [0.0, -1.915533, 0.0],     # bottom  
            [1.658213, 0.957767, 0.0], # top-right
            [-1.658213, 0.957767, 0.0],# top-left
            [1.658213, -0.957767, 0.0],# bottom-right
            [-1.658213, -0.957767, 0.0],# bottom-left
            [3.316426, 0.0, 0.0],      # far right
            [-3.316426, 0.0, 0.0],     # far left
            [1.658213, 2.873301, 0.0], # top far-right
            [-1.658213, 2.873301, 0.0],# top far-left
            [0.0, -3.831066, 0.0],     # far bottom
        ])
        best_config = inner_hex_data
        best_side_length = compute_outer_hex_side_length(inner_hex_data)
    
    # Final refinement with local search
    final_config = best_config.copy()
    final_inv_ratio = best_inv_ratio
    
    # Additional aggressive local search with better step sizing
    for iteration in range(1500):
        # Create perturbation with adaptive step sizes based on position
        perturbed = final_config.copy()
        
        # Different step sizes based on hexagon's radial distance from center
        for i in range(12):
            if i == 0:  # Keep center fixed
                continue
                
            # Step size varies based on position relative to center
            pos = perturbed[i][:2]
            dist_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
            
            # Outer hexagons get larger steps, inner ones smaller
            if dist_from_center > 3.0:  # Far outer ring
                step = 0.0015
            elif dist_from_center > 1.5:  # Middle ring
                step = 0.001
            else:  # Inner ring
                step = 0.0005
                
            # Apply small perturbations
            perturbed[i, 0] += np.random.normal(0, step)
            perturbed[i, 1] += np.random.normal(0, step)
            # Rotation adjustment with very small step
            perturbed[i, 2] += np.random.normal(0, 0.1)
            perturbed[i, 2] = perturbed[i, 2] % 360
        
        # Validate and test
        if validate_packing(perturbed, compute_minimal_outer_side_length(perturbed)):
            inv_side_length = 1.0 / compute_minimal_outer_side_length(perturbed)
            if inv_side_length > final_inv_ratio:
                final_inv_ratio = inv_side_length
                final_config = perturbed.copy()
    
    # Final validation
    outer_hex_side_length = compute_minimal_outer_side_length(final_config)
    
    # Centered at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return final_config, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
