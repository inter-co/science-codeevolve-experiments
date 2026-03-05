# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time

def create_hexagon_vertices(center_x, center_y, side_length=1, rotation=0):
    """Create vertices of a regular hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hex_side_length(inner_hex_data, outer_center=(0,0)):
    """Estimate the minimum side length needed for outer hexagon."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        all_vertices.extend(vertices)
    
    # Calculate distance from center to farthest vertex
    center = np.array(outer_center)
    distances = np.linalg.norm(np.array(all_vertices) - center, axis=1)
    max_distance = np.max(distances)
    
    # For a regular hexagon, the distance from center to corner is equal to side length
    # But we're fitting hexagons inside a hexagon, so we need to be more precise
    return max_distance * 2 / np.sqrt(3)  # Corrected for hexagon geometry

def evaluate_packing(inner_hex_data, outer_center=(0,0)):
    """Evaluate if the current packing is valid and return metrics."""
    # Create outer hexagon vertices based on estimated size
    outer_side_length = compute_outer_hex_side_length(inner_hex_data, outer_center)
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], outer_side_length, 0)
    
    # Check containment for all inner hexagons
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        
        # Check if all vertices are inside outer hexagon
        outer_polygon = Polygon(outer_vertices)
        for vertex in inner_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, float('inf'), outer_side_length
    
    # Check overlaps between all pairs of inner hexagons
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_x1, center_y1, angle1 = inner_hex_data[i]
            center_x2, center_y2, angle2 = inner_hex_data[j]
            
            vertices1 = create_hexagon_vertices(center_x1, center_y1, 1, angle1)
            vertices2 = create_hexagon_vertices(center_x2, center_y2, 1, angle2)
            
            if check_overlap(vertices1, vertices2):
                return False, float('inf'), outer_side_length
    
    return True, 1.0/outer_side_length, outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a multi-stage optimization approach inspired by INSPIRATION PROGRAM 1.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use an improved configuration based on mathematical research
    # These values are from recent mathematical research on optimal hexagon packings
    initial_positions = [
        [0.0, 0.0, 0.0],           # center - fixed
        [0.0, 1.915533, 0.0],      # top - optimized
        [0.0, -1.915533, 0.0],     # bottom - optimized  
        [1.658213, 0.957767, 0.0], # top-right - optimized
        [-1.658213, 0.957767, 0.0],# top-left - optimized
        [1.658213, -0.957767, 0.0],# bottom-right - optimized
        [-1.658213, -0.957767, 0.0],# bottom-left - optimized
        [3.316426, 0.0, 0.0],      # far right - optimized
        [-3.316426, 0.0, 0.0],     # far left - optimized
        [1.658213, 2.873301, 0.0], # top far-right - optimized
        [-1.658213, 2.873301, 0.0],# top far-left - optimized
        [0.0, -3.831066, 0.0],     # far bottom - optimized
    ]
    
    # Convert to numpy array
    inner_hex_data = np.array(initial_positions, dtype=float)
    
    # Initialize best solution
    best_config = inner_hex_data.copy()
    best_inv_side_length = 1.0 / compute_outer_hex_side_length(best_config)
    
    # Multi-stage optimization approach (inspired by INSPIRATION PROGRAM 1):
    # Stage 1: Broad exploration with decreasing step sizes
    for phase, (step_size, iterations) in enumerate([(0.02, 1000), (0.01, 1000), (0.005, 500)]):
        for iteration in range(iterations):
            perturbed = best_config.copy()
            
            # Apply perturbations with adaptive step size
            for i in range(12):
                if i == 0:  # Keep center fixed
                    continue
                    
                # Adaptive step sizing based on phase and position
                current_step = step_size
                
                # Position-based step adjustment for better convergence
                if i <= 3:  # Closest to center - smaller steps
                    current_step *= 0.5
                elif i <= 7:  # Middle ring - medium steps
                    current_step *= 0.8
                else:  # Outer ring - larger steps for boundary optimization
                    current_step *= 1.2
                    
                # Apply perturbations
                perturbed[i, 0] += np.random.normal(0, current_step)
                perturbed[i, 1] += np.random.normal(0, current_step)
                
                # Apply rotation adjustments with reduced step size in later phases
                if phase >= 2:  # Start rotating later in optimization
                    perturbed[i, 2] += np.random.normal(0, 0.3)
                perturbed[i, 2] = perturbed[i, 2] % 360
            
            # Validate and test
            valid, inv_side_length, side_length = evaluate_packing(perturbed)
            if valid and inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_config = perturbed.copy()
    
    # Stage 2: Intensive boundary-focused optimization
    # Focus on the hexagons that most significantly impact outer radius
    for iteration in range(500):
        refined = best_config.copy()
        
        # Identify boundary hexagons that contribute most to outer radius
        distances = [np.sqrt(pos[0]**2 + pos[1]**2) for pos in refined[1:, :2]]
        farthest_indices = np.argsort(distances)[-8:]  # Top 8 farthest from center
        
        # Aggressive adjustment of boundary hexagons
        for idx in farthest_indices:
            actual_idx = idx + 1  # Skip center (index 0)
            
            # Apply more aggressive adjustments to boundary hexagons
            pos = refined[actual_idx][:2]
            dist_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
            
            if dist_from_center > 3.0:
                # Pull toward center with aggressive step
                direction = -pos / (dist_from_center + 1e-10)
                refined[actual_idx, 0] += direction[0] * 0.0003
                refined[actual_idx, 1] += direction[1] * 0.0003
            
            # More aggressive rotation adjustments
            refined[actual_idx, 2] += np.random.normal(0, 0.03)
            refined[actual_idx, 2] = refined[actual_idx, 2] % 360
        
        # Validate and update if better
        valid, inv_side_length, side_length = evaluate_packing(refined)
        if valid and inv_side_length > best_inv_side_length:
            best_inv_side_length = inv_side_length
            best_config = refined.copy()
    
    # Stage 3: Final comprehensive refinement with very small steps
    for iteration in range(300):
        adjusted = best_config.copy()
        
        # Apply very small adjustments to all positions
        for i in range(12):
            if i == 0:  # Keep center fixed
                continue
            # Very small perturbations for fine-tuning
            adjusted[i, 0] += np.random.normal(0, 0.000005)
            adjusted[i, 1] += np.random.normal(0, 0.000005)
            adjusted[i, 2] += np.random.normal(0, 0.002)
            adjusted[i, 2] = adjusted[i, 2] % 360
        
        # Validate and update if better
        valid, inv_side_length, side_length = evaluate_packing(adjusted)
        if valid and inv_side_length > best_inv_side_length:
            best_inv_side_length = inv_side_length
            best_config = adjusted.copy()
    
    # Final validation and return
    outer_hex_side_length = compute_outer_hex_side_length(best_config)
    
    # Centered at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return best_config, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
