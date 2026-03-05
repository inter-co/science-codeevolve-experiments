# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import math
import random
import time


def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(theta)
        y = center_y + side_length * math.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def create_outer_hexagon(side_length):
    """Create vertices of outer hexagon centered at origin."""
    return generate_hexagon_vertices(0, 0, 0, side_length)

def point_in_polygon(point, polygon_vertices):
    """Check if a point is inside a polygon using Shapely."""
    poly = Polygon(polygon_vertices)
    pt = Point(point)
    return poly.contains(pt)

def hexagon_contains_point(hex_vertices, point):
    """Check if a point is inside a hexagon."""
    return point_in_polygon(point, hex_vertices)

def hexagon_intersects(hex1_vertices, hex2_vertices):
    """Check if two hexagons intersect using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hexagon_side_length(inner_hex_data, outer_radius_guess=None):
    """Calculate the minimum outer hexagon side length that contains all inner hexagons."""
    if outer_radius_guess is None:
        # Estimate based on maximum distance from center
        max_dist = 0
        for x, y, angle in inner_hex_data:
            # Get vertices of this hexagon
            vertices = generate_hexagon_vertices(x, y, angle)
            # Find maximum distance from center (0,0) to any vertex
            for vx, vy in vertices:
                dist = math.sqrt(vx*vx + vy*vy)
                max_dist = max(max_dist, dist)
        # Add some buffer for safety
        outer_radius_guess = max_dist + 1.0
    
    # Binary search for tightest fit with better precision
    min_radius = 0.1
    max_radius = outer_radius_guess * 2  # Start with a reasonable upper bound
    
    # Binary search for tightest fit - optimized iterations to stay within time budget
    for _ in range(20):  # Reduced iterations to stay within time budget
        if max_radius - min_radius < 0.0001:  # Stop if we're very close
            break
        mid_radius = (min_radius + max_radius) / 2
        outer_hex = create_outer_hexagon(mid_radius)
        
        # Check if all inner hexagons fit
        all_fit = True
        for x, y, angle in inner_hex_data:
            inner_hex = generate_hexagon_vertices(x, y, angle)
            # Check if all vertices of inner hex are within outer hex
            for vertex in inner_hex:
                if not hexagon_contains_point(outer_hex, vertex):
                    all_fit = False
                    break
            if not all_fit:
                break
        
        if all_fit:
            max_radius = mid_radius
        else:
            min_radius = mid_radius
    
    return max_radius

def evaluate_solution(inner_hex_data, outer_radius=None):
    """Comprehensive evaluation of a solution."""
    if outer_radius is None:
        outer_radius = calculate_outer_hexagon_side_length(inner_hex_data)
    
    # Create outer hexagon
    outer_hex = create_outer_hexagon(outer_radius)
    
    # Check containment - more thorough check
    for x, y, angle in inner_hex_data:
        inner_hex = generate_hexagon_vertices(x, y, angle)
        # Check all vertices are within outer hexagon
        for vertex in inner_hex:
            if not hexagon_contains_point(outer_hex, vertex):
                return False, outer_radius, float('inf')
    
    # Check overlaps - more thorough check
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]
            hex1 = generate_hexagon_vertices(x1, y1, angle1)
            hex2 = generate_hexagon_vertices(x2, y2, angle2)
            
            if hexagon_intersects(hex1, hex2):
                return False, outer_radius, float('inf')
    
    return True, outer_radius, 1.0 / outer_radius

def advanced_local_search(initial_positions, max_time=45):
    """Advanced local search with multiple strategies and better escape mechanisms"""
    start_time = time.time()
    
    # Multiple restart strategy inspired by INSPIRATION 2 and 3
    best_positions = None
    best_score = -float('inf')
    
    # Try multiple restarts with different strategies
    num_restarts = 8  # More restarts for better exploration
    
    # Base configuration that we know works well
    base_positions = [
        [0, 0, 0],              # center
        [0, 1.9419123, 0],      # top
        [0, -1.9419123, 0],     # bottom
        [1.680872, 0.970956, 0],# top-right
        [-1.680872, 0.970956, 0],# top-left
        [1.680872, -0.970956, 0],# bottom-right
        [-1.680872, -0.970956, 0],# bottom-left
        [3.361744, 0, 0],       # far right
        [-3.361744, 0, 0],      # far left
        [1.680872, 2.912868, 0],# top far-right
        [-1.680872, 2.912868, 0],# top far-left
        [1.680872, -2.912868, 0],# bottom far-right
    ]
    
    for restart in range(num_restarts):
        # For first restart, use base configuration
        if restart == 0:
            current_positions = np.array(base_positions)
        else:
            # For subsequent restarts, start with a slightly different configuration
            current_positions = np.array(base_positions).copy()
            
            # Apply different perturbations for diversity
            if restart < 4:  # First 4 restarts - moderate perturbations
                for i in range(12):
                    current_positions[i, 0] += random.uniform(-0.05, 0.05)
                    current_positions[i, 1] += random.uniform(-0.05, 0.05)
                    current_positions[i, 2] += random.uniform(-1, 1)
                    current_positions[i, 2] = max(0, min(360, current_positions[i, 2]))
            else:  # Last 4 restarts - more aggressive perturbations
                for i in range(12):
                    current_positions[i, 0] += random.uniform(-0.1, 0.1)
                    current_positions[i, 1] += random.uniform(-0.1, 0.1)
                    current_positions[i, 2] += random.uniform(-3, 3)
                    current_positions[i, 2] = max(0, min(360, current_positions[i, 2]))
        
        # Initialize best for this restart
        current_best_score = evaluate_solution(current_positions)[2]
        current_best_positions = current_positions.copy()
        
        # Track improvements for adaptive strategy
        last_improvement = 0
        stagnation_counter = 0
        
        # Multi-phase optimization with better time management
        phase = 0
        max_phases = 4
        iteration = 0
        max_iterations = 2500  # Reduced iterations to stay within time budget
        
        # Different step sizes for different phases
        step_sizes = [(0.1, 5), (0.05, 3), (0.02, 2), (0.01, 1)]
        
        while iteration < max_iterations and (time.time() - start_time) < max_time * 0.9:
            # Determine current step sizes based on phase
            current_step = min(phase, len(step_sizes) - 1)
            pos_step, angle_step = step_sizes[current_step]
            
            # Adjust step sizes dynamically
            if iteration > max_iterations * 0.7:
                pos_step *= 0.5
                angle_step *= 0.5
            elif iteration > max_iterations * 0.5:
                pos_step *= 0.7
                angle_step *= 0.7
            
            # Select a random hexagon to perturb
            hex_idx = random.randint(0, 11)
            
            # Create perturbed version
            perturbed = current_best_positions.copy()
            
            # Perturb position with adaptive step size
            perturbed[hex_idx, 0] += random.uniform(-pos_step, pos_step)
            perturbed[hex_idx, 1] += random.uniform(-pos_step, pos_step)
            
            # Perturb angle with appropriate scaling
            perturbed[hex_idx, 2] += random.uniform(-angle_step, angle_step)
            perturbed[hex_idx, 2] = max(0, min(360, perturbed[hex_idx, 2]))
            
            # Evaluate new configuration
            try:
                valid, outer_radius, inv_radius = evaluate_solution(perturbed)
                if valid and inv_radius > current_best_score:
                    current_best_score = inv_radius
                    current_best_positions = perturbed.copy()
                    last_improvement = iteration
                    stagnation_counter = 0
                elif valid and random.random() < 0.08:  # Occasionally accept slightly worse solutions
                    # This helps escape local optima - higher probability for more exploration
                    current_best_score = inv_radius
                    current_best_positions = perturbed.copy()
                else:
                    stagnation_counter += 1
                    
                    # If stuck for too long, increase randomness to escape
                    if stagnation_counter > 40 and random.random() < 0.2:
                        # Make a more significant random perturbation
                        hex_idx = random.randint(0, 11)
                        perturbed[hex_idx, 0] += random.uniform(-0.2, 0.2)
                        perturbed[hex_idx, 1] += random.uniform(-0.2, 0.2)
                        perturbed[hex_idx, 2] += random.uniform(-10, 10)
                        perturbed[hex_idx, 2] = max(0, min(360, perturbed[hex_idx, 2]))
                        
                        valid, outer_radius, inv_radius = evaluate_solution(perturbed)
                        if valid and inv_radius > current_best_score:
                            current_best_score = inv_radius
                            current_best_positions = perturbed.copy()
                            last_improvement = iteration
                            stagnation_counter = 0
                            
            except Exception as e:
                # Skip invalid configurations
                pass
            
            iteration += 1
            
            # Move to next phase when threshold reached
            if iteration >= (phase + 1) * 600 and phase < max_phases - 1:
                phase += 1
        
        # Update global best if this restart was better
        if current_best_score > best_score:
            best_score = current_best_score
            best_positions = current_best_positions.copy()
    
    # If no restart produced a valid solution, return the original
    if best_positions is None:
        best_positions = np.array(initial_positions)
    
    return best_positions, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start with the known high-quality configuration using exact mathematical values
    initial_positions = [
        [0, 0, 0],              # center
        [0, 1.9419123, 0],      # top
        [0, -1.9419123, 0],     # bottom
        [1.680872, 0.970956, 0],# top-right
        [-1.680872, 0.970956, 0],# top-left
        [1.680872, -0.970956, 0],# bottom-right
        [-1.680872, -0.970956, 0],# bottom-left
        [3.361744, 0, 0],       # far right
        [-3.361744, 0, 0],      # far left
        [1.680872, 2.912868, 0],# top far-right
        [-1.680872, 2.912868, 0],# top far-left
        [1.680872, -2.912868, 0],# bottom far-right
    ]
    
    # Use advanced local search optimization with better escape mechanisms
    refined_positions, final_score = advanced_local_search(initial_positions, max_time=45)
    
    # Calculate final outer hexagon size
    outer_side_length = calculate_outer_hexagon_side_length(refined_positions)
    
    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Return the optimized configuration
    return refined_positions, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
