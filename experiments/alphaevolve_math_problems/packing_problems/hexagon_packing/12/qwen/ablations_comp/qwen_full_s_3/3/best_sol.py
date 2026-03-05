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
    
    # Binary search for tightest fit with high precision (like inspiration 1)
    min_radius = 0.1
    max_radius = 10.0
    
    # Use more iterations for better precision (45 iterations like inspiration 1)
    for _ in range(45):  # Even higher precision
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

def simulated_annealing_optimize(initial_positions, max_time=30):
    """Use simulated annealing for better optimization"""
    start_time = time.time()
    
    best_positions = np.array(initial_positions)
    best_score = evaluate_solution(best_positions)[2]  # Get inv_radius directly
    
    # Parameters for simulated annealing - tuned for better convergence (like inspiration 1)
    temp = 800.0  # Higher initial temperature
    cooling_rate = 0.996  # Slightly slower cooling for better exploration
    min_temp = 1e-7  # Lower minimum temperature for more thorough search
    
    iteration = 0
    while temp > min_temp and (time.time() - start_time) < max_time:
        # Create a perturbed version
        perturbed = best_positions.copy()
        
        # Select a random hexagon to perturb
        hex_idx = random.randint(0, 11)
        
        # Perturb position with temperature-dependent step size
        step_size = temp / 600.0  # Better step size scaling
        perturbed[hex_idx, 0] += random.uniform(-step_size, step_size)
        perturbed[hex_idx, 1] += random.uniform(-step_size, step_size)
        
        # Perturb angle
        perturbed[hex_idx, 2] += random.uniform(-4, 4)  # Larger angle changes
        perturbed[hex_idx, 2] = max(0, min(360, perturbed[hex_idx, 2]))
        
        # Evaluate new configuration
        try:
            valid, outer_radius, inv_radius = evaluate_solution(perturbed)
            if valid and inv_radius > best_score:
                best_score = inv_radius
                best_positions = perturbed.copy()
        except:
            # Skip invalid configurations
            pass
        
        # Cool down
        temp *= cooling_rate
        iteration += 1
    
    return best_positions, best_score

def multi_start_optimization(max_starts=10, max_time_per_start=4):
    """Run multiple optimization starts with different initial configurations"""
    best_score = -float('inf')
    best_positions = None
    
    # Generate several good initial configurations based on mathematical research
    initial_configs = []
    
    # Configuration 1: Based on mathematical research (optimal known configuration)
    config1 = [
        [0, 0, 0],           # center
        [0, 1.9419123, 0],   # top
        [0, -1.9419123, 0],  # bottom
        [1.680872, 0.970956, 0],   # top-right
        [-1.680872, 0.970956, 0],  # top-left
        [1.680872, -0.970956, 0],  # bottom-right
        [-1.680872, -0.970956, 0], # bottom-left
        [3.361744, 0, 0],     # far right
        [-3.361744, 0, 0],    # far left
        [1.680872, 2.912868, 0],  # top far-right
        [-1.680872, 2.912868, 0], # top far-left
        [1.680872, -2.912868, 0], # bottom far-right
    ]
    initial_configs.append(config1)
    
    # Configuration 2: More spread out arrangement
    config2 = [
        [0, 0, 0],           # center
        [0, 2.0, 0],         # top
        [0, -2.0, 0],        # bottom
        [1.732, 1.0, 0],     # top-right
        [-1.732, 1.0, 0],    # top-left
        [1.732, -1.0, 0],    # bottom-right
        [-1.732, -1.0, 0],   # bottom-left
        [3.464, 0, 0],       # far right
        [-3.464, 0, 0],      # far left
        [1.732, 3.0, 0],     # top far-right
        [-1.732, 3.0, 0],    # top far-left
        [1.732, -3.0, 0],    # bottom far-right
    ]
    initial_configs.append(config2)
    
    # Configuration 3: Hexagonal pattern around center with some variation
    config3 = [
        [0, 0, 0],           # center
        [0, 1.9419123, 0],   # top
        [0, -1.9419123, 0],  # bottom
        [1.680872, 0.970956, 0],   # top-right
        [-1.680872, 0.970956, 0],  # top-left
        [1.680872, -0.970956, 0],  # bottom-right
        [-1.680872, -0.970956, 0], # bottom-left
        [3.361744, 0, 0],     # far right
        [-3.361744, 0, 0],    # far left
        [1.680872, 2.912868, 0],  # top far-right
        [-1.680872, 2.912868, 0], # top far-left
        [1.680872, -2.912868, 0], # bottom far-right
    ]
    # Slightly perturbed version of config1
    config3[0][0] = 0.01  # Small shift to test diversity
    config3[0][1] = -0.01
    initial_configs.append(config3)
    
    # Configuration 4: Another variant with different spacing
    config4 = [
        [0, 0, 0],           # center
        [0, 1.9, 0],         # top
        [0, -1.9, 0],        # bottom
        [1.65, 0.95, 0],     # top-right
        [-1.65, 0.95, 0],    # top-left
        [1.65, -0.95, 0],    # bottom-right
        [-1.65, -0.95, 0],   # bottom-left
        [3.3, 0, 0],         # far right
        [-3.3, 0, 0],        # far left
        [1.65, 2.85, 0],     # top far-right
        [-1.65, 2.85, 0],    # top far-left
        [1.65, -2.85, 0],    # bottom far-right
    ]
    initial_configs.append(config4)
    
    # Configuration 5: Very tight packing approach
    config5 = [
        [0, 0, 0],           # center
        [0, 1.93, 0],        # top
        [0, -1.93, 0],       # bottom
        [1.67, 0.96, 0],     # top-right
        [-1.67, 0.96, 0],    # top-left
        [1.67, -0.96, 0],    # bottom-right
        [-1.67, -0.96, 0],   # bottom-left
        [3.34, 0, 0],        # far right
        [-3.34, 0, 0],       # far left
        [1.67, 2.89, 0],     # top far-right
        [-1.67, 2.89, 0],    # top far-left
        [1.67, -2.89, 0],    # bottom far-right
    ]
    initial_configs.append(config5)
    
    # Configuration 6: Alternative arrangement with different symmetry
    config6 = [
        [0, 0, 0],           # center
        [0, 1.94, 0],        # top
        [0, -1.94, 0],       # bottom
        [1.68, 0.97, 0],     # top-right
        [-1.68, 0.97, 0],    # top-left
        [1.68, -0.97, 0],    # bottom-right
        [-1.68, -0.97, 0],   # bottom-left
        [3.36, 0, 0],        # far right
        [-3.36, 0, 0],       # far left
        [1.68, 2.91, 0],     # top far-right
        [-1.68, 2.91, 0],    # top far-left
        [1.68, -2.91, 0],    # bottom far-right
    ]
    initial_configs.append(config6)
    
    # Configuration 7: Another mathematical arrangement
    config7 = [
        [0, 0, 0],           # center
        [0, 1.935, 0],       # top
        [0, -1.935, 0],      # bottom
        [1.675, 0.965, 0],   # top-right
        [-1.675, 0.965, 0],  # top-left
        [1.675, -0.965, 0],  # bottom-right
        [-1.675, -0.965, 0], # bottom-left
        [3.35, 0, 0],        # far right
        [-3.35, 0, 0],       # far left
        [1.675, 2.905, 0],   # top far-right
        [-1.675, 2.905, 0],  # top far-left
        [1.675, -2.905, 0],  # bottom far-right
    ]
    initial_configs.append(config7)
    
    # Configuration 8: Highly symmetric arrangement
    config8 = [
        [0, 0, 0],           # center
        [0, 1.9419123, 0],   # top
        [0, -1.9419123, 0],  # bottom
        [1.680872, 0.970956, 0],   # top-right
        [-1.680872, 0.970956, 0],  # top-left
        [1.680872, -0.970956, 0],  # bottom-right
        [-1.680872, -0.970956, 0], # bottom-left
        [3.361744, 0, 0],     # far right
        [-3.361744, 0, 0],    # far left
        [1.680872, 2.912868, 0],  # top far-right
        [-1.680872, 2.912868, 0], # top far-left
        [1.680872, -2.912868, 0], # bottom far-right
    ]
    initial_configs.append(config8)
    
    # Configuration 9: Another variation with small adjustments
    config9 = [
        [0, 0, 0],           # center
        [0, 1.9419123, 0],   # top
        [0, -1.9419123, 0],  # bottom
        [1.680872, 0.970956, 0],   # top-right
        [-1.680872, 0.970956, 0],  # top-left
        [1.680872, -0.970956, 0],  # bottom-right
        [-1.680872, -0.970956, 0], # bottom-left
        [3.361744, 0, 0],     # far right
        [-3.361744, 0, 0],    # far left
        [1.680872, 2.912868, 0],  # top far-right
        [-1.680872, 2.912868, 0], # top far-left
        [1.680872, -2.912868, 0], # bottom far-right
    ]
    # Slight adjustment to first few positions
    config9[1][0] = 0.005
    config9[1][1] = 1.9419123 + 0.005
    config9[2][0] = 0.005
    config9[2][1] = -1.9419123 - 0.005
    initial_configs.append(config9)
    
    # Configuration 10: Another variant with different angle assignments
    config10 = [
        [0, 0, 0],           # center
        [0, 1.9419123, 0],   # top
        [0, -1.9419123, 0],  # bottom
        [1.680872, 0.970956, 0],   # top-right
        [-1.680872, 0.970956, 0],  # top-left
        [1.680872, -0.970956, 0],  # bottom-right
        [-1.680872, -0.970956, 0], # bottom-left
        [3.361744, 0, 0],     # far right
        [-3.361744, 0, 0],    # far left
        [1.680872, 2.912868, 0],  # top far-right
        [-1.680872, 2.912868, 0], # top far-left
        [1.680872, -2.912868, 0], # bottom far-right
    ]
    # Rotate some hexagons for variety
    config10[3][2] = 30.0
    config10[4][2] = 15.0
    config10[5][2] = 45.0
    initial_configs.append(config10)
    
    # Try multiple starting configurations
    for i, initial_config in enumerate(initial_configs[:max_starts]):
        try:
            # Run simulated annealing on this initial configuration
            refined_positions, score = simulated_annealing_optimize(initial_config, max_time_per_start)
            
            if score > best_score:
                best_score = score
                best_positions = refined_positions.copy()
                
        except Exception as e:
            continue
    
    # Final refinement with local search if we have a good solution
    if best_positions is not None:
        try:
            # Apply local search refinement with more time
            refined_positions, final_score = simulated_annealing_optimize(best_positions, 5)
            if final_score > best_score:
                best_score = final_score
                best_positions = refined_positions.copy()
        except:
            pass
    
    return best_positions, best_score

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use multi-start optimization with simulated annealing
    refined_positions, final_score = multi_start_optimization(max_starts=10, max_time_per_start=4)
    
    # If we didn't find a good solution, fall back to the original approach
    if refined_positions is None:
        # Original configuration from mathematical optimization studies
        initial_positions = [
            [0, 0, 0],           # center
            [0, 1.9419123, 0],   # top
            [0, -1.9419123, 0],  # bottom
            [1.680872, 0.970956, 0],   # top-right
            [-1.680872, 0.970956, 0],  # top-left
            [1.680872, -0.970956, 0],  # bottom-right
            [-1.680872, -0.970956, 0], # bottom-left
            [3.361744, 0, 0],     # far right
            [-3.361744, 0, 0],    # far left
            [1.680872, 2.912868, 0],  # top far-right
            [-1.680872, 2.912868, 0], # top far-left
            [1.680872, -2.912868, 0], # bottom far-right
        ]
        
        refined_positions, final_score = simulated_annealing_optimize(initial_positions, 8)
    
    # Calculate final outer hexagon size
    outer_side_length = calculate_outer_hexagon_side_length(refined_positions)
    
    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Return the optimized configuration
    return refined_positions, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
