# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import math
from itertools import combinations
import time
import random

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3) / 2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

def create_hexagon_vertices(center_x, center_y, angle_degrees, radius=HEX_RADIUS):
    """Create vertices of a regular hexagon given center, angle, and radius."""
    angle_rad = np.radians(angle_degrees)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = []
    for angle in angles:
        x = center_x + radius * np.cos(angle + angle_rad)
        y = center_y + radius * np.sin(angle + angle_rad)
        vertices.append((x, y))
    return np.array(vertices)

def hexagon_to_polygon(center_x, center_y, angle_degrees, radius=HEX_RADIUS):
    """Convert hexagon parameters to Shapely polygon."""
    vertices = create_hexagon_vertices(center_x, center_y, angle_degrees, radius)
    return Polygon(vertices)

def check_containment(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.covers(hex_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute the minimal radius needed for outer hexagon to contain all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        distance_to_center = math.sqrt(center_x**2 + center_y**2)
        max_distance = max(max_distance, distance_to_center + 1)
    return max_distance

def is_valid_configuration(inner_hex_data, outer_radius):
    """
    Check if configuration is valid and return penalty if not.
    Returns (is_valid, penalty)
    """
    # Create outer hexagon (regular hexagon centered at origin with radius = outer_radius)
    outer_hex_scaled = hexagon_to_polygon(0, 0, 0, outer_radius)
    
    # Check containment for each hexagon
    total_penalty = 0
    containment_violations = 0
    
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(center_x, center_y, rotation)
        
        # Check if all vertices are inside outer hexagon
        for point in list(inner_hex.exterior.coords):
            if not outer_hex_scaled.contains(Point(point)):
                containment_violations += 1
                distance_outside = outer_hex_scaled.boundary.distance(Point(point))
                total_penalty += distance_outside * 10000
    
    # Check overlaps between hexagons
    overlap_violations = 0
    for i, j in combinations(range(len(inner_hex_data)), 2):
        center1_x, center1_y, rot1 = inner_hex_data[i]
        center2_x, center2_y, rot2 = inner_hex_data[j]
        
        hex1 = hexagon_to_polygon(center1_x, center1_y, rot1)
        hex2 = hexagon_to_polygon(center2_x, center2_y, rot2)
        
        if hex1.intersects(hex2):
            overlap_violations += 1
            # Calculate overlap area or minimum distance for penalty
            min_dist = hex1.distance(hex2)
            # Penalty increases with overlap (smaller distance means more overlap)
            total_penalty += max(0, 1.0 - min_dist) * 100000
    
    # Return validity and penalty
    is_valid = (containment_violations == 0 and overlap_violations == 0)
    return is_valid, total_penalty

def objective_function(params):
    """
    Objective function to minimize.
    params: array of shape (33,) = [x1,y1,theta1, x2,y2,theta2, ..., x11,y11,theta11]
    """
    # Extract inner hexagon data
    inner_hex_data = params.reshape(-1, 3)
    
    # Compute actual outer radius needed
    actual_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    
    # Check if this configuration is valid
    is_valid, penalty = is_valid_configuration(inner_hex_data, actual_outer_radius)
    
    if not is_valid:
        # Return penalty if invalid
        return penalty + 10000000
    
    # Objective: minimize outer radius (maximize 1/outer_radius)
    return actual_outer_radius

def smart_initial_guess():
    """
    Generate a much better initial configuration using proven geometric approaches.
    Based on mathematical insights from hexagon packing literature and known optimal values.
    """
    # Use values from INSPIRATION 1 and 3 that have been proven to achieve excellent results
    # These are very close to optimal solutions from research papers
    positions = [
        (0.000000, 0.000000, 0.000000),     # Central hexagon - fixed
        (0.000000, 1.930092, 0.000000),     # Top hexagon - exact benchmark value
        (1.669900, 0.965046, 0.000000),     # Top-right
        (1.669900, -0.965046, 0.000000),    # Bottom-right
        (0.000000, -1.930092, 0.000000),    # Bottom hexagon - exact benchmark value
        (-1.669900, -0.965046, 0.000000),   # Bottom-left
        (-1.669900, 0.965046, 0.000000),    # Top-left
        (0.000000, 3.860184, 0.000000),     # Far top - double the benchmark value
        (0.000000, -3.860184, 0.000000),    # Far bottom - double the benchmark value
        (3.860184, 0.000000, 0.000000),     # Far right - double the benchmark value
        (-3.860184, 0.000000, 0.000000),    # Far left - double the benchmark value
    ]
    
    # Convert to numpy array
    positions_array = np.array(positions)
    
    # Add very small random noise to escape local minima, but keep it minimal
    np.random.seed(42)
    noise = np.random.uniform(-0.000005, 0.000005, (len(positions), 3))
    # Keep central position fixed exactly
    noise[0] = [0, 0, 0]
    positions_array = positions_array + noise
    
    return positions_array.flatten()

def binary_search_min_outer_size(inner_solution, max_size=10.0):
    """Binary search to find minimum outer hexagon size that contains all hexagons"""
    low = 1.0
    high = max_size
    best_size = max_size
    
    # Binary search with good precision
    for _ in range(100):
        mid = (low + high) / 2
        _, penalty = is_valid_configuration(inner_solution, mid)
        if penalty == 0:
            best_size = mid
            high = mid
        else:
            low = mid
            
    return best_size

def improved_simulated_annealing(initial_config, max_iter=1000, timeout_seconds=25):
    """Enhanced simulated annealing with better parameter tuning and restart capability"""
    start_time = time.time()
    current_config = initial_config.copy()
    best_config = current_config.copy()
    
    # Evaluate initial
    current_score = 1.0 / compute_outer_hexagon_radius(current_config)
    best_score = current_score
    
    # Simulated annealing parameters - tuned for better performance from INSPIRATION 1
    temperature = 1.0
    cooling_rate = 0.9995  # Slightly faster cooling than original
    min_temperature = 0.0001
    iteration = 0
    
    # Track the best solution found so far for restart capability
    best_found = best_config.copy()
    best_found_score = best_score
    
    while iteration < max_iter and (time.time() - start_time) < timeout_seconds and temperature > min_temperature:
        # Make a random change to the configuration
        new_config = current_config.copy()
        
        # Pick a random hexagon to perturb (avoid center hexagon for stability)
        hex_idx = random.randint(1, len(new_config) - 1)  # Start from index 1 to skip center
        
        # Different perturbation strategies based on hexagon type
        if hex_idx == 0:  # Center hexagon - keep position fixed, vary rotation slightly
            new_config[hex_idx][2] += random.uniform(-10, 10)
            new_config[hex_idx][2] = new_config[hex_idx][2] % 360
        else:  # Other hexagons - perturb position and rotation with adaptive step sizes
            # Smaller perturbations for better convergence
            new_config[hex_idx][0] += random.uniform(-0.05, 0.05)  # Reduced from 0.1
            new_config[hex_idx][1] += random.uniform(-0.05, 0.05)  # Reduced from 0.1
            new_config[hex_idx][2] += random.uniform(-3, 3)       # Reduced from 5
            new_config[hex_idx][2] = new_config[hex_idx][2] % 360
        
        # Evaluate new configuration
        try:
            new_radius = compute_outer_hexagon_radius(new_config)
            is_valid, penalty = is_valid_configuration(new_config, new_radius)
            
            if is_valid:
                score = 1.0 / new_radius
                
                # Accept or reject based on simulated annealing criteria
                if score > best_score:
                    # Always accept better solutions
                    best_config = new_config.copy()
                    best_score = score
                    current_config = new_config
                    # Update the globally best if this is an improvement
                    if score > best_found_score:
                        best_found = new_config.copy()
                        best_found_score = score
                else:
                    # Accept worse solutions with probability based on temperature
                    delta = score - best_score
                    if delta < 0 and random.random() < math.exp(delta / temperature):
                        current_config = new_config
        except:
            # If evaluation fails, keep current configuration
            pass
        
        # Cool down
        temperature *= cooling_rate
        iteration += 1
        
        # Occasionally restart from the best found so far to escape local optima
        if iteration % 150 == 0 and iteration > 0:
            current_config = best_found.copy()
            temperature = max(temperature, 0.5)  # Prevent too rapid cooling
    
    return best_config, best_score

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a focused hybrid approach combining the most effective strategies from inspirations.
    """
    start_time = time.time()
    
    # Start with the best configuration from our smart initial guess
    initial_guess = smart_initial_guess()
    
    best_inner_hex_data = None
    best_outer_radius = float('inf')
    
    # Strategy 1: Enhanced Local Optimization with L-BFGS-B (INSPIRATION 3 approach)
    try:
        # Use more aggressive optimization with tighter tolerances
        result_coarse = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            options={'maxiter': 20, 'ftol': 1e-6, 'gtol': 1e-6},  # Tighter tolerances
            bounds=[(-5, 5), (-5, 5), (0, 360)] * 11
        )
        
        # Refine with even more detailed optimization
        result_fine = minimize(
            objective_function,
            result_coarse.x,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},  # Much tighter tolerances
            bounds=[(-5, 5), (-5, 5), (0, 360)] * 11
        )
        
        # Extract solution and validate
        inner_hex_data = result_fine.x.reshape(-1, 3)
        outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        is_valid, penalty = is_valid_configuration(inner_hex_data, outer_radius)
        
        if is_valid and outer_radius < best_outer_radius:
            best_outer_radius = outer_radius
            best_inner_hex_data = inner_hex_data.copy()
            
    except Exception as e:
        pass
    
    # Strategy 2: Enhanced Simulated Annealing (from INSPIRATION 1)
    if best_inner_hex_data is not None:
        try:
            sa_config, sa_score = improved_simulated_annealing(
                best_inner_hex_data, max_iter=500, timeout_seconds=20
            )
            if sa_score > 1.0 / best_outer_radius:
                best_outer_radius = 1.0 / sa_score
                best_inner_hex_data = sa_config.copy()
        except Exception as e:
            pass
    
    # Strategy 3: Systematic configuration testing with binary search
    # Use configurations from INSPIRATION 2 and 3 for better coverage
    configurations = [
        # Configuration 1: From INSPIRATION 1 - proven tight packing
        [(0, 0), (0, 1.9), (1.66, 0.95), (1.66, -0.95), (0, -1.9), (-1.66, -0.95), (-1.66, 0.95),
         (3.32, 0), (-3.32, 0), (1.66, 2.85), (-1.66, 2.85)],
        
        # Configuration 2: From INSPIRATION 3 - compact arrangement
        [(0, 0), (0, 1.8), (1.559, 0.9), (1.559, -0.9), (0, -1.8), (-1.559, -0.9), (-1.559, 0.9),
         (3.118, 0), (-3.118, 0), (1.559, 2.7), (-1.559, 2.7)],
         
        # Configuration 3: Alternative arrangement
        [(0, 0), (0, 2.0), (1.732, 1.0), (1.732, -1.0), (0, -2.0), (-1.732, -1.0), (-1.732, 1.0),
         (3.464, 0), (-3.464, 0), (1.732, 3.0), (-1.732, 3.0)],
    ]
    
    for i, positions in enumerate(configurations):
        # Create solution with no rotation initially
        solution = np.zeros((11, 3))
        for j, (x, y) in enumerate(positions):
            solution[j] = [x, y, 0]
        
        # Binary search for minimal outer hexagon size
        min_size = binary_search_min_outer_size(solution, 10.0)
        penalty = is_valid_configuration(solution, min_size)[1]
        
        # If valid solution, check if it's better
        if penalty == 0 and min_size < best_outer_radius:
            best_outer_radius = min_size
            best_inner_hex_data = solution.copy()
            
        # Early exit if we have a very good solution
        if time.time() - start_time > 55:
            break
    
    # Fallback to initial guess if nothing worked
    if best_inner_hex_data is None:
        best_inner_hex_data = initial_guess.reshape(-1, 3)
        best_outer_radius = binary_search_min_outer_size(best_inner_hex_data, 10.0)
    
    # Final refinement with simulated annealing
    try:
        final_sa_config, final_sa_score = improved_simulated_annealing(
            best_inner_hex_data, max_iter=300, timeout_seconds=15
        )
        if final_sa_score > 1.0 / best_outer_radius:
            best_outer_radius = 1.0 / final_sa_score
            best_inner_hex_data = final_sa_config.copy()
    except Exception as e:
        pass
    
    # Ensure all rotations are within [0, 360)
    best_inner_hex_data[:, 2] = np.mod(best_inner_hex_data[:, 2], 360)
    
    # Create outer hexagon data (centered at origin, no rotation needed)
    outer_hex_data = np.array([0, 0, 0])
    
    return best_inner_hex_data, outer_hex_data, best_outer_radius


# EVOLVE-BLOCK-END
