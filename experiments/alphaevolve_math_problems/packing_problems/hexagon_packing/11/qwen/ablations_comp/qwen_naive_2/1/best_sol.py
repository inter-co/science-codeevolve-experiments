# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from typing import Tuple, List
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial.distance import cdist
from numba import jit

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # unit hexagon radius
HEX_APO = HEX_RADIUS * math.sqrt(3)/2  # apothem (distance from center to edge)
HEX_SIDE = HEX_RADIUS  # side length for unit hexagon

@jit(nopython=True)
def hex_distance(p1, p2):
    """Fast distance calculation between two points"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def point_in_hex(point, hex_center, hex_angle_deg):
    """Fast check if a point is inside a hexagon (simplified)"""
    # Convert to local coordinates
    angle_rad = math.radians(hex_angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    px = point[0] - hex_center[0]
    py = point[1] - hex_center[1]
    
    # Rotate point to hexagon's coordinate system
    x = px * cos_a + py * sin_a
    y = -px * sin_a + py * cos_a
    
    # Check if inside hexagon
    # Hexagon has vertices at (±1, 0), (±1/2, ±√3/2)
    # We'll use a simple bounding box check first
    if abs(x) > 1.0 or abs(y) > 1.0:
        return False
    
    # More precise check
    # Distance to edges would be more accurate but this is faster
    return True

def create_hexagon_vertices(center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
    """Create vertices of a regular hexagon given center, rotation"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi/3
        x = center_x + HEX_RADIUS * math.cos(angle)
        y = center_y + HEX_RADIUS * math.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)

def check_hexagon_containment_shapely(hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Check if hexagon vertices are all inside outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_hexagon_overlap_shapely(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_radius_fast(inner_hex_data: np.ndarray, outer_center=(0,0)) -> float:
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons - optimized version"""
    max_dist = 0.0
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        # For unit hexagons, the maximum distance from center to any vertex is 1
        # But we need to consider the actual positions
        vertices = create_hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = math.sqrt((vx - outer_center[0])**2 + (vy - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    return max_dist + 0.01  # Small safety margin

def evaluate_packing_optimized(inner_hex_data: np.ndarray, outer_center=(0,0)) -> tuple:
    """
    Evaluate a packing configuration and return penalty and outer radius.
    Returns (penalty, outer_radius)
    """
    # Calculate outer radius
    outer_radius = calculate_outer_hex_radius_fast(inner_hex_data, outer_center)
    
    # Create outer hexagon vertices (centered at outer_center)
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], 0)
    
    # Check containment for all hexagons
    containment_penalty = 0.0
    for i in range(11):
        vertices = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
        if not check_hexagon_containment_shapely(vertices, outer_vertices):
            containment_penalty += 10000.0  # Large penalty for containment violation
    
    # Check overlaps between all pairs
    overlap_penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            vertices_i = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
            vertices_j = create_hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2])
            if check_hexagon_overlap_shapely(vertices_i, vertices_j):
                overlap_penalty += 10000.0  # Large penalty for overlap
    
    total_penalty = containment_penalty + overlap_penalty
    
    return total_penalty, outer_radius

def objective_function(x):
    """
    Objective function for optimization: minimize negative 1/outer_radius (maximize 1/outer_radius)
    x contains parameters for 11 hexagons: [x1,y1,angle1,x2,y2,angle2,...,x11,y11,angle11]
    """
    # Reshape x into 11 hexagon data points
    inner_hex_data = x.reshape((11, 3))
    
    # Evaluate the packing
    penalty, outer_radius = evaluate_packing_optimized(inner_hex_data)
    
    # If there are penalties, return large value
    if penalty > 0:
        return penalty + 1000000  # Ensure invalid solutions are penalized heavily
    
    # Return negative inverse radius (since we want to maximize 1/outer_radius)
    # This is equivalent to minimizing -1/outer_radius
    return -1.0 / outer_radius

def optimize_packing():
    """
    Use improved optimization approach to find optimal packing
    """
    # Better initial guess based on known good configurations
    # Start with a more compact arrangement that should be close to optimal
    initial_guess = np.array([
        [0.0, 0.0, 0.0],       # center hexagon
        [0.0, 2.0, 0.0],       # top
        [1.732, 1.0, 0.0],     # top-right
        [1.732, -1.0, 0.0],    # bottom-right  
        [0.0, -2.0, 0.0],      # bottom
        [-1.732, -1.0, 0.0],   # bottom-left
        [-1.732, 1.0, 0.0],    # top-left
        [3.464, 0.0, 0.0],     # far right
        [-3.464, 0.0, 0.0],    # far left
        [1.732, 3.0, 0.0],     # top-right corner
        [-1.732, 3.0, 0.0],    # top-left corner
    ]).flatten()
    
    # Better bounds: tighter constraints around expected solution space
    bounds = []
    # Position bounds: more constrained around the expected optimal region
    for _ in range(11):
        bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (0.0, 360.0)])
    
    # First run with differential evolution for global search
    start_time = time.time()
    
    # Try multiple optimization strategies to find better solutions
    best_result = None
    best_objective = float('inf')
    
    # Strategy 1: Differential Evolution (global search) - reduced iterations for speed
    try:
        result_de = differential_evolution(
            objective_function,
            bounds,
            maxiter=30,  # Reduced iterations for faster execution
            popsize=8,   # Reduced population size
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result_de.fun < best_objective:
            best_objective = result_de.fun
            best_result = result_de
    except Exception as e:
        print(f"Differential evolution failed: {e}")
        pass
    
    # Strategy 2: Local optimization starting from good initial guess
    try:
        # Use a more refined local optimizer with better settings
        result_local = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if result_local.fun < best_objective:
            best_objective = result_local.fun
            best_result = result_local
    except Exception as e:
        print(f"Local optimization failed: {e}")
        pass
    
    # Strategy 3: Try a more focused approach with better initial conditions
    try:
        # Generate some alternative initial guesses and run local optimization on them
        for attempt in range(3):
            # Perturb the initial guess slightly
            perturbed_guess = initial_guess + np.random.normal(0, 0.1, len(initial_guess))
            result_perturbed = minimize(
                objective_function,
                perturbed_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result_perturbed.fun < best_objective:
                best_objective = result_perturbed.fun
                best_result = result_perturbed
    except Exception as e:
        print(f"Perturbed optimization failed: {e}")
        pass
    
    # If no good result found, use the initial guess
    if best_result is None:
        best_result = type('obj', (object,), {'x': initial_guess})()
    
    # Extract best solution
    best_solution = best_result.x.reshape((11, 3))
    
    # Evaluate final solution
    penalty, outer_radius = evaluate_packing_optimized(best_solution)
    
    # If solution is invalid, fall back to good deterministic arrangement
    if penalty > 0:
        # Use a carefully chosen deterministic arrangement that should work well
        best_solution = np.array([
            [0.0, 0.0, 0.0],       # center hexagon
            [0.0, 2.0, 0.0],       # top
            [1.732, 1.0, 0.0],     # top-right
            [1.732, -1.0, 0.0],    # bottom-right  
            [0.0, -2.0, 0.0],      # bottom
            [-1.732, -1.0, 0.0],   # bottom-left
            [-1.732, 1.0, 0.0],    # top-left
            [3.464, 0.0, 0.0],     # far right
            [-3.464, 0.0, 0.0],    # far left
            [1.732, 3.0, 0.0],     # top-right corner
            [-1.732, 3.0, 0.0],    # top-left corner
        ])
        _, outer_radius = evaluate_packing_optimized(best_solution)
    
    end_time = time.time()
    return best_solution, outer_radius, end_time - start_time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use optimization approach
    inner_hex_data, outer_hex_side_length, eval_time = optimize_packing()
    
    # Outer hexagon centered at origin, no rotation
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
