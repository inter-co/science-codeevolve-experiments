# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from typing import Tuple, List
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial.distance import cdist
from numba import jit
import warnings

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # unit hexagon radius
HEX_APO = HEX_RADIUS * math.sqrt(3)/2  # apothem (distance from center to edge)
HEX_SIDE = HEX_RADIUS  # side length for unit hexagon

@jit(nopython=True)
def distance_point_to_line(point, line_start, line_end):
    """Fast computation of point-to-line distance"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

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

def create_outer_hexagon_vertices(center_x: float, center_y: float, outer_radius: float) -> np.ndarray:
    """Create vertices of outer hexagon given center and radius"""
    vertices = []
    for i in range(6):
        angle = i * math.pi/3
        x = center_x + outer_radius * math.cos(angle)
        y = center_y + outer_radius * math.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)

def check_hexagon_containment_fast(hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Fast containment check using vertex checking with proper Shapely usage"""
    try:
        outer_polygon = Polygon(outer_hex_vertices)
        inner_polygon = Polygon(hex_vertices)
        return outer_polygon.contains(inner_polygon)
    except:
        # Fallback for numerical issues
        return False

def check_hexagon_overlap_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Fast overlap check using Shapely with proper error handling"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        # Fallback for numerical issues
        return True  # Assume overlap to avoid invalid configurations

def calculate_outer_hex_radius(inner_hex_data: np.ndarray, outer_center=(0,0)) -> float:
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_dist = 0
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = math.sqrt((vx - outer_center[0])**2 + (vy - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    # Add buffer for safety margin - reduce from 0.1 to 0.05 for better precision
    return max_dist + 0.05

def evaluate_packing(inner_hex_data: np.ndarray, outer_center=(0,0)) -> tuple:
    """
    Evaluate a packing configuration and return penalty and outer radius.
    Returns (penalty, outer_radius)
    """
    # Calculate outer radius
    outer_radius = calculate_outer_hex_radius(inner_hex_data, outer_center)
    
    # Create outer hexagon vertices (centered at outer_center)
    outer_vertices = create_outer_hexagon_vertices(outer_center[0], outer_center[1], outer_radius)
    
    # Check containment for all hexagons
    containment_penalty = 0.0
    for i in range(11):
        vertices = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
        if not check_hexagon_containment_fast(vertices, outer_vertices):
            containment_penalty += 1000.0  # Large penalty for containment violation
    
    # Check overlaps between all pairs
    overlap_penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            vertices_i = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2])
            vertices_j = create_hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], inner_hex_data[j][2])
            if check_hexagon_overlap_fast(vertices_i, vertices_j):
                overlap_penalty += 1000.0  # Large penalty for overlap
    
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
    penalty, outer_radius = evaluate_packing(inner_hex_data)
    
    # If there are penalties, return large value
    if penalty > 0:
        return penalty + 1000000  # Ensure invalid solutions are penalized heavily
    
    # Return negative inverse radius (since we want to maximize 1/outer_radius)
    # This is equivalent to minimizing -1/outer_radius
    return -1.0 / outer_radius

def optimize_packing():
    """
    Use differential evolution to find optimal packing with better strategies
    """
    # Use a more sophisticated initial guess based on known good configurations
    # This is a carefully arranged pattern that should perform better than the previous attempt
    initial_guess = np.array([
        [0.0, 0.0, 0.0],      # center hexagon
        [0.0, 2.0, 0.0],      # top
        [1.732, 1.0, 0.0],    # top-right
        [1.732, -1.0, 0.0],   # bottom-right  
        [0.0, -2.0, 0.0],     # bottom
        [-1.732, -1.0, 0.0],  # bottom-left
        [-1.732, 1.0, 0.0],   # top-left
        [3.464, 0.0, 0.0],    # far right
        [-3.464, 0.0, 0.0],   # far left
        [1.732, 3.0, 0.0],    # top-right corner
        [-1.732, 3.0, 0.0],   # top-left corner
    ]).flatten()
    
    # Better bounds - tighter constraints to focus search
    bounds = []
    # Position bounds: more reasonable range for optimization
    for _ in range(11):
        bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0.0, 360.0)])
    
    # Run optimization with more iterations and better parameters
    start_time = time.time()
    
    # Use a more aggressive optimization strategy with better parameters
    try:
        # Try multiple optimization approaches to improve results
        best_result = None
        best_value = float('-inf')
        
        # Multiple runs with different seeds to increase chance of finding better solution
        for seed_val in [42, 123, 456, 789, 999]:
            try:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=300,  # Increase iterations
                    popsize=30,    # Larger population
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=seed_val,
                    disp=False,
                    atol=1e-10,
                    rtol=1e-10
                )
                
                # Check if this result is better
                if -result.fun > best_value:
                    best_value = -result.fun
                    best_result = result
                    
            except Exception as e:
                continue
        
        # If we found a good result, refine it with local optimization
        if best_result is not None:
            # Local refinement around the best result
            refined_result = minimize(
                lambda x: -objective_function(x),  # We want to maximize, so negate
                best_result.x,
                method='L-BFGS-B',
                bounds=[(b[0], b[1]) for b in bounds],
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            # Use the refined result if it's better
            if refined_result.success and -objective_function(refined_result.x) > -objective_function(best_result.x):
                best_solution = refined_result.x.reshape((11, 3))
            else:
                best_solution = best_result.x.reshape((11, 3))
        else:
            # Fall back to initial guess if nothing worked
            best_solution = initial_guess.reshape((11, 3))
            
    except Exception as e:
        # Fallback to basic approach if something goes wrong
        warnings.warn(f"Optimization failed: {str(e)}")
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=200,
                popsize=25,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            best_solution = result.x.reshape((11, 3))
        except:
            # Last resort - deterministic arrangement
            best_solution = np.array([
                [0.0, 0.0, 0.0],      # center hexagon
                [0.0, 2.0, 0.0],      # top
                [1.732, 1.0, 0.0],    # top-right
                [1.732, -1.0, 0.0],   # bottom-right  
                [0.0, -2.0, 0.0],     # bottom
                [-1.732, -1.0, 0.0],  # bottom-left
                [-1.732, 1.0, 0.0],   # top-left
                [3.464, 0.0, 0.0],    # far right
                [-3.464, 0.0, 0.0],   # far left
                [1.732, 3.0, 0.0],    # top-right corner
                [-1.732, 3.0, 0.0],   # top-left corner
            ])
    
    end_time = time.time()
    
    # Evaluate final solution
    penalty, outer_radius = evaluate_packing(best_solution)
    
    # If solution is invalid, fall back to a good deterministic arrangement
    if penalty > 0:
        # Use a carefully chosen deterministic arrangement that should work well
        best_solution = np.array([
            [0.0, 0.0, 0.0],      # center hexagon
            [0.0, 2.0, 0.0],      # top
            [1.732, 1.0, 0.0],    # top-right
            [1.732, -1.0, 0.0],   # bottom-right  
            [0.0, -2.0, 0.0],     # bottom
            [-1.732, -1.0, 0.0],  # bottom-left
            [-1.732, 1.0, 0.0],   # top-left
            [3.464, 0.0, 0.0],    # far right
            [-3.464, 0.0, 0.0],   # far left
            [1.732, 3.0, 0.0],    # top-right corner
            [-1.732, 3.0, 0.0],   # top-left corner
        ])
        _, outer_radius = evaluate_packing(best_solution)
    
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
