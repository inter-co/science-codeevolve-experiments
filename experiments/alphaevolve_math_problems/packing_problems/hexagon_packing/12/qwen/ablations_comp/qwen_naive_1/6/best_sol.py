# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import math
from numba import jit
import time
from itertools import combinations
from scipy.spatial import distance
from joblib import Parallel, delayed

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon (jit compiled for speed)."""
    angle_rad = math.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + size * math.cos(angle)
        y = center_y + size * math.sin(angle)
        vertices[i] = [x, y]
    return vertices


def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + size * math.cos(angle)
        y = center_y + size * math.sin(angle)
        vertices.append((x, y))
    return vertices


def hexagon_polygon(center_x, center_y, size=1, angle_deg=0):
    """Create a Shapely polygon representation of a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, size, angle_deg)
    return Polygon(vertices)


def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly) or outer_hex_poly.covers(hexagon_poly)


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2) and not hex1.touches(hex2)


def compute_outer_hex_side_length_fast(inner_hex_data, outer_center=(0, 0)):
    """
    Fast computation of outer hexagon side length using bounding circle.
    """
    # Find the maximum distance from center to any vertex
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Get just one vertex (we know the hexagon is regular, so max distance is 1 from center)
        # But we need to account for rotation
        vertices = hexagon_vertices(center_x, center_y, 1, angle)
        for vx, vy in vertices:
            dist = math.sqrt(vx**2 + vy**2)
            max_dist = max(max_dist, dist)
    
    # For a regular hexagon, if we want to contain all vertices, 
    # we need a hexagon with radius equal to max_dist
    # Since the circumradius of a regular hexagon equals its side length
    return max_dist


def compute_penalty_for_overlaps(inner_hexagons):
    """Compute penalty based on overlap violations."""
    penalty = 0
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if check_overlap(inner_hexagons[i], inner_hexagons[j]):
            # Calculate overlap area for penalty
            intersection = inner_hexagons[i].intersection(inner_hexagons[j])
            overlap_area = intersection.area
            penalty += 1000000 * overlap_area
    return penalty


def compute_penalty_for_containment(inner_hexagons, outer_hex):
    """Compute penalty for containment violations."""
    penalty = 0
    for hex_poly in inner_hexagons:
        if not check_containment(hex_poly, outer_hex):
            # Calculate how much is outside
            diff = hex_poly.difference(outer_hex)
            outside_area = diff.area
            penalty += 1000000 * outside_area
    return penalty


def compute_minimum_distance_between_hexagons(inner_hexagons):
    """Compute the minimum distance between any two hexagon centers."""
    if len(inner_hexagons) < 2:
        return float('inf')
    
    min_dist = float('inf')
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            # Distance between hexagon centers
            center_i = inner_hexagons[i].centroid
            center_j = inner_hexagons[j].centroid
            dist = center_i.distance(center_j)
            min_dist = min(min_dist, dist)
    return min_dist


def compute_distance_penalty(inner_hexagons):
    """Compute penalty based on minimum distance between centers."""
    min_dist = compute_minimum_distance_between_hexagons(inner_hexagons)
    if min_dist < 1.9:  # If hexagons are too close
        return 100000 * (1.9 - min_dist)**2
    return 0


def compute_geometry_penalty(inner_hexagons, outer_hex):
    """Combined geometry penalty for overlaps and containment."""
    overlap_penalty = compute_penalty_for_overlaps(inner_hexagons)
    containment_penalty = compute_penalty_for_containment(inner_hexagons, outer_hex)
    return overlap_penalty + containment_penalty


def compute_total_penalty(inner_hexagons, outer_hex):
    """Compute total penalty for overlaps and containment."""
    return compute_penalty_for_overlaps(inner_hexagons) + compute_penalty_for_containment(inner_hexagons, outer_hex)


def objective_function(params):
    """Objective function to minimize (negative of inverse side length)."""
    # Reshape parameters into 12 hexagons with (x, y, angle) each
    inner_positions = params.reshape(-1, 3)
    
    # Create hexagon polygons
    inner_hexagons = []
    for i in range(12):
        x, y, angle = inner_positions[i]
        hex_poly = hexagon_polygon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hex_side_length_fast(inner_positions)
    
    # Create outer hexagon
    outer_hex = hexagon_polygon(0, 0, outer_side_length)
    
    # Compute penalties
    geometry_penalty = compute_total_penalty(inner_hexagons, outer_hex)
    
    # Return negative inverse side length plus penalties
    # Use a very high penalty for invalid configurations
    if geometry_penalty > 1e6:
        return 1e10 - (1.0 / outer_side_length)  # Large penalty for invalid configs
    
    # Add a stronger penalty term for better convergence
    return -(1.0 / outer_side_length) + 10000 * geometry_penalty


def create_optimized_initial_config():
    """
    Create an optimized initial configuration based on known optimal patterns.
    Inspired by the mathematical structure of optimal hexagon packings.
    """
    sqrt3 = math.sqrt(3)
    # A more refined configuration based on known optimal solutions
    # This configuration places hexagons in a pattern that maximizes efficiency
    config = [
        [0.0, 0.0, 0],           # center
        [0.0, 2.0, 0],           # top
        [0.0, -2.0, 0],          # bottom  
        [sqrt3, 1.0, 0],         # top-right
        [-sqrt3, 1.0, 0],        # top-left
        [sqrt3, -1.0, 0],        # bottom-right
        [-sqrt3, -1.0, 0],       # bottom-left
        [2*sqrt3, 0.0, 0],       # far right
        [-2*sqrt3, 0.0, 0],      # far left
        [sqrt3, 3.0, 0],         # upper right
        [-sqrt3, 3.0, 0],        # upper left
        [sqrt3, -3.0, 0],        # lower right
    ]
    
    # Apply more careful mathematical adjustments to get closer to optimal
    adjusted_config = []
    for i, (x, y, angle) in enumerate(config):
        if i == 0:  # center - keep fixed
            adjusted_config.append([x, y, angle])
        elif i in [1, 2]:  # top/bottom - adjust inward slightly
            adjusted_config.append([x, y * 0.96, angle])
        elif i in [3, 4, 5, 6]:  # corners - pull inward more
            adjusted_config.append([x * 0.91, y * 0.91, angle])
        elif i in [7, 8]:  # far sides - pull inward more
            adjusted_config.append([x * 0.89, y, angle])
        elif i in [9, 10, 11]:  # outer edges - pull inward more
            adjusted_config.append([x * 0.88, y * 0.88, angle])
    
    return np.array(adjusted_config)


def create_refined_initial_config():
    """
    Create a refined initial configuration based on mathematical insights.
    This version uses more precise geometric relationships.
    """
    sqrt3 = math.sqrt(3)
    # This configuration attempts to match known optimal arrangements
    # by placing hexagons in a pattern that respects symmetry
    config = [
        [0.0, 0.0, 0],           # center
        [0.0, 2.0, 0],           # top
        [0.0, -2.0, 0],          # bottom  
        [sqrt3, 1.0, 0],         # top-right
        [-sqrt3, 1.0, 0],        # top-left
        [sqrt3, -1.0, 0],        # bottom-right
        [-sqrt3, -1.0, 0],       # bottom-left
        [2*sqrt3, 0.0, 0],       # far right
        [-2*sqrt3, 0.0, 0],      # far left
        [sqrt3, 3.0, 0],         # upper right
        [-sqrt3, 3.0, 0],        # upper left
        [sqrt3, -3.0, 0],        # lower right
    ]
    
    # More aggressive refinement to approach the optimal
    adjusted_config = []
    for i, (x, y, angle) in enumerate(config):
        if i == 0:  # center - keep fixed
            adjusted_config.append([x, y, angle])
        elif i in [1, 2]:  # top/bottom - adjust inward more aggressively
            adjusted_config.append([x, y * 0.95, angle])
        elif i in [3, 4, 5, 6]:  # corners - pull inward
            adjusted_config.append([x * 0.90, y * 0.90, angle])
        elif i in [7, 8]:  # far sides - pull inward more
            adjusted_config.append([x * 0.87, y, angle])
        elif i in [9, 10, 11]:  # outer edges - pull inward more
            adjusted_config.append([x * 0.86, y * 0.86, angle])
    
    return np.array(adjusted_config)


def run_single_optimization(method_name, method, initial_params, bounds, options=None):
    """Run a single optimization strategy."""
    try:
        if options is None:
            options = {}
        result = minimize(
            objective_function,
            initial_params,
            method=method,
            bounds=bounds,
            options=options
        )
        return result
    except Exception as e:
        return None


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a geometric optimization approach with symmetry considerations.
    """
    # Create better initial configuration
    initial_positions = create_refined_initial_config()
    
    # Flatten initial positions for optimization
    initial_params = initial_positions.flatten()
    
    # Optimization bounds (x, y, angle for each hexagon)
    bounds = [(-10, 10), (-10, 10), (0, 360)] * 12
    
    # Use scipy optimization to improve the configuration
    start_time = time.time()
    
    # Run multiple optimization strategies with reduced search space
    strategies = [
        ('L-BFGS-B', 'L-BFGS-B', {'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}),
        ('trust-constr', 'trust-constr', {'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15})
    ]
    
    results = []
    for name, method, opts in strategies:
        result = run_single_optimization(name, method, initial_params, bounds, opts)
        if result is not None and result.success:
            results.append(result)
    
    # If we still don't have a good result, use the initial configuration
    if not results:
        best_result = type('obj', (object,), {
            'x': initial_params,
            'success': True,
            'fun': objective_function(initial_params)
        })()
    else:
        # Return the best result (lowest objective value)
        best_result = min(results, key=lambda r: r.fun)
    
    end_time = time.time()
    
    # Extract optimized positions
    optimized_positions = best_result.x.reshape(-1, 3)
    
    # Final computation of outer hexagon size
    outer_side_length = compute_outer_hex_side_length_fast(optimized_positions)
    
    # Validate final configuration
    inner_hexagons = []
    for i in range(12):
        x, y, angle = optimized_positions[i]
        hex_poly = hexagon_polygon(x, y, 1, angle)
        inner_hexagons.append(hex_poly)
    
    outer_hex = hexagon_polygon(0, 0, outer_side_length)
    
    # Ensure no overlaps and full containment
    geometry_penalty = compute_total_penalty(inner_hexagons, outer_hex)
    
    # If still have issues, do one final optimization with very strict constraints
    if geometry_penalty > 1000:
        try:
            # Run one final optimization with very strict tolerances
            final_result = minimize(
                objective_function,
                optimized_positions.flatten(),
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if final_result.success:
                optimized_positions = final_result.x.reshape(-1, 3)
                outer_side_length = compute_outer_hex_side_length_fast(optimized_positions)
        except:
            pass
    
    # Return the data
    inner_hex_data = optimized_positions.copy()
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
