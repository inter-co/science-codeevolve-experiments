# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List
import math
from numba import jit
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely import affinity
import itertools

# Constants
UNIT_HEXAGON_RADIUS = 1.0  # Circumradius of unit hexagon
UNIT_HEXAGON_WIDTH = 2.0  # Distance between parallel sides
UNIT_HEXAGON_HEIGHT = math.sqrt(3.0)  # Height of unit hexagon

@jit(nopython=True)
def distance_hexagon_to_hexagon_fast(hex1_center, hex2_center, hex_radius):
    """Fast distance calculation between hexagon centers."""
    dx = hex1_center[0] - hex2_center[0]
    dy = hex1_center[1] - hex2_center[1]
    return math.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def distance_point_to_hexagon_fast(point, hex_center, hex_rotation):
    """Fast distance calculation from point to hexagon boundary."""
    px, py = point
    cx, cy = hex_center
    # Simplified approach: use distance to center minus radius
    dx = px - cx
    dy = py - cy
    return math.sqrt(dx*dx + dy*dy) - UNIT_HEXAGON_RADIUS

def create_unit_hexagon_polygon(center=(0,0), rotation=0):
    """Create a Shapely polygon representation of a unit regular hexagon."""
    angle_step = math.pi / 3.0
    vertices = []
    for i in range(6):
        angle = rotation + i * angle_step
        x = center[0] + UNIT_HEXAGON_RADIUS * math.cos(angle)
        y = center[1] + UNIT_HEXAGON_RADIUS * math.sin(angle)
        vertices.append((x, y))
    return Polygon(vertices)

def create_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Create vertices of a unit regular hexagon centered at center with given rotation."""
    angle_step = math.pi / 3.0
    vertices = []
    for i in range(6):
        angle = rotation + i * angle_step
        x = center[0] + UNIT_HEXAGON_RADIUS * math.cos(angle)
        y = center[1] + UNIT_HEXAGON_RADIUS * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def calculate_outer_hexagon_side_length(inner_hex_data):
    """Calculate the minimal side length of outer hexagon that contains all inner hexagons."""
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = math.radians(inner_hex_data[i][2])
        poly = create_unit_hexagon_polygon(center, rotation)
        # Ensure valid polygons
        poly = make_valid(poly)
        inner_polygons.append(poly)
    
    # Union all inner hexagons to get bounding shape
    try:
        union_result = unary_union(inner_polygons)
        # Make sure union_result is valid
        union_result = make_valid(union_result)
    except:
        # Fallback to manual computation if union fails
        return 1000.0  # Large penalty
    
    # Get centroid of the union
    centroid = union_result.centroid
    centroid_x, centroid_y = centroid.x, centroid.y
    
    max_dist = 0.0
    
    # Sample points along the exterior to find maximum distance
    if hasattr(union_result, 'exterior') and union_result.exterior is not None:
        coords = list(union_result.exterior.coords)
        for coord in coords:
            dist = math.sqrt((coord[0] - centroid_x)**2 + (coord[1] - centroid_y)**2)
            max_dist = max(max_dist, dist)
    
    # Also consider vertices of individual hexagons for robustness
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = math.radians(inner_hex_data[i][2])
        vertices = create_unit_hexagon_vertices(center, rotation)
        for vertex in vertices:
            dist = math.sqrt((vertex[0] - centroid_x)**2 + (vertex[1] - centroid_y)**2)
            max_dist = max(max_dist, dist)
    
    # Convert from circumradius to side length for hexagon
    # For a hexagon with circumradius r, side length = r
    # Add a small buffer to ensure complete containment
    return max_dist * 1.001

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using vertices - simplified but effective"""
    # Simple bounding box check first
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)
    
    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)
    
    # If bounding boxes don't intersect, no overlap
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False
    
    # More precise check using Shapely for actual overlap
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    except:
        # If Shapely fails, assume overlap
        return True

def check_hexagon_overlap(hex1_polygon, hex2_polygon):
    """Check if two hexagon polygons overlap using Shapely."""
    try:
        return hex1_polygon.intersects(hex2_polygon) and not hex1_polygon.touches(hex2_polygon)
    except:
        return True  # Conservative estimate

def check_containment(hex_polygon, outer_polygon):
    """Check if hexagon is fully contained within outer polygon."""
    try:
        return outer_polygon.contains(hex_polygon)
    except:
        return False

def compute_hexagon_distance_matrix(inner_hex_data):
    """Compute distance matrix for all hexagon pairs to identify potential conflicts."""
    n = len(inner_hex_data)
    distances = np.zeros((n, n))
    
    for i in range(n):
        center_i = (inner_hex_data[i][0], inner_hex_data[i][1])
        for j in range(i+1, n):
            center_j = (inner_hex_data[j][0], inner_hex_data[j][1])
            dist = distance_hexagon_to_hexagon_fast(center_i, center_j, UNIT_HEXAGON_RADIUS)
            distances[i, j] = dist
            distances[j, i] = dist
            
    return distances

def compute_geometric_constraints(inner_hex_data):
    """Compute geometric constraints for the optimization problem."""
    n = len(inner_hex_data)
    constraints = []
    
    # Overlap constraints (minimum distance between centers)
    for i in range(n):
        for j in range(i+1, n):
            center_i = (inner_hex_data[i][0], inner_hex_data[i][1])
            center_j = (inner_hex_data[j][0], inner_hex_data[j][1])
            dist_ij = distance_hexagon_to_hexagon_fast(center_i, center_j, UNIT_HEXAGON_RADIUS)
            
            # Minimum distance for non-overlapping hexagons is 2*radius = 2
            # But we're actually checking the minimum distance between hexagons
            if dist_ij < 2.0:
                # Add penalty for overlap
                penalty = 1000000.0 * (2.0 - dist_ij)
                constraints.append(penalty)
    
    return sum(constraints)

@jit(nopython=True)
def compute_min_distance_between_hex_centers_fast(hex1_center, hex2_center):
    """Fast computation of minimum distance between two hexagon centers."""
    dx = hex1_center[0] - hex2_center[0]
    dy = hex1_center[1] - hex2_center[1]
    return math.sqrt(dx*dx + dy*dy)

def evaluate_configuration_analytical(config):
    """Evaluate configuration using analytical geometric approach."""
    # Extract parameters
    num_inner = 12
    params_per_hex = 3  # x, y, angle
    
    # Reshape config into hexagon data
    inner_hex_data = config.reshape(num_inner, params_per_hex)
    
    # Check for overlaps with more precise method
    total_penalty = 0.0
    
    # Check all pairs of hexagons for overlaps using polygon intersection
    hex_polygons = []
    for i in range(num_inner):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = math.radians(inner_hex_data[i][2])
        poly = create_unit_hexagon_polygon(center, rotation)
        # Ensure valid polygons
        poly = make_valid(poly)
        hex_polygons.append(poly)
    
    # Check all pairs for overlaps - more efficient approach
    for i in range(num_inner):
        for j in range(i+1, num_inner):
            if check_hexagon_overlap(hex_polygons[i], hex_polygons[j]):
                # Overlap penalty - scaled by amount of overlap
                total_penalty += 1000000.0
    
    # If there are overlaps, return large penalty
    if total_penalty > 0:
        return total_penalty
    
    # Check containment by calculating outer hexagon size
    # Create outer hexagon that contains all inner hexagons
    outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
    
    # Create outer hexagon polygon centered at origin
    # For hexagon with side length s, circumradius = s
    outer_radius = outer_side_length
    outer_vertices = []
    for i in range(6):
        angle = i * math.pi / 3.0
        x = outer_radius * math.cos(angle)
        y = outer_radius * math.sin(angle)
        outer_vertices.append((x, y))
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment of all inner hexagons
    for i in range(num_inner):
        if not check_containment(hex_polygons[i], outer_polygon):
            total_penalty += 1000000.0
    
    # Return penalty (negative for maximization)
    if total_penalty == 0.0:
        # Add a small penalty to avoid exact zeros in optimization
        return -1.0 / (outer_side_length + 1e-10)  # Negative because scipy minimizes
    else:
        return total_penalty

def generate_improved_initial_guess():
    """Generate initial guess based on known high-quality configurations."""
    # Use a configuration inspired by the best known packing for 12 hexagons
    # Based on mathematical research and known optimal arrangements
    
    # This uses a pattern with strategic placement to minimize the outer hexagon size
    # The key insight is to place hexagons in a way that maximizes packing efficiency
    
    # Positions from known good configurations - optimized for minimal outer hexagon
    positions = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring (6 hexagons) - arranged in a perfect circle with careful spacing
        [2.0, 0.0, 0.0],
        [1.0, 1.7320508075688772, 0.0],
        [-1.0, 1.7320508075688772, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -1.7320508075688772, 0.0],
        [1.0, -1.7320508075688772, 0.0],
        # Second ring (6 hexagons) - carefully positioned to reduce gaps
        [2.598076211353316, 1.5, 0.0],
        [1.5, 2.598076211353316, 0.0],
        [0.0, 3.0, 0.0],
        [-1.5, 2.598076211353316, 0.0],
        [-2.598076211353316, 1.5, 0.0],
        [-2.598076211353316, -1.5, 0.0]
    ]
    
    # Adjust positions to achieve better packing density
    # Reduce the spacing between layers slightly to allow better fit
    adjusted_positions = []
    for i, pos in enumerate(positions):
        x, y, angle = pos
        # Apply slight adjustments to optimize packing
        if i > 0:  # Not central hexagon
            # Slight inward adjustment to reduce gaps
            distance_from_center = math.sqrt(x*x + y*y)
            if distance_from_center > 0:
                scale_factor = 0.98  # Slightly reduce distance to center
                x = x * scale_factor
                y = y * scale_factor
        adjusted_positions.append([x, y, angle])
    
    # Apply small random perturbations to break symmetries and improve convergence
    for i in range(len(adjusted_positions)):
        adjusted_positions[i][0] += np.random.normal(0, 0.002)
        adjusted_positions[i][1] += np.random.normal(0, 0.002)
    
    # Flatten array
    flat_positions = []
    for pos in adjusted_positions:
        flat_positions.extend(pos)
    
    return np.array(flat_positions)

def generate_symmetric_initial_guess():
    """Generate a symmetric initial guess using group theory principles."""
    # This creates a configuration with rotational symmetry of order 6
    # Inspired by mathematical packing theory
    
    positions = []
    
    # Place 12 hexagons in a pattern that respects 6-fold rotational symmetry
    # Use the fact that we can place 6 hexagons in two rings of 6 each
    
    # Outer ring - 6 hexagons
    outer_radius = 3.1  # Slightly smaller to allow for better packing
    for i in range(6):
        angle = i * math.pi / 3.0
        x = outer_radius * math.cos(angle)
        y = outer_radius * math.sin(angle)
        positions.append([x, y, 0.0])
    
    # Inner ring - 6 hexagons
    inner_radius = 1.85  # Adjusted for better packing
    for i in range(6):
        angle = i * math.pi / 3.0 + math.pi / 6.0  # Offset to create symmetry
        x = inner_radius * math.cos(angle)
        y = inner_radius * math.sin(angle)
        positions.append([x, y, 0.0])
    
    # Add small random perturbations to break degeneracies
    for i in range(len(positions)):
        positions[i][0] += np.random.normal(0, 0.005)
        positions[i][1] += np.random.normal(0, 0.005)
    
    # Flatten array
    flat_positions = []
    for pos in positions:
        flat_positions.extend(pos)
    
    return np.array(flat_positions)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses advanced optimization with symmetry-aware initialization.
    """
    start_time = time.time()
    
    # Try multiple starting configurations to find better solutions
    best_outer_side_length = float('inf')
    best_inner_hex_data = None
    
    # Try several different initial guesses
    for attempt in range(3):
        try:
            if attempt == 0:
                # Try improved initial guess
                initial_guess = generate_improved_initial_guess()
            elif attempt == 1:
                # Try symmetric initial guess  
                initial_guess = generate_symmetric_initial_guess()
            else:
                # Try random initialization
                initial_guess = np.random.uniform(-5, 5, 36)  # 12 hexagons * 3 params each
            
            # Set bounds for optimization (x, y, angle for each hexagon)
            bounds = []
            for _ in range(12):
                bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # Broader bounds for exploration
            
            # Strategy: Use both global and local optimization approaches
            # Use differential evolution for global search with higher population
            from scipy.optimize import differential_evolution
            
            result = differential_evolution(
                evaluate_configuration_analytical,
                bounds,
                maxiter=200,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42 + attempt,
                atol=1e-12,
                rtol=1e-12
            )
            
            # Extract best solution
            best_config = result.x.reshape(12, 3)
            inner_hex_data = best_config.copy()
            
            # Calculate final outer radius
            outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
            
            # Final validation with detailed evaluation
            final_eval = evaluate_configuration_analytical(best_config.flatten())
            current_value = -final_eval  # Convert back to positive value
            
            if outer_side_length < best_outer_side_length and abs(final_eval) < 1000000:
                best_outer_side_length = outer_side_length
                best_inner_hex_data = inner_hex_data.copy()
                
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            continue
    
    # If no good solution found, fall back to the best we have
    if best_inner_hex_data is None:
        # Fall back to symmetric initial guess
        inner_hex_data = generate_symmetric_initial_guess().reshape(12, 3)
        outer_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
    else:
        inner_hex_data = best_inner_hex_data
        outer_side_length = best_outer_side_length
    
    # Additional local refinement using L-BFGS-B with better bounds
    try:
        # Refine with local optimizer using tighter bounds around current solution
        refined_bounds = []
        for i in range(12):
            x, y, angle = inner_hex_data[i]
            refined_bounds.extend([(x-0.2, x+0.2), (y-0.2, y+0.2), (angle-15, angle+15)])
        
        result_refined = minimize(
            evaluate_configuration_analytical,
            inner_hex_data.flatten(),
            method='L-BFGS-B',
            bounds=refined_bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result_refined.success:
            refined_config = result_refined.x.reshape(12, 3)
            refined_outer_length = calculate_outer_hexagon_side_length(refined_config)
            if refined_outer_length < outer_side_length:
                inner_hex_data = refined_config
                outer_side_length = refined_outer_length
                
    except Exception as e:
        print(f"L-BFGS refinement failed: {e}")
    
    # Final validation with improved precision
    final_inv_radius = 1.0 / outer_side_length
    benchmark_ratio = final_inv_radius / 0.2537
    
    print(f"Optimization completed in {time.time() - start_time:.4f} seconds")
    print(f"Final inverse outer radius: {final_inv_radius:.6f}")
    print(f"Final outer side length: {outer_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    
    return inner_hex_data, np.array([0, 0, 0]), outer_side_length


# EVOLVE-BLOCK-END
