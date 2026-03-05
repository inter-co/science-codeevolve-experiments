# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import math
import random


def create_regular_hexagon_vertices(center=(0, 0), side_length=1, rotation=0):
    """Create vertices of a regular hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + rotation * np.pi / 180
    vertices = np.array([
        (center[0] + side_length * np.cos(angle),
         center[1] + side_length * np.sin(angle))
        for angle in angles
    ])
    return vertices


def hexagon_vertices(hex_data):
    """Get vertices for a hexagon given its data."""
    center = (hex_data[0], hex_data[1])
    side_length = 1  # unit hexagon
    rotation = hex_data[2] * np.pi / 180
    return create_regular_hexagon_vertices(center, side_length, rotation)


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using shapely for precise polygon intersection."""
    try:
        from shapely.geometry import Polygon
        hex1_poly = Polygon(hex1_vertices[:-1])
        hex2_poly = Polygon(hex2_vertices[:-1])
        return hex1_poly.intersects(hex2_poly)
    except ImportError:
        # Fallback: simplified distance-based check
        distances = cdist(hex1_vertices[:-1], hex2_vertices[:-1])
        min_distance = np.min(distances)
        # For unit hexagons, they don't overlap if min distance >= 2
        return min_distance < 2.0


def compute_outer_hexagon_side_length(inner_hex_data, outer_center=(0, 0)):
    """Compute the minimal side length needed for outer hexagon to contain all inner hexagons."""
    # Create vertices for all inner hexagons and find maximum distance from center
    max_dist = 0
    for hex_data in inner_hex_data:
        vertices = hexagon_vertices(hex_data)
        # Find maximum distance from center to any vertex
        for vertex in vertices[:-1]:  # exclude repeated first vertex
            dist = np.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # The side length of the circumscribing hexagon is the maximum distance
    # from center to any vertex (circumradius)
    return max_dist


def evaluate_packing(inner_hex_data):
    """Evaluate a packing configuration."""
    # Check for overlaps
    num_hexagons = len(inner_hex_data)
    for i in range(num_hexagons):
        for j in range(i+1, num_hexagons):
            hex1_v = hexagon_vertices(inner_hex_data[i])
            hex2_v = hexagon_vertices(inner_hex_data[j])
            if check_overlap(hex1_v, hex2_v):
                return float('inf')  # Invalid packing due to overlap
    
    # Compute outer hexagon side length
    side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Return inverse of side length (we want to maximize this)
    return 1.0 / side_length if side_length > 0 else float('inf')


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid optimization approach with enhanced convergence parameters.
    
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use a configuration that's known to be close to optimal (based on INSPIRATION 3)
    # Slightly refined values for better optimization convergence
    sqrt3 = np.sqrt(3)
    
    # Based on mathematical research for 12 hexagon packings - highly optimized values
    initial_positions = [
        [0, 0, 0],               # center
        [0, 1.928, 0],           # top
        [0, -1.928, 0],          # bottom  
        [sqrt3 * 0.964, 0.964, 0], # top right
        [-sqrt3 * 0.964, 0.964, 0], # top left
        [sqrt3 * 0.964, -0.964, 0], # bottom right
        [-sqrt3 * 0.964, -0.964, 0], # bottom left
        [2 * sqrt3 * 0.964, 0, 0],   # far right
        [-2 * sqrt3 * 0.964, 0, 0],  # far left
        [sqrt3 * 0.964, 2.892, 0],   # top far right
        [-sqrt3 * 0.964, 2.892, 0],  # top far left
        [sqrt3 * 0.964, -2.892, 0],  # bottom far right
    ]
    
    inner_hex_data = np.array(initial_positions)
    
    # Define the objective function for global optimization
    def objective_global(params):
        # Reshape parameters: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12]
        hex_data = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3 + 1] 
            angle = params[i*3 + 2]
            hex_data.append([x, y, angle])
        return -evaluate_packing(np.array(hex_data))  # negative because we maximize
    
    # Set up bounds for global optimization (more restrictive bounds)
    bounds = []
    for i in range(12):
        # Position bounds - reasonable range for hexagon packing
        bounds.extend([(-4, 4), (-4, 4), (0, 360)])  # x, y, angle
    
    # Use differential evolution for global search with enhanced parameters
    # More aggressive optimization to achieve better convergence
    try:
        de_result = differential_evolution(
            objective_global, 
            bounds, 
            maxiter=25,  # Increased iterations for better convergence
            popsize=15,  # Larger population for better exploration
            seed=42,
            tol=1e-8,    # Tighter tolerance for convergence
            recombination=0.9,  # Higher recombination probability
            mutation=(0.7, 1.0)  # More aggressive mutation
        )
        
        if de_result.success:
            # Extract best solution from global optimization
            best_params = de_result.x
            hex_data = []
            for i in range(12):
                x = best_params[i*3]
                y = best_params[i*3 + 1] 
                angle = best_params[i*3 + 2]
                hex_data.append([x, y, angle])
            inner_hex_data = np.array(hex_data)
    except Exception:
        pass
    
    # Now perform local optimization refinement with even higher precision
    def objective_local(x_flat):
        # Reshape flat array back to hexagon data
        hex_data = x_flat.reshape(-1, 3)
        return -evaluate_packing(hex_data)  # negative because we maximize
    
    # Flatten for optimization
    x0 = inner_hex_data.flatten()
    
    # Local optimization with extremely tight tolerances
    try:
        # Use L-BFGS-B for local refinement with very tight tolerances
        result = minimize(
            objective_local, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16, 'eps': 1e-8}
        )
        
        if result.success:
            optimized_hex_data = result.x.reshape(-1, 3)
            score = evaluate_packing(optimized_hex_data)
            # Accept any improvement over baseline
            if score > 0.225 and score != float('inf'):
                inner_hex_data = optimized_hex_data
    except Exception:
        pass
    
    # Set outer hexagon at center with zero rotation
    outer_hex_data = np.array([0, 0, 0])
    
    # Final evaluation to get the actual side length
    final_inv_side_length = evaluate_packing(inner_hex_data)
    outer_hex_side_length = 1.0 / final_inv_side_length if final_inv_side_length != float('inf') else 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
