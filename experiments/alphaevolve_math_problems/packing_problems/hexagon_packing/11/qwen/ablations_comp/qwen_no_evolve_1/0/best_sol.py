# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time


def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(angle),
               center[1] + side_length * np.sin(angle)) for angle in angles]
    return Polygon(points)


def hexagon_vertices(center, side_length, rotation):
    """Get vertices of a regular hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return np.array([(center[0] + side_length * np.cos(angle),
                      center[1] + side_length * np.sin(angle)) for angle in angles])


def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly)


def compute_distance_matrix(hexagon_centers, hexagon_rotations, side_length=1):
    """Compute pairwise distances between hexagon centers."""
    n = len(hexagon_centers)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(hexagon_centers[i] - hexagon_centers[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    return dist_matrix


def compute_collision_penalty(hexagon_centers, hexagon_rotations, side_length=1):
    """Compute penalty for overlapping hexagons."""
    penalty = 0
    n = len(hexagon_centers)
    
    for i in range(n):
        for j in range(i+1, n):
            center_i = hexagon_centers[i]
            center_j = hexagon_centers[j]
            dist = np.linalg.norm(center_i - center_j)
            
            # Minimum distance between hexagons (2 units for unit hexagons)
            min_dist = 2.0
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return penalty


def evaluate_packing(hexagon_data, outer_hex_side_length):
    """Evaluate the packing quality."""
    # Extract data
    hexagon_centers = hexagon_data[:, :2]
    hexagon_rotations = hexagon_data[:, 2]
    
    # Create hexagon polygons
    inner_hexagons = []
    for i, (center, rotation) in enumerate(zip(hexagon_centers, hexagon_rotations)):
        hex_poly = create_regular_hexagon(center, 1, rotation)
        inner_hexagons.append(hex_poly)
    
    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_hex_side_length, 0)
    
    # Check containment and collisions
    total_penalty = 0
    for i, hex_poly in enumerate(inner_hexagons):
        if not check_containment(hex_poly, outer_hex):
            total_penalty += 1000  # Large penalty for containment violation
            
    # Collision penalty
    collision_penalty = compute_collision_penalty(hexagon_centers, hexagon_rotations)
    total_penalty += collision_penalty
    
    # Calculate how much we're under the optimal (we want to maximize 1/outer_hex_side_length)
    # So we minimize outer_hex_side_length
    return total_penalty + outer_hex_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses force-directed optimization approach.
    """
    # Initial configuration - more strategic placement than simple grid
    initial_hexagon_data = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom
        [1.732, 1, 0],  # top-right
        [-1.732, 1, 0], # top-left
        [1.732, -1, 0], # bottom-right
        [-1.732, -1, 0],# bottom-left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 2, 0],  # top-right corner
        [-1.732, 2, 0], # top-left corner
    ])
    
    # Add some randomness to initial positions
    np.random.seed(42)
    initial_hexagon_data[:, :2] += np.random.normal(0, 0.1, (11, 2))
    
    # Optimize using scipy
    def objective(params):
        # params: [x1, y1, rot1, ..., x11, y11, rot11, outer_radius]
        n = 11
        hexagon_data = params[:3*n].reshape(n, 3)
        outer_radius = params[3*n]
        
        # Ensure minimum outer radius
        if outer_radius < 2:
            return 1e6
            
        return evaluate_packing(hexagon_data, outer_radius)
    
    # Flatten initial parameters
    initial_params = np.concatenate([
        initial_hexagon_data.flatten(),
        [6.0]  # initial outer radius
    ])
    
    # Bounds for positions (-10, 10) and rotations (0, 360)
    bounds = []
    for i in range(11):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, rotation
    bounds.append((2.0, 20.0))  # outer radius
    
    # Optimization with constraints
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        # Extract results
        final_params = result.x
        n = 11
        final_hexagon_data = final_params[:3*n].reshape(n, 3)
        outer_hex_side_length = final_params[3*n]
        
        # Refine final result to ensure tight packing
        # Use a local refinement step
        def refine_objective(params):
            hexagon_data = params[:3*n].reshape(n, 3)
            outer_radius = params[3*n]
            return evaluate_packing(hexagon_data, outer_radius)
        
        # Final refinement with smaller bounds
        refined_result = minimize(
            refine_objective,
            final_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        final_params = refined_result.x
        final_hexagon_data = final_params[:3*n].reshape(n, 3)
        outer_hex_side_length = final_params[3*n]
        
    except Exception as e:
        # Fallback to original simple configuration if optimization fails
        print(f"Optimization failed: {e}")
        outer_hex_side_length = 4.0
        final_hexagon_data = initial_hexagon_data.copy()
    
    # Ensure we have valid output
    if outer_hex_side_length <= 0:
        outer_hex_side_length = 4.0
    
    # Set outer hexagon at center
    outer_hex_data = np.array([0, 0, 0])
    
    return final_hexagon_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
