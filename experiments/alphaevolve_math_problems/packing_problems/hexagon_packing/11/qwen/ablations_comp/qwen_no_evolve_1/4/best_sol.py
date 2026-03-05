# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math

def create_hexagon(center, side_length, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angles = [math.pi/3 * i + math.radians(rotation) for i in range(6)]
    points = [(center[0] + side_length * math.cos(angle), 
               center[1] + side_length * math.sin(angle)) for angle in angles]
    return Polygon(points)

def get_hexagon_vertices(center, side_length, rotation=0):
    """Get vertices of a hexagon"""
    angles = [math.pi/3 * i + math.radians(rotation) for i in range(6)]
    return [(center[0] + side_length * math.cos(angle), 
             center[1] + side_length * math.sin(angle)) for angle in angles]

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained in outer hexagon"""
    return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def compute_total_force(positions, rotations, side_length, outer_side_length):
    """Compute total forces on all hexagons"""
    # Create hexagons
    hexagons = []
    for pos, rot in zip(positions, rotations):
        hexagons.append(create_hexagon(pos, side_length, rot))
    
    # Create outer hexagon
    outer_hex = create_hexagon((0, 0), outer_side_length)
    
    # Initialize forces
    forces = np.zeros_like(positions)
    
    # Repulsion forces between overlapping hexagons
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if check_overlap(hexagons[i], hexagons[j]):
                # Compute repulsion force
                diff = np.array(hexagons[i].centroid.coords[0]) - np.array(hexagons[j].centroid.coords[0])
                distance = np.linalg.norm(diff)
                if distance > 0:
                    force_magnitude = 1.0 / (distance * distance + 0.1)
                    forces[i] += force_magnitude * diff / distance
                    forces[j] -= force_magnitude * diff / distance
    
    # Attraction forces to keep hexagons inside outer hexagon
    for i in range(len(hexagons)):
        centroid = np.array(hexagons[i].centroid.coords[0])
        if not check_containment(hexagons[i], outer_hex):
            # Push back towards center
            force = -0.1 * centroid
            forces[i] += force
    
    return forces

def objective_function(params):
    """Objective function to minimize (negative of inverse outer hex side length)"""
    # Parse parameters
    positions = params[:22].reshape(-1, 2)
    rotations = params[22:33]
    outer_side_length = params[33]
    
    # Create hexagons
    hexagons = []
    for pos, rot in zip(positions, rotations):
        hexagons.append(create_hexagon(pos, 1.0, rot))
    
    # Create outer hexagon
    outer_hex = create_hexagon((0, 0), outer_side_length)
    
    # Check constraints
    # Overlap constraint
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if check_overlap(hexagons[i], hexagons[j]):
                return 1e10  # Penalty for overlap
    
    # Containment constraint  
    for hexagon in hexagons:
        if not check_containment(hexagon, outer_hex):
            return 1e10  # Penalty for containment violation
    
    # Return negative of inverse outer side length (we want to maximize 1/R)
    return -1.0 / outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-based optimization approach.
    """
    # Initial configuration - arrange in a more efficient pattern
    initial_positions = np.array([
        [0, 0],      # center
        [0, 2],      # top
        [0, -2],     # bottom  
        [1.732, 1],  # top right
        [-1.732, 1], # top left
        [1.732, -1], # bottom right
        [-1.732, -1], # bottom left
        [3.464, 0],  # far right
        [-3.464, 0], # far left
        [1.732, 3],  # top far right
        [-1.732, 3]  # top far left
    ])
    
    # Initial rotations (all 0 for simplicity)
    initial_rotations = np.zeros(11)
    
    # Initial outer hexagon size
    initial_outer_size = 6.0
    
    # Combine all parameters
    initial_params = np.concatenate([
        initial_positions.flatten(),
        initial_rotations,
        [initial_outer_size]
    ])
    
    # Define bounds for optimization
    bounds = []
    # Positions bounds
    for _ in range(22):
        bounds.extend([(-10, 10)])  # Reasonable bounds for positions
    # Rotations bounds (0-360 degrees)
    for _ in range(11):
        bounds.extend([(0, 360)])
    # Outer size bound
    bounds.append((2, 10))  # Reasonable bounds for outer size
    
    # Optimization settings
    options = {'maxiter': 1000, 'disp': False}
    
    # Optimize using scipy minimize
    result = minimize(
        objective_function, 
        initial_params, 
        method='L-BFGS-B', 
        bounds=bounds, 
        options=options,
        tol=1e-6
    )
    
    # Extract results
    final_positions = result.x[:22].reshape(-1, 2)
    final_rotations = result.x[22:33]
    outer_side_length = result.x[33]
    
    # Convert to desired output format
    inner_hex_data = np.column_stack([final_positions, final_rotations])
    outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
