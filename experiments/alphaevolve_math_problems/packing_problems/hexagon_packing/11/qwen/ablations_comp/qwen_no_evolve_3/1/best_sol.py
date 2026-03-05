# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time


def create_regular_hexagon(center=(0, 0), radius=1, rotation=0):
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius * np.cos(angles),
                             center[1] + radius * np.sin(angles)])
    return points[:-1]  # Remove last point to close the polygon


def hexagon_vertices(center, radius=1, rotation=0):
    """Get vertices of a hexagon."""
    return create_regular_hexagon(center, radius, rotation)


def check_containment(hex_points, outer_hex_points):
    """Check if hexagon is fully contained within outer hexagon."""
    inner_polygon = Polygon(hex_points)
    outer_polygon = Polygon(outer_hex_points)
    return outer_polygon.contains(inner_polygon)


def hexagon_distance(h1_center, h1_rotation, h2_center, h2_rotation):
    """Calculate minimum distance between two hexagons."""
    h1_points = hexagon_vertices(h1_center, 1, h1_rotation)
    h2_points = hexagon_vertices(h2_center, 1, h2_rotation)
    
    # Use distance between centroids as proxy for minimum distance
    dist = np.linalg.norm(np.array(h1_center) - np.array(h2_center))
    return max(0, dist - 2)  # Subtract 2 (diameter of unit hexagon)


def compute_total_energy(positions_and_angles, outer_radius):
    """Compute total energy for force-based optimization."""
    n = len(positions_and_angles) // 3
    positions = positions_and_angles[:n*2].reshape(n, 2)
    angles = positions_and_angles[n*2:]
    
    energy = 0.0
    
    # Repulsion energy between hexagons (inverse distance)
    for i in range(n):
        for j in range(i+1, n):
            pos_i = positions[i]
            pos_j = positions[j]
            dist = np.linalg.norm(pos_i - pos_j)
            if dist < 2:  # Overlapping
                energy += 1000 / (2 - dist)  # Strong repulsion when overlapping
            elif dist < 4:  # Close but not overlapping
                energy += 10 / (4 - dist)  # Moderate repulsion
    
    # Boundary energy - penalize if any hexagon extends beyond outer hexagon
    outer_hex_points = hexagon_vertices((0, 0), outer_radius, 0)
    outer_polygon = Polygon(outer_hex_points)
    
    for i in range(n):
        hex_points = hexagon_vertices(positions[i], 1, angles[i])
        hex_polygon = Polygon(hex_points)
        
        if not outer_polygon.contains(hex_polygon):
            # Calculate how much it's outside
            try:
                intersection = outer_polygon.intersection(hex_polygon)
                if intersection.area < hex_polygon.area * 0.5:
                    energy += 10000  # Severe penalty for major overlap
                else:
                    # Partial containment penalty
                    energy += 1000 / (hex_polygon.area - intersection.area + 1e-6)
            except:
                energy += 10000
    
    return energy


def optimize_hexagon_positions(initial_positions, initial_angles, outer_radius):
    """Optimize hexagon positions using scipy minimize."""
    # Flatten parameters
    initial_params = np.concatenate([initial_positions.flatten(), initial_angles])
    
    # Define bounds
    bounds = []
    # Position bounds
    for i in range(len(initial_positions)):
        bounds.extend([(-outer_radius*2, outer_radius*2), (-outer_radius*2, outer_radius*2)])
    # Angle bounds
    for i in range(len(initial_angles)):
        bounds.extend([(-180, 180)])
    
    # Optimization callback to track progress
    def callback(x):
        pass  # Could add logging here
    
    # Optimize
    result = minimize(
        lambda x: compute_total_energy(x, outer_radius),
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-6},
        callback=callback
    )
    
    return result.x


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a force-based optimization approach to find better arrangements than simple grid layouts.
    """
    # Start with a more informed initial guess based on known good patterns
    # This is inspired by the "kissing number" arrangement but optimized
    initial_positions = np.array([
        [0, 0],      # center
        [0, 2],      # top
        [0, -2],     # bottom  
        [1.732, 1],  # top right
        [-1.732, 1], # top left
        [1.732, -1], # bottom right
        [-1.732, -1],# bottom left
        [3.464, 0],  # far right
        [-3.464, 0], # far left
        [1.732, 3],  # top far right
        [-1.732, 3], # top far left
    ])
    
    initial_angles = np.zeros(11)  # All horizontal initially
    
    # Estimate initial outer radius
    max_dist_from_center = np.max(np.linalg.norm(initial_positions, axis=1))
    outer_radius = max_dist_from_center + 1.5  # Add some buffer
    
    # Run optimization
    optimized_params = optimize_hexagon_positions(initial_positions, initial_angles, outer_radius)
    
    # Extract results
    n = 11
    positions = optimized_params[:n*2].reshape(n, 2)
    angles = optimized_params[n*2:]
    
    # Create final data structure
    inner_hex_data = np.column_stack([positions, angles])
    
    # Compute final outer hexagon size
    max_dist = 0
    for i in range(n):
        pos = positions[i]
        # Get hexagon vertices and find maximum distance from center
        hex_points = hexagon_vertices(pos, 1, angles[i])
        for vertex in hex_points:
            dist = np.linalg.norm(vertex)
            max_dist = max(max_dist, dist)
    
    # Add buffer for safety and round up to reasonable value
    final_outer_radius = max_dist + 0.5
    
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, final_outer_radius


# EVOLVE-BLOCK-END
