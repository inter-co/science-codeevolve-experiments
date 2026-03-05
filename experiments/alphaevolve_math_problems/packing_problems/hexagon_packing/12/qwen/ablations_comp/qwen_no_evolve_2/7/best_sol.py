# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
import time
from typing import Tuple, List
import math

def create_hexagon_vertices(center_x: float, center_y: float, side_length: float, rotation_deg: float) -> np.ndarray:
    """Create vertices of a regular hexagon given center, side length, and rotation."""
    rotation_rad = math.radians(rotation_deg)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def is_hexagon_contained(hex_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
    """Check if all vertices of inner hexagon are within outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def do_hexagons_overlap(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hexagon_radius(inner_hex_data: np.ndarray, outer_center=(0, 0)) -> float:
    """Calculate minimum radius needed to contain all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Calculate distance from center to furthest vertex of this hexagon
        hex_vertices = create_hexagon_vertices(center_x, center_y, 1.0, angle)
        for vertex in hex_vertices:
            distance = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_distance = max(max_distance, distance)
    
    # Add small buffer to account for potential floating point errors
    return max_distance * 1.01

def evaluate_packing(inner_hex_data: np.ndarray, outer_center=(0, 0)) -> Tuple[float, bool]:
    """
    Evaluate a packing configuration.
    Returns (negative_outer_radius, is_valid)
    """
    # Calculate required outer hexagon size
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center)
    
    # Create outer hexagon vertices
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], outer_radius, 0)
    
    # Check containment of all inner hexagons
    all_contained = True
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_vertices = create_hexagon_vertices(center_x, center_y, 1.0, angle)
        if not is_hexagon_contained(hex_vertices, outer_vertices):
            all_contained = False
            break
    
    if not all_contained:
        return -float('inf'), False
    
    # Check for overlaps
    no_overlaps = True
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            hex1_vertices = create_hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1.0, inner_hex_data[i][2])
            hex2_vertices = create_hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1.0, inner_hex_data[j][2])
            if do_hexagons_overlap(hex1_vertices, hex2_vertices):
                no_overlaps = False
                break
        if not no_overlaps:
            break
    
    if not no_overlaps:
        return -float('inf'), False
    
    # Return negative radius (since we want to maximize 1/r, which means minimize r)
    return -outer_radius, True

def generate_symmetric_initial_population(n_individuals: int) -> List[np.ndarray]:
    """Generate symmetric initial population for optimization."""
    populations = []
    
    # Base symmetric arrangement
    base_config = np.array([
        [0, 0, 0],      # center
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [1.732, 1.0, 0], # top right
        [-1.732, 1.0, 0], # top left
        [1.732, -1.0, 0], # bottom right
        [-1.732, -1.0, 0], # bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3.0, 0], # top far right
        [-1.732, 3.0, 0], # top far left
        [1.732, -3.0, 0], # bottom far right
    ])
    
    for _ in range(n_individuals):
        # Add some random perturbation while maintaining approximate symmetry
        perturbed = base_config.copy().astype(float)
        for i in range(len(perturbed)):
            # Small random movement
            perturbed[i][0] += np.random.normal(0, 0.1)
            perturbed[i][1] += np.random.normal(0, 0.1)
            # Keep angles close to 0 (no rotation for now)
            perturbed[i][2] = 0
        populations.append(perturbed)
    
    return populations

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a symmetry-aware evolutionary approach.
    """
    # Define bounds for optimization (x, y, angle) for each of 12 hexagons
    # We'll optimize positions and keep rotations fixed at 0 degrees for simplicity
    bounds = []
    for _ in range(12):
        # x, y positions (limited to reasonable range)
        bounds.extend([(-8, 8), (-8, 8), (0, 0)])  # Fixed angle = 0
    
    # Use differential evolution with custom objective function
    def objective(x):
        # Convert flat array back to 12 hexagons with (x, y, angle)
        hex_data = np.zeros((12, 3))
        for i in range(12):
            hex_data[i] = [x[3*i], x[3*i+1], x[3*i+2]]
        
        # Evaluate the packing
        neg_radius, valid = evaluate_packing(hex_data)
        
        # Return negative because we want to maximize 1/r (minimize r)
        if not valid:
            return 1000000  # Large penalty for invalid configurations
        return -neg_radius
    
    # Run optimization
    start_time = time.time()
    
    # Multi-start approach with different initial populations
    best_result = None
    best_value = float('inf')
    
    for _ in range(5):  # Try 5 different starting points
        # Generate random starting population
        popsize = 15
        individual_size = 36  # 12 * 3
        population = []
        for _ in range(popsize):
            individual = []
            for i in range(12):
                individual.extend([
                    np.random.uniform(-5, 5),  # x
                    np.random.uniform(-5, 5),  # y  
                    0                          # angle (fixed)
                ])
            population.append(individual)
        
        # Run differential evolution with this population
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                init=population,
                disp=False
            )
            
            if result.success:
                # Convert result back to hex data format
                hex_data = np.zeros((12, 3))
                for i in range(12):
                    hex_data[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
                
                # Re-evaluate to get final score
                neg_radius, valid = evaluate_packing(hex_data)
                if valid and neg_radius < best_value:
                    best_value = neg_radius
                    best_result = hex_data.copy()
                    
        except Exception:
            continue
    
    # If we didn't find a good solution, fall back to a better symmetric arrangement
    if best_result is None:
        # Try a known good symmetric arrangement
        best_result = np.array([
            [0, 0, 0],
            [0, 2.0, 0],
            [0, -2.0, 0],
            [1.732, 1.0, 0],
            [-1.732, 1.0, 0],
            [1.732, -1.0, 0],
            [-1.732, -1.0, 0],
            [3.464, 0, 0],
            [-3.464, 0, 0],
            [1.732, 3.0, 0],
            [-1.732, 3.0, 0],
            [1.732, -3.0, 0],
        ])
    
    # Final validation and calculation
    final_neg_radius, valid = evaluate_packing(best_result)
    
    # Calculate outer hexagon side length
    outer_radius = -final_neg_radius
    # For a regular hexagon, the side length equals the radius
    outer_hex_side_length = outer_radius
    
    # Outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    return best_result, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
