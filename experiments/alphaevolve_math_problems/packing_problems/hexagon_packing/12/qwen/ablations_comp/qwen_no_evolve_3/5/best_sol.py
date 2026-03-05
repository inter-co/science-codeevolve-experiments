# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from typing import Tuple, List

def create_regular_hexagon(center: Tuple[float, float], radius: float, rotation: float = 0) -> Polygon:
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) 
              for angle in angles]
    return Polygon(points)

def check_hexagon_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex.contains(hexagon) or outer_hex.touches(hexagon)

def check_hexagon_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2) and not hex1.touches(hex2)

def evaluate_configuration(inner_positions: np.ndarray, inner_rotations: np.ndarray, 
                          outer_center: Tuple[float, float], outer_radius: float) -> Tuple[float, bool]:
    """Evaluate configuration: returns (1/outer_radius, is_valid)."""
    # Create outer hexagon
    outer_hex = create_regular_hexagon(outer_center, outer_radius)
    
    # Create inner hexagons
    inner_hexagons = []
    for i in range(12):
        pos = tuple(inner_positions[i])
        rot = inner_rotations[i]
        hexagon = create_regular_hexagon(pos, 1.0, rot)
        inner_hexagons.append(hexagon)
    
    # Check containment
    for hexagon in inner_hexagons:
        if not check_hexagon_containment(hexagon, outer_hex):
            return 0.0, False
    
    # Check overlaps
    for i in range(12):
        for j in range(i+1, 12):
            if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
                return 0.0, False
    
    return 1.0 / outer_radius, True

def generate_symmetric_initial_population() -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate initial configuration using symmetric arrangement."""
    # Start with a symmetric pattern inspired by known good packings
    positions = np.array([
        [0, 0, 0],      # center
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom  
        [1.732, 1.0, 0], # top-right
        [-1.732, 1.0, 0], # top-left
        [1.732, -1.0, 0], # bottom-right
        [-1.732, -1.0, 0], # bottom-left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3.0, 0], # top far-right
        [-1.732, 3.0, 0], # top far-left
        [1.732, -3.0, 0], # bottom far-right
    ])
    
    rotations = np.zeros(12)
    outer_radius = 4.0
    
    return positions, rotations, outer_radius

def optimize_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Use evolutionary approach with symmetry awareness."""
    # Initialize population with symmetric configurations
    best_score = 0.0
    best_positions = None
    best_rotations = None
    best_radius = float('inf')
    
    # Use a simple evolutionary approach with mutation
    for generation in range(1000):  # Limit generations for time constraint
        # Generate candidate solutions
        candidates = []
        
        # Generate some random variations around good starting point
        for _ in range(20):
            # Start with symmetric configuration
            positions, rotations, radius = generate_symmetric_initial_population()
            
            # Add small random perturbations
            positions += np.random.normal(0, 0.2, positions.shape)
            rotations += np.random.normal(0, 15, rotations.shape)
            
            # Randomly vary the outer radius within reasonable bounds
            radius = max(2.0, radius + np.random.normal(0, 0.5))
            
            score, valid = evaluate_configuration(positions[:, :2], rotations, (0, 0), radius)
            
            if valid and score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_rotations = rotations.copy()
                best_radius = radius
                
            candidates.append((positions, rotations, radius, score, valid))
        
        # Early stopping if we're getting close to target
        if best_score > 0.2535:  # Close to target
            break
            
    # Refine the best solution with local search
    if best_positions is not None:
        # Try to improve further with gradient-like steps
        for _ in range(50):
            # Small adjustments to positions and rotations
            adjusted_positions = best_positions.copy()
            adjusted_rotations = best_rotations.copy()
            
            # Perturb positions slightly
            noise = np.random.normal(0, 0.05, adjusted_positions.shape)
            adjusted_positions[:, :2] += noise[:, :2]
            
            # Perturb rotations
            adjusted_rotations += np.random.normal(0, 5, adjusted_rotations.shape)
            
            # Try to reduce outer radius slightly
            test_radius = max(2.0, best_radius - np.random.exponential(0.05))
            
            score, valid = evaluate_configuration(adjusted_positions[:, :2], adjusted_rotations, (0, 0), test_radius)
            
            if valid and score > best_score:
                best_score = score
                best_positions = adjusted_positions.copy()
                best_rotations = adjusted_rotations.copy()
                best_radius = test_radius
    
    return best_positions, best_rotations, best_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Get optimized configuration
    inner_positions, inner_rotations, outer_radius = optimize_hexagon_packing()
    
    # Convert to required format
    if inner_positions is None:
        # Fallback to basic configuration if optimization fails
        inner_positions = np.array([
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
        inner_rotations = np.zeros(12)
        outer_radius = 4.0
    
    # Create final result array
    inner_hex_data = np.column_stack([inner_positions[:, :2], inner_rotations])
    
    # Outer hexagon centered at origin with calculated radius
    outer_hex_data = np.array([0, 0, 0])
    
    # Convert to side length (for regular hexagon, radius = side length)
    outer_hex_side_length = outer_radius
    
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
