# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import math
from typing import Tuple, List
import time

def create_regular_hexagon(center_x: float, center_y: float, side_length: float, rotation_deg: float = 0) -> Polygon:
    """Create a regular hexagon as a Shapely polygon."""
    rotation_rad = math.radians(rotation_deg)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return Polygon(vertices)

def check_hexagon_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if a hexagon is fully contained within the outer hexagon."""
    return outer_hex.contains(hexagon) or outer_hex.touches(hexagon)

def check_hexagon_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def evaluate_solution(solution: np.ndarray, outer_radius: float) -> Tuple[float, bool]:
    """
    Evaluate a solution: returns (penalty, is_valid)
    penalty = 0 if valid, positive if invalid
    """
    # Extract parameters
    # First 33 values: 11 hexagons * (x, y, angle)
    # Last value: outer hexagon radius
    inner_params = solution[:-1].reshape(-1, 3)
    outer_radius = solution[-1]
    
    # Create outer hexagon
    outer_hex = create_regular_hexagon(0, 0, outer_radius)
    
    # Check if all inner hexagons are valid
    total_penalty = 0
    
    # Check containment and overlaps for all inner hexagons
    inner_hexagons = []
    for i, (x, y, angle) in enumerate(inner_params):
        hexagon = create_regular_hexagon(x, y, 1, angle)
        inner_hexagons.append(hexagon)
        
        # Check containment
        if not check_hexagon_containment(hexagon, outer_hex):
            total_penalty += 1000  # Large penalty for containment violation
            
        # Check overlaps with other hexagons
        for j in range(i):
            if check_hexagon_overlap(hexagon, inner_hexagons[j]):
                total_penalty += 1000  # Large penalty for overlap
    
    # If any violations, return high penalty
    if total_penalty > 0:
        return total_penalty, False
    
    # Return negative of inverse radius (since we want to maximize 1/R, minimize R)
    return -1.0 / outer_radius, True

def generate_initial_population(n_individuals: int, n_hexagons: int = 11) -> np.ndarray:
    """Generate initial population for evolutionary algorithm."""
    population = []
    
    # Generate diverse starting solutions
    for _ in range(n_individuals):
        # Random positions and rotations for inner hexagons
        individual = []
        for _ in range(n_hexagons):
            # Random position within reasonable bounds
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            angle = np.random.uniform(0, 360)
            individual.extend([x, y, angle])
        
        # Outer hexagon radius - start with a reasonable estimate
        outer_radius = np.random.uniform(3, 8)
        individual.append(outer_radius)
        population.append(individual)
    
    return np.array(population)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary algorithm with geometric constraints.
    """
    # Parameters
    n_hexagons = 11
    n_vars = n_hexagons * 3 + 1  # 11 hexagons * 3 params + 1 outer radius
    max_outer_radius = 10.0
    min_outer_radius = 2.0
    
    # Bounds for variables: [x1, y1, angle1, ..., x11, y11, angle11, outer_radius]
    bounds = []
    for i in range(n_hexagons):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])  # x, y, angle for each hexagon
    bounds.append((min_outer_radius, max_outer_radius))  # outer radius
    
    # Use differential evolution for global optimization
    def objective(x):
        penalty, is_valid = evaluate_solution(x, x[-1])
        if not is_valid:
            return penalty + 10000  # Add large penalty for invalid solutions
        return penalty
    
    # Run optimization
    start_time = time.time()
    
    # Use differential evolution with bounds
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    end_time = time.time()
    
    # Extract best solution
    best_solution = result.x
    inner_params = best_solution[:-1].reshape(-1, 3)
    outer_radius = best_solution[-1]
    
    # Convert to required format
    inner_hex_data = inner_params.copy()  # Already in (x, y, angle) format
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
