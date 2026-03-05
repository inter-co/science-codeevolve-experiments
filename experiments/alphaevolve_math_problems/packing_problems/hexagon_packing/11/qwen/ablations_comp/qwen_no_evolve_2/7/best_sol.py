# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import time
from typing import Tuple, List
import math

# Constants
UNIT_HEX_RADIUS = 1.0  # Circumradius of unit hexagon
UNIT_HEX_APOGEE = math.sqrt(3)/2  # Apothem of unit hexagon
UNIT_HEX_SIDE = 1.0  # Side length of unit hexagon

def create_unit_hexagon(center=(0,0), rotation=0) -> Polygon:
    """Create a unit regular hexagon centered at center with given rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + UNIT_HEX_RADIUS * np.cos(angle),
               center[1] + UNIT_HEX_RADIUS * np.sin(angle)) for angle in angles]
    return Polygon(points)

def check_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer_hex."""
    return outer_hex.contains(hexagon)

def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2) and not hex1.touches(hex2)

def calculate_outer_hex_side_length(inner_hex_data: np.ndarray, outer_center=(0,0), outer_rotation=0) -> float:
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    max_dist = 0
    
    # Create the outer hexagon (for checking containment)
    outer_hex = create_unit_hexagon(outer_center, outer_rotation)
    
    # Calculate maximum distance from center to any vertex of inner hexagons
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        inner_hex = create_unit_hexagon(center, rotation)
        
        # Get all vertices of the inner hexagon
        vertices = list(inner_hex.exterior.coords)
        for vertex in vertices[:-1]:  # Exclude last point which is duplicate
            dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Convert from circumradius to side length
    # For a regular hexagon, side length = circumradius
    # But we need to account for the fact that our hexagons are oriented
    # We want the outer hexagon to have radius that can contain all vertices
    # The minimum side length of outer hexagon = max_dist / (sqrt(3)/2)
    # But since outer hexagon has circumradius = side length, we need:
    # side_length >= max_dist / (sqrt(3)/2) = max_dist * 2/sqrt(3)
    return max_dist * 2.0 / math.sqrt(3)

def evaluate_configuration(inner_hex_data: np.ndarray, outer_center=(0,0), outer_rotation=0) -> Tuple[float, bool]:
    """Evaluate a configuration: returns (inverse_side_length, is_valid)."""
    # Create outer hexagon
    outer_hex = create_unit_hexagon(outer_center, outer_rotation)
    
    # Check containment and overlap
    valid = True
    total_overlaps = 0
    
    # Check each inner hexagon for containment
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        inner_hex = create_unit_hexagon(center, rotation)
        
        if not check_containment(inner_hex, outer_hex):
            valid = False
            break
            
        # Check for overlaps with other hexagons
        for j in range(i+1, len(inner_hex_data)):
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            inner_hex2 = create_unit_hexagon(center2, rotation2)
            
            if check_overlap(inner_hex, inner_hex2):
                valid = False
                total_overlaps += 1
                break
    
    if not valid:
        return 0.0, False
    
    # Calculate inverse side length
    side_length = calculate_outer_hex_side_length(inner_hex_data, outer_center, outer_rotation)
    return 1.0 / side_length, True

def generate_initial_population(pop_size: int) -> List[np.ndarray]:
    """Generate initial population of configurations."""
    population = []
    for _ in range(pop_size):
        # Random positions and rotations for 11 hexagons
        hex_data = np.random.rand(11, 3) * 10  # Random positions in range [0,10)  
        hex_data[:, 2] = np.random.rand(11) * 360  # Random rotations [0,360)
        population.append(hex_data)
    return population

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual configuration."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position
            mutated[i, 0] += (random.random() - 0.5) * 2.0
            mutated[i, 1] += (random.random() - 0.5) * 2.0
            # Mutate rotation
            mutated[i, 2] += (random.random() - 0.5) * 60  # ±30 degrees
            mutated[i, 2] %= 360
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parents."""
    child = parent1.copy()
    # Single-point crossover for positions and rotations
    crossover_point = random.randint(1, len(child)-1)
    child[crossover_point:, :] = parent2[crossover_point:, :]
    return child

def evolve_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Evolve an optimal hexagon packing using genetic algorithm."""
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    
    # Generate initial population
    population = generate_initial_population(pop_size)
    
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness, valid = evaluate_configuration(individual)
            if valid:
                fitness_scores.append(fitness)
            else:
                fitness_scores.append(0.0)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection (tournament selection)
        selected = []
        for _ in range(pop_size):
            tournament_size = 3
            tournament_indices = random.sample(range(pop_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        # Crossover and mutation
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i+1) % pop_size]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.append(mutate_individual(child1, mutation_rate))
            new_population.append(mutate_individual(child2, mutation_rate))
        
        population = new_population
    
    # Final validation and refinement
    final_fitness, valid = evaluate_configuration(best_individual)
    if not valid:
        # Fall back to a good heuristic arrangement
        return heuristic_hexagon_packing()
    
    # Return best configuration found
    outer_center = (0, 0)
    outer_rotation = 0
    outer_side_length = 1.0 / final_fitness
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_rotation])
    
    return best_individual, outer_hex_data, outer_side_length

def heuristic_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Return a better heuristic arrangement than the baseline."""
    # More sophisticated arrangement based on hexagonal lattice
    # Center hexagon surrounded by 6 others in a hexagonal pattern
    # Plus 4 additional ones to fill gaps
    
    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 2.0, 0.0],           # top
        [1.732, 1.0, 0.0],         # top-right
        [1.732, -1.0, 0.0],        # bottom-right
        [0.0, -2.0, 0.0],          # bottom
        [-1.732, -1.0, 0.0],       # bottom-left
        [-1.732, 1.0, 0.0],        # top-left
        [3.464, 2.0, 0.0],         # far top-right
        [-3.464, 2.0, 0.0],        # far top-left
        [3.464, -2.0, 0.0],        # far bottom-right
        [-3.464, -2.0, 0.0],       # far bottom-left
    ])
    
    # Calculate the minimum outer hexagon size
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to improve upon baseline arrangements.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Try evolutionary optimization first
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = evolve_hexagon_packing()
        eval_time = time.time() - start_time
        
        # Validate the result
        inv_side_length, valid = evaluate_configuration(inner_hex_data)
        if not valid:
            # Fallback to heuristic if evolution failed
            inner_hex_data, outer_hex_data, outer_hex_side_length = heuristic_hexagon_packing()
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to heuristic arrangement if anything goes wrong
        inner_hex_data, outer_hex_data, outer_hex_side_length = heuristic_hexagon_packing()
        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
