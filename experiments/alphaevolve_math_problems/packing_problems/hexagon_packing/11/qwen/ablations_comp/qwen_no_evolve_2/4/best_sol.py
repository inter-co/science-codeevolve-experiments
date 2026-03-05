# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from typing import Tuple, List
import time

# Constants
UNIT_HEX_RADIUS = 1.0  # Distance from center to corner for unit hexagon
UNIT_HEX_WIDTH = 2.0  # Width of unit hexagon
UNIT_HEX_HEIGHT = np.sqrt(3.0)  # Height of unit hexagon
MAX_EVAL_TIME = 60.0  # Maximum evaluation time in seconds

def create_unit_hexagon(center_x: float, center_y: float, angle_deg: float) -> Polygon:
    """Create a unit regular hexagon centered at (center_x, center_y) rotated by angle_deg degrees."""
    # Vertices of a unit hexagon centered at origin
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = [(np.cos(angle), np.sin(angle)) for angle in angles]
    
    # Apply rotation and translation
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotated_vertices = []
    for x, y in vertices:
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        rotated_vertices.append((new_x + center_x, new_y + center_y))
    
    return Polygon(rotated_vertices)

def create_outer_hexagon(side_length: float, center_x: float = 0.0, center_y: float = 0.0) -> Polygon:
    """Create a regular hexagon with given side length centered at (center_x, center_y)."""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = []
    for angle in angles:
        x = side_length * np.cos(angle) + center_x
        y = side_length * np.sin(angle) + center_y
        vertices.append((x, y))
    return Polygon(vertices)

def check_hexagon_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer_hex."""
    # Check if all vertices of inner hexagon are within outer hexagon
    for point in hexagon.exterior.coords:
        if not outer_hex.contains(Point(point)):
            return False
    return True

def check_hexagon_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def evaluate_packing(inner_hex_data: np.ndarray, outer_side_length: float) -> Tuple[float, bool, bool]:
    """
    Evaluate a hexagon packing configuration.
    Returns:
        inv_side_length: 1/outer_side_length (higher is better)
        valid_containment: whether all hexagons are contained
        valid_overlap: whether there are no overlaps
    """
    try:
        # Create outer hexagon
        outer_hex = create_outer_hexagon(outer_side_length)
        
        # Create inner hexagons
        inner_hexagons = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle_deg = inner_hex_data[i]
            hexagon = create_unit_hexagon(center_x, center_y, angle_deg)
            inner_hexagons.append(hexagon)
        
        # Check containment
        valid_containment = all(check_hexagon_containment(h, outer_hex) for h in inner_hexagons)
        
        # Check overlaps
        valid_overlap = True
        for i in range(len(inner_hexagons)):
            for j in range(i+1, len(inner_hexagons)):
                if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
                    valid_overlap = False
                    break
            if not valid_overlap:
                break
        
        # Calculate objective
        inv_side_length = 1.0 / outer_side_length if valid_containment and valid_overlap else 0.0
        
        return inv_side_length, valid_containment, valid_overlap
    
    except Exception as e:
        return 0.0, False, False

def generate_initial_population(pop_size: int, max_side_length: float = 10.0) -> List[np.ndarray]:
    """Generate initial population of hexagon configurations."""
    population = []
    for _ in range(pop_size):
        # Generate random positions and rotations for 11 hexagons
        individual = np.zeros((11, 3))
        for i in range(11):
            # Random position within bounds (simplified approach)
            individual[i, 0] = random.uniform(-max_side_length/2, max_side_length/2)
            individual[i, 1] = random.uniform(-max_side_length/2, max_side_length/2)
            individual[i, 2] = random.uniform(0, 360)  # Rotation angle
        population.append(individual)
    return population

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual with hexagon-specific operators."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position slightly
            mutated[i, 0] += random.gauss(0, 0.2)
            mutated[i, 1] += random.gauss(0, 0.2)
            # Mutate rotation
            mutated[i, 2] += random.gauss(0, 10)
            # Keep rotation in [0, 360)
            mutated[i, 2] = mutated[i, 2] % 360
    return mutated

def crossover_parents(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Single-point crossover on hexagon parameters
    crossover_point = random.randint(1, len(parent1) - 1)
    
    # Swap parameter sets after crossover point
    child1[crossover_point:] = parent2[crossover_point:]
    child2[crossover_point:] = parent1[crossover_point:]
    
    return child1, child2

def optimize_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Use genetic algorithm to find optimal hexagon packing."""
    start_time = time.time()
    
    # Parameters
    pop_size = 50
    generations = 100
    elite_size = 5
    mutation_rate = 0.1
    
    # Initialize population
    population = generate_initial_population(pop_size)
    best_inv_side_length = 0.0
    best_individual = None
    best_side_length = float('inf')
    
    # Start evolution
    for generation in range(generations):
        if time.time() - start_time > MAX_EVAL_TIME - 1:
            break
            
        # Evaluate fitness of population
        fitness_scores = []
        for individual in population:
            # Try different side lengths to find minimum needed
            min_side_length = 2.0  # Minimum reasonable side length
            max_side_length = 10.0
            current_side_length = (min_side_length + max_side_length) / 2
            
            # Binary search for minimum valid side length
            while max_side_length - min_side_length > 0.01:
                inv_side_length, valid_containment, valid_overlap = evaluate_packing(individual, current_side_length)
                
                if valid_containment and valid_overlap:
                    max_side_length = current_side_length
                    current_side_length = (min_side_length + max_side_length) / 2
                else:
                    min_side_length = current_side_length
                    current_side_length = (min_side_length + max_side_length) / 2
            
            # Final evaluation with the computed side length
            final_side_length = max_side_length
            inv_side_length, _, _ = evaluate_packing(individual, final_side_length)
            
            fitness_scores.append((inv_side_length, individual, final_side_length))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Update best solution
        if fitness_scores[0][0] > best_inv_side_length:
            best_inv_side_length = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
            best_side_length = fitness_scores[0][2]
        
        # Selection and reproduction
        # Keep elite
        new_population = [fitness_scores[i][1] for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            parent1 = fitness_scores[winner_idx][1]
            
            # Select second parent
            tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent2 = fitness_scores[winner_idx][1]
            
            # Crossover
            child1, child2 = crossover_parents(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
    
    # Final validation of best solution
    if best_individual is not None:
        # Refine the final solution
        min_side_length = 1.0
        max_side_length = 10.0
        refined_side_length = max_side_length
        
        # Binary search for precise minimum side length
        while max_side_length - min_side_length > 0.001:
            inv_side_length, valid_containment, valid_overlap = evaluate_packing(best_individual, (min_side_length + max_side_length) / 2)
            
            if valid_containment and valid_overlap:
                refined_side_length = (min_side_length + max_side_length) / 2
                max_side_length = refined_side_length
            else:
                min_side_length = (min_side_length + max_side_length) / 2
        
        # Final check with refined side length
        final_inv_side_length, _, _ = evaluate_packing(best_individual, refined_side_length)
        
        if final_inv_side_length > 0:
            best_inv_side_length = final_inv_side_length
            best_side_length = refined_side_length
    
    # Return results
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    return best_individual, outer_hex_data, best_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure we have a valid result
    if inner_hex_data is None:
        # Fallback to initial solution if optimization fails
        inner_hex_data = np.array([
            [0, 0, 0],      # center
            [-2.5, 0, 0],   # left
            [2.5, 0, 0],    # right
            [-1.25, 2.17, 0], # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0], # bottom-left
            [1.25, -2.17, 0], # bottom-right
            [-3.75, 2.17, 0], # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0], # far bottom-left
            [3.75, -2.17, 0], # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
