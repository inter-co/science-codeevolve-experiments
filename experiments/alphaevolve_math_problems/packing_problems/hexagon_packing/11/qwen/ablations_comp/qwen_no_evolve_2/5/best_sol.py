# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import random
from typing import Tuple, List
import time

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * np.sqrt(3) / 2  # Distance from center to edge
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon

def create_regular_hexagon(center: Tuple[float, float], radius: float, angle_deg: float = 0) -> Polygon:
    """Create a regular hexagon as a Shapely polygon."""
    angle_rad = np.radians(angle_deg)
    points = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def get_hexagon_vertices(center: Tuple[float, float], radius: float, angle_deg: float = 0) -> List[Tuple[float, float]]:
    """Get the vertices of a hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

def check_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if a hexagon is fully contained within the outer hexagon."""
    return outer_hex.contains(hexagon) or outer_hex.touches(hexagon)

def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def calculate_outer_hex_side_length(inner_hex_data: np.ndarray) -> float:
    """Calculate the minimum side length of the outer hexagon needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        angle = inner_hex_data[i][2]
        hex_points = get_hexagon_vertices(center, HEX_RADIUS, angle)
        all_vertices.extend(hex_points)
    
    if len(all_vertices) == 0:
        return 1.0
    
    # Calculate bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Find the distance from center to the farthest vertex
    max_dist = max(np.sqrt(x*x + y*y) for x, y in zip(xs, ys))
    
    # The outer hexagon needs to have a radius that covers this distance
    # For a regular hexagon, the side length equals the radius
    return max_dist + HEX_RADIUS

def evaluate_solution(inner_hex_data: np.ndarray) -> Tuple[float, bool]:
    """
    Evaluate a solution: returns (inverse_side_length, is_valid)
    """
    try:
        # Calculate outer hexagon side length
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
        inv_side_length = 1.0 / outer_side_length
        
        # Create polygons for all hexagons
        hexagons = []
        for i in range(len(inner_hex_data)):
            center = (inner_hex_data[i][0], inner_hex_data[i][1])
            angle = inner_hex_data[i][2]
            hexagon = create_regular_hexagon(center, HEX_RADIUS, angle)
            hexagons.append(hexagon)
        
        # Check containment (assuming outer hexagon is centered at origin with appropriate size)
        outer_hex_radius = outer_side_length
        outer_hex = create_regular_hexagon((0, 0), outer_hex_radius)
        
        # Check containment for all inner hexagons
        for hexagon in hexagons:
            if not check_containment(hexagon, outer_hex):
                return (0.0, False)
        
        # Check overlaps between all pairs of inner hexagons
        for i in range(len(hexagons)):
            for j in range(i + 1, len(hexagons)):
                if check_overlap(hexagons[i], hexagons[j]):
                    return (0.0, False)
        
        return (inv_side_length, True)
        
    except Exception as e:
        return (0.0, False)

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate a single individual (hexagon configuration)."""
    mutated = individual.copy()
    
    # Mutate positions and rotations
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            if random.random() < 0.5:
                # Mutate position (x, y)
                mutated[i, 0] += random.gauss(0, 0.2)
                mutated[i, 1] += random.gauss(0, 0.2)
            else:
                # Mutate rotation
                mutated[i, 2] += random.gauss(0, 10)  # Degrees
                
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Single-point crossover between two parents."""
    crossover_point = random.randint(1, len(parent1) - 1)
    child = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    return child

def generate_initial_population(pop_size: int, num_hexagons: int) -> List[np.ndarray]:
    """Generate initial population with diverse arrangements."""
    population = []
    
    # Generate some diverse initial configurations
    for _ in range(pop_size):
        # Start with a reasonable spread
        individual = np.zeros((num_hexagons, 3))
        
        # Place first few hexagons in a pattern
        individual[0] = [0, 0, 0]  # Center hexagon
        
        # Place around it in a hexagonal pattern
        angles = [0, 60, 120, 180, 240, 300]
        positions = [(0, 0), (HEX_WIDTH, 0), (HEX_WIDTH/2, HEX_HEIGHT), 
                     (-HEX_WIDTH/2, HEX_HEIGHT), (-HEX_WIDTH, 0), 
                     (-HEX_WIDTH/2, -HEX_HEIGHT), (HEX_WIDTH/2, -HEX_HEIGHT)]
        
        # Fill remaining positions with small random offsets
        for i in range(1, min(len(positions), num_hexagons)):
            individual[i] = [
                positions[i][0] + random.uniform(-0.5, 0.5),
                positions[i][1] + random.uniform(-0.5, 0.5),
                random.uniform(0, 360)
            ]
            
        # Fill remaining hexagons with random positions and rotations
        for i in range(len(positions), num_hexagons):
            individual[i] = [
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(0, 360)
            ]
        
        population.append(individual)
    
    return population

def genetic_algorithm_hexagon_packing(max_generations: int = 100, pop_size: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Use genetic algorithm to find optimal hexagon packing.
    """
    # Initialize population
    population = generate_initial_population(pop_size, 11)
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        valid_individuals = []
        
        for individual in population:
            fitness, is_valid = evaluate_solution(individual)
            if is_valid:
                fitness_scores.append(fitness)
                valid_individuals.append((individual, fitness))
            else:
                fitness_scores.append(0.0)
                valid_individuals.append((individual, 0.0))
        
        # Sort by fitness
        valid_individuals.sort(key=lambda x: x[1], reverse=True)
        
        if valid_individuals and valid_individuals[0][1] > best_fitness:
            best_fitness = valid_individuals[0][1]
            best_individual = valid_individuals[0][0].copy()
        
        # Early stopping if we're close to good solution
        if best_fitness > 0.25 and generation > 20:
            break
            
        # Selection: keep top 30%
        num_keep = max(5, int(0.3 * len(valid_individuals)))
        selected = [ind for ind, _ in valid_individuals[:num_keep]]
        
        # Generate new population through crossover and mutation
        new_population = selected.copy()
        
        # Add offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_individual, np.array([0, 0, 0]), 1.0 / best_fitness if best_fitness > 0 else 1.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run genetic algorithm
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = genetic_algorithm_hexagon_packing(
            max_generations=50, pop_size=30
        )
        
        # Final validation
        fitness, is_valid = evaluate_solution(inner_hex_data)
        if not is_valid:
            # Fall back to better initial solution if GA failed
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
            outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)
            
        # Ensure we don't exceed time limits
        elapsed = time.time() - start_time
        if elapsed > 55:  # Leave 5 seconds for final processing
            pass
            
    except Exception as e:
        # Fallback to basic arrangement if anything fails
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
        outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Make sure the outer hexagon is properly sized
    outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    return inner_hex_data, np.array([0, 0, 0]), outer_hex_side_length


# EVOLVE-BLOCK-END
