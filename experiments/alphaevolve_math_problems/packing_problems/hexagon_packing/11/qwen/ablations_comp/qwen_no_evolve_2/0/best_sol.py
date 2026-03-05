# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from typing import Tuple, List
import time

def create_regular_hexagon(center: Tuple[float, float], radius: float, rotation: float = 0) -> Polygon:
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]
    return Polygon(points)

def get_hexagon_vertices(center: Tuple[float, float], radius: float, rotation: float = 0) -> List[Tuple[float, float]]:
    """Get vertices of a regular hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]

def check_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex.contains(hexagon)

def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def calculate_outer_hex_side_length(inner_hex_data: np.ndarray) -> float:
    """Calculate minimum side length of outer hexagon that contains all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        # Unit hexagon has radius 1, so vertices are at distance 1 from center
        vertices = get_hexagon_vertices(center, 1.0, rotation)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return float('inf')
    
    # Convert to numpy array for easier computation
    vertices_array = np.array(all_vertices)
    
    # Find bounding circle - we need to find the minimum radius circle that contains all vertices
    # This is equivalent to finding the maximum distance from origin to any vertex
    distances = np.sqrt(np.sum(vertices_array**2, axis=1))
    max_distance = np.max(distances)
    
    # For a regular hexagon, if we know the circumradius, the side length equals the circumradius
    # But we want the minimal outer hexagon that contains everything
    # We compute the minimal circumscribing regular hexagon
    return max_distance * 2 / np.sqrt(3)  # Approximate side length

def evaluate_solution(inner_hex_data: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate solution quality.
    Returns (penalty, inv_outer_side_length)
    """
    # Calculate outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Check constraints
    penalty = 0.0
    
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hex_poly = create_regular_hexagon(center, 1.0, rotation)
        inner_polygons.append(hex_poly)
    
    # Check for overlaps
    for i in range(len(inner_polygons)):
        for j in range(i+1, len(inner_polygons)):
            if check_overlap(inner_polygons[i], inner_polygons[j]):
                penalty += 1000.0  # Heavy penalty for overlaps
    
    # Check containment
    outer_hex = create_regular_hexagon((0, 0), outer_side_length, 0)
    for poly in inner_polygons:
        if not check_containment(poly, outer_hex):
            penalty += 1000.0  # Heavy penalty for containment violations
    
    # Return negative penalty (since we're maximizing) and inverse side length
    inv_side_length = 1.0 / outer_side_length if outer_side_length > 0 else 0.0
    return penalty, inv_side_length

def generate_initial_population(pop_size: int, num_hexagons: int) -> np.ndarray:
    """Generate initial population of random hexagon configurations."""
    population = []
    for _ in range(pop_size):
        # Generate random positions and rotations for 11 hexagons
        hex_data = np.zeros((num_hexagons, 3))
        for i in range(num_hexagons):
            # Random position within a reasonable area
            hex_data[i][0] = random.uniform(-5, 5)  # x coordinate
            hex_data[i][1] = random.uniform(-5, 5)  # y coordinate
            hex_data[i][2] = random.uniform(0, 360)  # rotation angle
        population.append(hex_data)
    return population

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual by slightly changing positions and rotations."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position
            mutated[i][0] += random.gauss(0, 0.5)
            mutated[i][1] += random.gauss(0, 0.5)
        if random.random() < mutation_rate:
            # Mutate rotation
            mutated[i][2] += random.gauss(0, 15)
            mutated[i][2] %= 360
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Simple crossover between two parents."""
    child = parent1.copy()
    # Take half from parent1, half from parent2
    split_point = len(parent1) // 2
    child[split_point:] = parent2[split_point:]
    return child

def hexagon_packing_11():
    """
    Evolved approach: Genetic algorithm for optimal hexagon packing.
    Uses evolutionary computation to find the best arrangement of 11 unit hexagons.
    """
    # Parameters
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Generate initial population
    population = generate_initial_population(pop_size, 11)
    
    best_fitness = float('-inf')
    best_individual = None
    
    # Evolutionary process
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            penalty, inv_side_length = evaluate_solution(individual)
            # Fitness is inverse side length minus penalties
            fitness = inv_side_length - penalty * 0.001
            fitness_scores.append(fitness)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection: keep top individuals
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite = [population[i] for i in sorted_indices[:elite_size]]
        
        # Create new population through crossover and mutation
        new_population = elite.copy()
        
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = population[random.choice(sorted_indices[:pop_size//2])]
            parent2 = population[random.choice(sorted_indices[:pop_size//2])]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    # Final evaluation of best individual
    penalty, inv_side_length = evaluate_solution(best_individual)
    
    # Adjust final positions to ensure containment
    outer_side_length = calculate_outer_hex_side_length(best_individual)
    
    # Return the best solution found
    return best_individual, np.array([0, 0, 0]), outer_side_length


# EVOLVE-BLOCK-END
