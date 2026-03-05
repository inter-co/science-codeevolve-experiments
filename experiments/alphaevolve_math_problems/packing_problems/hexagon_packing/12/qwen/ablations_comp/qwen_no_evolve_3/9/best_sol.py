# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from typing import Tuple, List
import math


def create_unit_hexagon(center: Tuple[float, float], angle_deg: float = 0) -> Polygon:
    """Create a unit regular hexagon with given center and rotation."""
    angle_rad = math.radians(angle_deg)
    # Vertices of unit hexagon centered at origin
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = math.cos(theta)
        y = math.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return Polygon(vertices)


def check_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex.contains(hexagon)


def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)


def calculate_outer_hexagon_radius(inner_hex_data: np.ndarray) -> float:
    """Calculate minimum radius needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        angle = inner_hex_data[i][2]
        hexagon = create_unit_hexagon(center, angle)
        
        # Add all vertices of this hexagon
        for vertex in hexagon.exterior.coords[:-1]:  # Exclude repeated last vertex
            all_vertices.append(vertex)
    
    # Find the maximum distance from origin to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = math.sqrt(vertex[0]**2 + vertex[1]**2)
        max_dist = max(max_dist, dist)
    
    return max_dist


def evaluate_configuration(inner_hex_data: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate configuration: returns (penalty, outer_radius)
    penalty: 0 if valid, positive if invalid
    outer_radius: radius of smallest enclosing hexagon
    """
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        angle = inner_hex_data[i][2]
        hexagon = create_unit_hexagon(center, angle)
        inner_polygons.append(hexagon)
    
    # Check for overlaps
    penalty = 0.0
    for i in range(len(inner_polygons)):
        for j in range(i+1, len(inner_polygons)):
            if check_overlap(inner_polygons[i], inner_polygons[j]):
                penalty += 1000.0  # Large penalty for overlaps
    
    # Calculate minimum outer radius that contains all hexagons
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data)
    
    # Additional penalty for configurations that might not be optimal
    # We want to minimize outer radius, so we'll use it as part of our objective
    # But we also need to ensure it's reasonable
    if penalty > 0:
        return penalty, outer_radius
    
    # Check containment - if any hexagon extends beyond a circle of radius 10
    # This is just a safety check
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        max_dist = max(max_dist, math.sqrt(center[0]**2 + center[1]**2))
    
    # If any hexagon center is too far, penalize heavily
    if max_dist > 15:
        penalty += 5000.0
    
    return penalty, outer_radius


def generate_initial_population(pop_size: int) -> List[np.ndarray]:
    """Generate diverse initial population of hexagon arrangements."""
    population = []
    
    # Generate some symmetric configurations
    for _ in range(pop_size // 3):
        # Configuration 1: 2x2 grid with central hexagon
        config = np.zeros((12, 3))
        # Center hexagon
        config[0] = [0, 0, 0]
        # Surrounding hexagons in a pattern
        positions = [(0, 2), (sqrt(3), 1), (-sqrt(3), 1), (0, -2), (sqrt(3), -1), (-sqrt(3), -1)]
        for i, pos in enumerate(positions):
            config[i+1] = [pos[0], pos[1], 0]
        # Add remaining positions
        config[7] = [2*sqrt(3), 0, 0]
        config[8] = [-2*sqrt(3), 0, 0]
        config[9] = [0, 2*sqrt(3), 0]
        config[10] = [0, -2*sqrt(3), 0]
        config[11] = [sqrt(3), sqrt(3), 0]
        population.append(config)
    
    # Generate random configurations
    for _ in range(pop_size // 3):
        config = np.random.rand(12, 3) * 8 - 4  # Random positions in [-4, 4]
        config[:, 2] = np.random.rand(12) * 360  # Random angles
        population.append(config)
    
    # Generate another symmetric configuration
    for _ in range(pop_size // 3):
        config = np.zeros((12, 3))
        # Hexagonal arrangement around center
        angles = [i * 60 for i in range(6)]  # 6 directions
        distances = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]  # All same distance
        for i in range(6):
            angle_rad = math.radians(angles[i])
            x = distances[i] * math.cos(angle_rad)
            y = distances[i] * math.sin(angle_rad)
            config[i] = [x, y, 0]
        
        # Place additional hexagons
        config[6] = [0, 3, 0]
        config[7] = [0, -3, 0]
        config[8] = [3, 0, 0]
        config[9] = [-3, 0, 0]
        config[10] = [1.5, 1.5, 0]
        config[11] = [-1.5, -1.5, 0]
        population.append(config)
    
    return population


def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual configuration."""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate position
            mutated[i, 0] += np.random.normal(0, 0.2)
            mutated[i, 1] += np.random.normal(0, 0.2)
            
        if np.random.random() < mutation_rate:
            # Mutate angle
            mutated[i, 2] += np.random.normal(0, 15)
            mutated[i, 2] %= 360
    
    return mutated


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Single-point crossover between two parents."""
    crossover_point = np.random.randint(1, len(parent1))
    child = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    return child


def hexagon_packing_12():
    """
    Evolves an optimal arrangement of 12 unit regular hexagons using genetic algorithm.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) 
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) 
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Constants
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population
    population = generate_initial_population(pop_size)
    
    best_fitness = float('inf')
    best_individual = None
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            penalty, radius = evaluate_configuration(individual)
            # Fitness is negative penalty plus radius (smaller radius better)
            fitness = penalty + radius * 1000  # Weight radius heavily
            fitness_scores.append(fitness)
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Track best individual
        if fitness_scores[0] < best_fitness:
            best_fitness = fitness_scores[0]
            best_individual = population[0].copy()
        
        # Elitism: keep best individuals
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 3
            selected_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmin(tournament_fitness)]
            
            # Clone winner
            offspring = population[winner_idx].copy()
            
            # Apply crossover with another random individual
            if np.random.random() < 0.7:
                other_idx = np.random.choice(len(population))
                offspring = crossover(offspring, population[other_idx])
            
            # Apply mutation
            offspring = mutate_individual(offspring, mutation_rate)
            
            new_population.append(offspring)
        
        population = new_population
    
    # Final evaluation of best individual
    penalty, outer_radius = evaluate_configuration(best_individual)
    
    # Convert to required output format
    inner_hex_data = best_individual.copy()
    
    # Outer hexagon centered at origin with appropriate size
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = outer_radius + 0.5  # Add buffer for safety
    
    # Ensure the outer hexagon is actually large enough
    # Let's recalculate with proper hexagon bounds
    max_distance = 0
    for i in range(len(inner_hex_data)):
        x, y = inner_hex_data[i][0], inner_hex_data[i][1]
        distance = math.sqrt(x*x + y*y)
        max_distance = max(max_distance, distance)
    
    # Account for hexagon size (radius is sqrt(3) for unit hexagon)
    outer_hex_side_length = max_distance + 1.732  # sqrt(3) for unit hexagon radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# Helper function for square root
def sqrt(x):
    return math.sqrt(x)


# EVOLVE-BLOCK-END
