# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from typing import Tuple, List
import time

def create_regular_hexagon(center=(0, 0), radius=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + radius*np.cos(angle), center[1] + radius*np.sin(angle)) for angle in angles]
    return Polygon(points)

def get_hexagon_vertices(center, radius=1, rotation=0):
    """Get vertices of a regular hexagon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + radius*np.cos(angle), center[1] + radius*np.sin(angle)) for angle in angles]

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def calculate_outer_hex_side_length(inner_hex_data, padding=0.01):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        vertices = get_hexagon_vertices(center, 1, rotation)
        all_vertices.extend(vertices)
    
    if len(all_vertices) < 3:
        return 1000
    
    # Find bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Calculate distance from center to farthest vertex
    max_dist = max(np.sqrt((x - np.mean(xs))**2 + (y - np.mean(ys))**2) for x, y in all_vertices)
    
    # For a hexagon, the side length is equal to the circumradius
    # We need to account for the fact that we're fitting hexagons
    # A reasonable estimate is to use the maximum distance from center
    # But we also need to ensure we have room for the hexagon size
    side_length = max_dist + 1 + padding  # +1 for hexagon radius, +padding for safety
    
    return side_length

def evaluate_solution(inner_hex_data):
    """Evaluate how well a solution fits"""
    # Create outer hexagon based on current arrangement
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Create outer hexagon polygon
    outer_hex = create_regular_hexagon((0, 0), outer_side_length)
    
    # Check containment and overlap for all hexagons
    total_penalty = 0
    valid = True
    
    # Check each inner hexagon
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        inner_hex = create_regular_hexagon(center, 1, rotation)
        
        # Check containment
        if not check_containment(inner_hex, outer_hex):
            total_penalty += 10000  # Large penalty for containment violation
            
        # Check overlaps with others
        for j in range(i+1, len(inner_hex_data)):
            other_center = (inner_hex_data[j][0], inner_hex_data[j][1])
            other_rotation = inner_hex_data[j][2]
            other_hex = create_regular_hexagon(other_center, 1, other_rotation)
            
            if check_overlap(inner_hex, other_hex):
                total_penalty += 1000  # Penalty for overlap
    
    # The objective is to minimize outer hexagon size
    # So we want to maximize 1/outer_side_length
    objective_value = 1.0 / (outer_side_length + total_penalty * 0.001)
    
    return objective_value, outer_side_length

def mutate_individual(individual, mutation_rate=0.1):
    """Mutate an individual solution"""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position (x, y) and rotation
            mutated[i][0] += random.uniform(-0.5, 0.5)  # x position
            mutated[i][1] += random.uniform(-0.5, 0.5)  # y position
            mutated[i][2] += random.uniform(-30, 30)   # rotation in degrees
            
    return mutated

def crossover(parent1, parent2):
    """Crossover two individuals"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Simple uniform crossover
    for i in range(len(parent1)):
        if random.random() > 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()
            
    return child1, child2

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary algorithm approach for better optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize population with diverse configurations
    population_size = 50
    generations = 100
    
    # Start with some good initial configurations
    initial_configs = [
        # Configuration 1: Star pattern
        np.array([
            [0, 0, 0],      # center
            [0, 2.0, 0],    # top
            [0, -2.0, 0],   # bottom
            [1.73, 1.0, 0], # top-right
            [-1.73, 1.0, 0],# top-left
            [1.73, -1.0, 0],# bottom-right
            [-1.73, -1.0, 0],# bottom-left
            [3.46, 2.0, 0], # far top-right
            [-3.46, 2.0, 0],# far top-left
            [3.46, -2.0, 0],# far bottom-right
            [-3.46, -2.0, 0],# far bottom-left
        ]),
        # Configuration 2: Honeycomb-like pattern
        np.array([
            [0, 0, 0],      # center
            [0, 2.0, 0],    # top
            [1.73, 1.0, 0], # top-right
            [1.73, -1.0, 0],# bottom-right
            [0, -2.0, 0],   # bottom
            [-1.73, -1.0, 0],# bottom-left
            [-1.73, 1.0, 0], # top-left
            [3.46, 2.0, 0], # far top-right
            [3.46, -2.0, 0],# far bottom-right
            [-3.46, -2.0, 0],# far bottom-left
            [-3.46, 2.0, 0],# far top-left
        ]),
        # Configuration 3: Spiral pattern
        np.array([
            [0, 0, 0],      # center
            [0, 2.0, 0],    # top
            [1.73, 1.0, 0], # top-right
            [1.73, -1.0, 0],# bottom-right
            [0, -2.0, 0],   # bottom
            [-1.73, -1.0, 0],# bottom-left
            [-1.73, 1.0, 0], # top-left
            [0, 3.0, 0],    # further top
            [0, -3.0, 0],   # further bottom
            [2.5, 2.0, 0],  # further top-right
            [-2.5, 2.0, 0], # further top-left
        ])
    ]
    
    # Generate population
    population = []
    for config in initial_configs:
        population.append(config.copy())
    
    # Fill remaining population with random configurations
    for _ in range(population_size - len(initial_configs)):
        # Random positions and rotations
        individual = np.zeros((11, 3))
        for i in range(11):
            individual[i] = [
                random.uniform(-3.0, 3.0),
                random.uniform(-3.0, 3.0),
                random.uniform(0, 360)
            ]
        population.append(individual)
    
    best_solution = None
    best_fitness = -float('inf')
    best_side_length = float('inf')
    
    # Evolutionary process
    for generation in range(generations):
        fitness_scores = []
        
        # Evaluate fitness for all individuals
        for individual in population:
            fitness, side_length = evaluate_solution(individual)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()
                best_side_length = side_length
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        
        # Keep top 50% and generate new offspring
        elite_size = population_size // 2
        new_population = population[:elite_size]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.randint(0, elite_size-1)
            parent2_idx = random.randint(0, elite_size-1)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1)
            child2 = mutate_individual(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # Final evaluation of best solution
    final_fitness, final_side_length = evaluate_solution(best_solution)
    
    # Set outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return best_solution, outer_hex_data, final_side_length


# EVOLVE-BLOCK-END
