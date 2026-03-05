# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

def create_regular_hexagon(center: np.ndarray, side_length: float, rotation_degrees: float = 0) -> np.ndarray:
    """Create a regular hexagon as a list of 6 vertices"""
    rotation_rad = np.radians(rotation_degrees)
    angle_step = 2 * np.pi / 6
    vertices = []
    for i in range(6):
        angle = i * angle_step + rotation_rad
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_hexagon(point: np.ndarray, hexagon_vertices: np.ndarray) -> bool:
    """Check if a point is inside a hexagon using ray casting"""
    # Simple implementation using winding number or cross product method
    # For efficiency, we'll use a more direct approach with cross products
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    # Check if point is on same side of all edges
    n = len(hexagon_vertices)
    inside = True
    p1 = hexagon_vertices[0]
    for i in range(1, n + 1):
        p2 = hexagon_vertices[i % n]
        if sign(point, p1, p2) < 0:
            inside = False
            break
        p1 = p2
    
    return inside

def hexagons_intersect(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
    """Check if two hexagons intersect using separating axis theorem"""
    # Get all edges of both hexagons
    edges1 = []
    edges2 = []
    
    for i in range(6):
        edge1 = hex1_vertices[(i+1)%6] - hex1_vertices[i]
        edge2 = hex2_vertices[(i+1)%6] - hex2_vertices[i]
        edges1.append(edge1)
        edges2.append(edge2)
    
    # Get normal vectors for all edges
    normals1 = []
    normals2 = []
    
    for edge in edges1:
        # Normal vector (perpendicular)
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)  # Normalize
        normals1.append(normal)
        
    for edge in edges2:
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)  # Normalize
        normals2.append(normal)
    
    # Test all axes
    all_normals = normals1 + normals2
    
    for normal in all_normals:
        # Project both polygons onto this axis
        proj1 = [np.dot(vertex, normal) for vertex in hex1_vertices]
        proj2 = [np.dot(vertex, normal) for vertex in hex2_vertices]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        # Check for overlap
        if max1 < min2 or max2 < min1:
            return False  # No overlap - hexagons don't intersect
    
    return True  # Hexagons intersect

def compute_outer_hexagon_radius(inner_hex_data: np.ndarray, outer_center: np.ndarray, side_length: float) -> float:
    """Compute the radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        # Distance from outer center to inner hex center
        dist = np.linalg.norm(np.array(center) - np.array(outer_center))
        # Add distance from center to furthest vertex of inner hexagon (sqrt(3) for unit hexagon)
        dist_to_vertex = dist + np.sqrt(3)
        max_dist = max(max_dist, dist_to_vertex)
    
    return max_dist

def evaluate_solution(inner_hex_data: np.ndarray, outer_center: np.ndarray = (0, 0)) -> Tuple[float, float]:
    """
    Evaluate how well a solution fits.
    Returns: (max_side_length, penalty_score)
    """
    # Compute side length of outer hexagon
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        # Distance from outer center to inner hex center
        dist = np.linalg.norm(np.array(center) - np.array(outer_center))
        # Add distance from center to furthest vertex of inner hexagon (sqrt(3) for unit hexagon)
        dist_to_vertex = dist + np.sqrt(3)
        max_dist = max(max_dist, dist_to_vertex)
    
    # Check for collisions between hexagons
    penalty = 0
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = inner_hex_data[i][:2]
            center2 = inner_hex_data[j][:2]
            angle1 = inner_hex_data[i][2]
            angle2 = inner_hex_data[j][2]
            
            # Create hexagon vertices
            hex1 = create_regular_hexagon(center1, 1, angle1)
            hex2 = create_regular_hexagon(center2, 1, angle2)
            
            # Check if they intersect
            if hexagons_intersect(hex1, hex2):
                penalty += 1000  # Heavy penalty for intersections
    
    # Check containment in outer hexagon
    outer_radius = max_dist
    # Create outer hexagon vertices
    outer_hex = create_regular_hexagon(outer_center, outer_radius)
    
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = inner_hex_data[i][2]
        hex_vertices = create_regular_hexagon(center, 1, angle)
        
        # Check if any vertex of inner hex is outside outer hex
        for vertex in hex_vertices:
            if not point_in_hexagon(vertex, outer_hex):
                penalty += 10000  # Heavy penalty for containment violation
    
    return outer_radius, penalty

def generate_initial_population(pop_size: int, num_hexagons: int) -> List[np.ndarray]:
    """Generate initial population of random solutions"""
    population = []
    for _ in range(pop_size):
        # Generate random positions and rotations for hexagons
        individual = np.random.rand(num_hexagons, 3)  # [x, y, angle]
        # Scale positions to reasonable range
        individual[:, 0] = (individual[:, 0] - 0.5) * 10  # x: -5 to 5
        individual[:, 1] = (individual[:, 1] - 0.5) * 10  # y: -5 to 5
        individual[:, 2] = individual[:, 2] * 360  # angle: 0 to 360 degrees
        population.append(individual)
    return population

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly change position or rotation
            if random.random() < 0.5:
                # Mutate position
                mutated[i, 0] += (random.random() - 0.5) * 2
                mutated[i, 1] += (random.random() - 0.5) * 2
            else:
                # Mutate rotation
                mutated[i, 2] += (random.random() - 0.5) * 60  # ±30 degrees
                mutated[i, 2] = mutated[i, 2] % 360
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple crossover between two individuals"""
    # Single point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def genetic_algorithm_hexagon_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Use genetic algorithm to find optimal packing
    """
    pop_size = 50
    generations = 100
    num_hexagons = 11
    
    # Initialize population
    population = generate_initial_population(pop_size, num_hexagons)
    
    best_individual = None
    best_fitness = float('inf')
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            outer_radius, penalty = evaluate_solution(individual)
            # Combine objective and penalties
            fitness = outer_radius + penalty
            fitness_scores.append(fitness)
        
        # Track best solution
        min_fitness_idx = np.argmin(fitness_scores)
        if fitness_scores[min_fitness_idx] < best_fitness:
            best_fitness = fitness_scores[min_fitness_idx]
            best_individual = population[min_fitness_idx].copy()
        
        # Selection (tournament selection)
        selected = []
        for _ in range(pop_size):
            tournament_size = 3
            tournament_indices = random.sample(range(pop_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])
        
        # Crossover and mutation
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % pop_size]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1)
            child2 = mutate_individual(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
        
        # Early stopping if improvement is minimal
        if generation > 10 and abs(best_fitness - fitness_scores[min_fitness_idx]) < 0.01:
            break
    
    # Final evaluation to get precise values
    final_radius, _ = evaluate_solution(best_individual)
    
    # Return in required format
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    return best_individual, outer_hex_data, final_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses genetic algorithm for optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use genetic algorithm approach
    start_time = time.time()
    inner_hex_data, outer_hex_data, outer_hex_side_length = genetic_algorithm_hexagon_packing()
    end_time = time.time()
    
    # Ensure we have proper format
    # Convert back to the required format if necessary
    if len(inner_hex_data.shape) == 1:
        inner_hex_data = inner_hex_data.reshape(-1, 3)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
