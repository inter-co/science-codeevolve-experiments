# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_WIDTH = 2 * HEX_RADIUS  # Width of unit hexagon
HEX_HEIGHT = np.sqrt(3) * HEX_RADIUS  # Height of unit hexagon
HEX_VERTICES = 6

def get_hexagon_vertices(center_x: float, center_y: float, angle_deg: float) -> np.ndarray:
    """Get vertices of a regular hexagon given center and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, HEX_VERTICES + 1)[:-1] + angle_rad
    vertices = np.column_stack([
        center_x + HEX_RADIUS * np.cos(angles),
        center_y + HEX_RADIUS * np.sin(angles)
    ])
    return vertices

def hexagon_contains_point(hex_vertices: np.ndarray, point: np.ndarray) -> bool:
    """Check if a point is inside a hexagon using ray casting."""
    # Simplified version: check if point is inside convex hull
    # For hexagon vertices, we can use a more direct approach
    # But for now, let's use the fact that hexagons are convex
    # and check distance from center vs radius
    
    # Get center of hexagon
    center = np.mean(hex_vertices, axis=0)
    # Check if point is within the hexagon bounds
    # This is a simplified approach - proper containment would require
    # more sophisticated polygon containment checking
    return True  # Placeholder - actual implementation needed

def hexagon_intersects(hex1: np.ndarray, hex2: np.ndarray) -> bool:
    """Check if two hexagons intersect using separating axis theorem."""
    # Simplified version - in practice this would be more complex
    # For now, we'll use distance between centers
    center1 = np.mean(hex1, axis=0)
    center2 = np.mean(hex2, axis=0)
    distance = np.linalg.norm(center1 - center2)
    # Two unit hexagons overlap if their centers are closer than 2 units apart
    return distance < 2.0

def compute_outer_hexagon_radius(inner_positions: np.ndarray, inner_angles: np.ndarray) -> float:
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    max_dist = 0.0
    
    # Check distance from origin to each hexagon vertex
    for i in range(len(inner_positions)):
        center_x, center_y = inner_positions[i]
        angle_deg = inner_angles[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        
        # Find maximum distance from origin to any vertex
        distances = np.linalg.norm(vertices, axis=1)
        max_dist = max(max_dist, np.max(distances))
    
    # Add some buffer to account for hexagon size
    return max_dist + HEX_RADIUS

def evaluate_solution(positions: np.ndarray, angles: np.ndarray) -> Tuple[float, bool]:
    """
    Evaluate a solution: returns (inverse_outer_radius, is_valid).
    """
    # Check if all hexagons are valid (containment and non-overlap)
    valid = True
    outer_radius = compute_outer_hexagon_radius(positions, angles)
    
    # Check containment: all vertices must be within outer hexagon
    # This is a simplified check - proper implementation would involve
    # checking containment against the actual outer hexagon boundary
    
    # Check for overlaps between hexagons
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            # Simplified overlap check based on distance
            center_i = positions[i]
            center_j = positions[j]
            distance = np.linalg.norm(center_i - center_j)
            if distance < 2.0:  # Overlapping hexagons
                valid = False
                break
        if not valid:
            break
    
    inv_radius = 1.0 / outer_radius if valid else 0.0
    return inv_radius, valid

def generate_initial_population(pop_size: int, num_hexagons: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate initial population of solutions."""
    population = []
    
    for _ in range(pop_size):
        # Random positions and angles for inner hexagons
        positions = np.random.uniform(-5, 5, (num_hexagons, 2))  # Random positions
        angles = np.random.uniform(0, 360, num_hexagons)         # Random rotations
        
        # Ensure center hexagon is at origin
        positions[0] = [0, 0]
        angles[0] = 0
        
        population.append((positions, angles))
    
    return population

def mutate_individual(positions: np.ndarray, angles: np.ndarray, mutation_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Mutate an individual solution."""
    new_positions = positions.copy()
    new_angles = angles.copy()
    
    # Mutate positions
    for i in range(len(positions)):
        if random.random() < mutation_rate:
            new_positions[i] += np.random.normal(0, 0.2, 2)
    
    # Mutate angles
    for i in range(len(angles)):
        if random.random() < mutation_rate:
            new_angles[i] += random.uniform(-30, 30)
            new_angles[i] = new_angles[i] % 360
    
    return new_positions, new_angles

def crossover(parent1: Tuple[np.ndarray, np.ndarray], parent2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    positions1, angles1 = parent1
    positions2, angles2 = parent2
    
    # Single-point crossover
    crossover_point = random.randint(1, len(positions1) - 1)
    
    new_positions = np.vstack([positions1[:crossover_point], positions2[crossover_point:]])
    new_angles = np.hstack([angles1[:crossover_point], angles2[crossover_point:]])
    
    return new_positions, new_angles

def genetic_algorithm_hexagon_packing(num_generations: int = 1000, pop_size: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evolutionary algorithm to find optimal hexagon packing."""
    # Initialize population
    population = generate_initial_population(pop_size, 11)
    best_fitness = 0.0
    best_solution = None
    
    for generation in range(num_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for positions, angles in population:
            fitness, valid = evaluate_solution(positions, angles)
            fitness_scores.append((fitness, valid, positions, angles))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Keep track of best solution
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_solution = (fitness_scores[0][2], fitness_scores[0][3])
        
        # Early stopping condition
        if best_fitness > 0.2544:  # Beat the benchmark
            break
            
        # Select top performers (tournament selection)
        selected = []
        tournament_size = 5
        for _ in range(pop_size):
            tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
            winner = max(tournament, key=lambda x: x[0])
            selected.append((winner[2], winner[3]))
        
        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]
            
            child1_pos, child1_ang = crossover(parent1, parent2)
            child2_pos, child2_ang = crossover(parent2, parent1)
            
            # Apply mutations
            child1_pos, child1_ang = mutate_individual(child1_pos, child1_ang)
            child2_pos, child2_ang = mutate_individual(child2_pos, child2_ang)
            
            new_population.extend([(child1_pos, child1_ang), (child2_pos, child2_ang)])
        
        # Keep population size constant
        population = new_population[:pop_size]
    
    if best_solution is None:
        # Return last best from final population
        final_scores = [(evaluate_solution(pos, ang)[0], pos, ang) for pos, ang in population]
        best_final = max(final_scores, key=lambda x: x[0])
        best_fitness = best_final[0]
        best_solution = (best_final[1], best_final[2])
    
    return best_solution[0], best_solution[1], 1.0 / best_fitness if best_fitness > 0 else 1000.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary algorithm approach.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run genetic algorithm
    positions, angles, outer_radius = genetic_algorithm_hexagon_packing(
        num_generations=500, 
        pop_size=30
    )
    
    # Convert to required format
    inner_hex_data = np.column_stack([positions, angles])
    
    # Outer hexagon centered at origin with computed radius
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
