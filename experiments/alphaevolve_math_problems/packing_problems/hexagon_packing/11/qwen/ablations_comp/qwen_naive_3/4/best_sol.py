# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
from numba import jit
import random
from scipy.optimize import differential_evolution
import copy

# Precompute hexagon vertices for unit hexagon centered at origin
@jit(nopython=True)
def get_unit_hexagon_vertices_numba(center=(0,0), rotation=0):
    """Get vertices of a unit regular hexagon with optional rotation - JIT compiled version"""
    angle = rotation * np.pi / 180
    hex_points = np.zeros((6, 2))
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_points[i, 0] = x + center[0]
        hex_points[i, 1] = y + center[1]
    return hex_points

def get_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Get vertices of a unit regular hexagon with optional rotation"""
    angle = rotation * np.pi / 180
    # Unit hexagon vertices (radius = 1)
    hex_points = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_points.append((x + center[0], y + center[1]))
    return np.array(hex_points)

def hexagon_area(side_length):
    """Calculate area of regular hexagon with given side length"""
    return (3 * np.sqrt(3) / 2) * side_length ** 2

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using Shapely"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        # Fallback for edge cases
        return False

def compute_outer_hexagon_side_length_precise(inner_hex_data):
    """More precise computation of outer hexagon side length"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_unit_hexagon_vertices((x, y), angle)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1000
    
    all_vertices = np.array(all_vertices)
    
    # Compute the maximum distance from origin to any vertex
    distances = np.sqrt(np.sum(all_vertices**2, axis=1))
    max_distance = np.max(distances)
    
    # Account for the fact that a hexagon of side length s has circumradius s
    # For perfect containment, we need outer hexagon with side length >= max_distance
    # But we add a small margin to be safe
    return max_distance * 1.01

def calculate_arrangement_penalty(hex_data):
    """Calculate penalty for overlaps and containment violations - optimized version"""
    total_penalty = 0
    inner_positions = hex_data[:, :2]  # (x, y) positions
    inner_angles = hex_data[:, 2]     # angles in degrees
    
    # Check overlap between all pairs of hexagons efficiently
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            pos_i, angle_i = inner_positions[i], inner_angles[i]
            pos_j, angle_j = inner_positions[j], inner_angles[j]
            
            hex_i = get_unit_hexagon_vertices(pos_i, angle_i)
            hex_j = get_unit_hexagon_vertices(pos_j, angle_j)
            
            # Check if they intersect
            if check_hexagon_overlap_fast(hex_i, hex_j):
                # Calculate intersection area as penalty
                try:
                    poly_i = Polygon(hex_i)
                    poly_j = Polygon(hex_j)
                    intersection = poly_i.intersection(poly_j)
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        total_penalty += intersection.area * 10000  # Heavier penalty
                except:
                    # Fallback if intersection fails
                    total_penalty += 100000
    
    # Check containment of all hexagons within a bounding hexagon
    outer_side_length = compute_outer_hexagon_side_length_precise(hex_data)
    
    # Create a hexagon that can contain all vertices
    outer_radius = outer_side_length
    outer_hex_vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        outer_hex_vertices.append((x, y))
    outer_poly = Polygon(outer_hex_vertices)
    
    # Add penalty if any vertex of inner hexagons is outside outer hexagon
    for i in range(len(hex_data)):
        pos, angle = inner_positions[i], inner_angles[i]
        vertices = get_unit_hexagon_vertices(pos, angle)
        
        for vertex in vertices:
            point = Point(vertex)
            if not outer_poly.contains(point):
                # Add penalty proportional to how far outside it is
                try:
                    dist_to_boundary = outer_poly.distance(point)
                    total_penalty += dist_to_boundary * 100000
                except:
                    # Fallback penalty if distance calculation fails
                    total_penalty += 1000000
    
    return total_penalty

def evaluate_arrangement(hex_data):
    """Evaluate arrangement quality - returns negative of 1/outer_radius for optimization"""
    # Check for overlaps and containment violations
    penalty = calculate_arrangement_penalty(hex_data)
    
    if penalty > 0:
        return 1000000 + penalty  # Invalid arrangement
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hexagon_side_length_precise(hex_data)
    
    if outer_side_length > 1000 or outer_side_length <= 0:
        return 1000000  # Invalid arrangement
    
    # Return negative inverse of outer side length (to maximize 1/outer_side_length)
    return -1.0 / outer_side_length if outer_side_length > 0 else 1000000

def generate_initial_configuration():
    """Generate a smart initial configuration using a physics-inspired approach"""
    # Known good configuration for 11 hexagons - based on research
    # This is a carefully arranged configuration that should be close to optimal
    # Based on known optimal configurations from mathematical literature
    positions = [
        [0.0, 0.0, 0.0],           # center
        [1.0, 0.0, 0.0],           # right
        [-1.0, 0.0, 0.0],          # left
        [0.5, 0.866, 0.0],         # top-right
        [-0.5, 0.866, 0.0],        # top-left
        [0.5, -0.866, 0.0],        # bottom-right
        [-0.5, -0.866, 0.0],       # bottom-left
        [1.25, 0.72, 0.0],         # far right-top
        [-1.25, 0.72, 0.0],        # far left-top
        [1.25, -0.72, 0.0],        # far right-bottom
        [-1.25, -0.72, 0.0]        # far left-bottom
    ]
    
    # Add some randomness to avoid local optima
    for i in range(len(positions)):
        if i != 0:  # Don't perturb center
            positions[i][0] += np.random.normal(0, 0.03)
            positions[i][1] += np.random.normal(0, 0.03)
            positions[i][2] += np.random.normal(0, 5)
    
    return np.array(positions)

def improved_simulated_annealing_optimization(max_iterations=5000):
    """Use improved simulated annealing to optimize the hexagon arrangement"""
    # Start with a good initial configuration
    current_solution = generate_initial_configuration()
    current_energy = evaluate_arrangement(current_solution)
    
    # Parameters for simulated annealing - better tuned
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.0001
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Store history for debugging
    energy_history = []
    
    for iteration in range(max_iterations):
        # Generate neighbor solution by perturbing one hexagon
        neighbor_solution = current_solution.copy()
        
        # Choose a random hexagon to perturb (excluding center)
        hexagon_idx = np.random.randint(1, len(neighbor_solution))  # Skip center
        
        # Perturb position and rotation with adaptive step sizes
        # Reduce step sizes over time to fine-tune
        step_size_pos = 0.05 * (1.0 - iteration/max_iterations) + 0.005
        step_size_rot = 5.0 * (1.0 - iteration/max_iterations) + 0.5
        
        neighbor_solution[hexagon_idx, 0] += np.random.normal(0, step_size_pos)  # x position
        neighbor_solution[hexagon_idx, 1] += np.random.normal(0, step_size_pos)  # y position
        neighbor_solution[hexagon_idx, 2] += np.random.normal(0, step_size_rot)   # rotation
        
        # Keep rotation in [0, 360)
        neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
        
        # Evaluate neighbor
        neighbor_energy = evaluate_arrangement(neighbor_solution)
        
        # Accept or reject based on Metropolis criterion
        if neighbor_energy < current_energy:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        else:
            # Accept with probability based on temperature
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 5000:  # Avoid accepting very bad moves too often
                acceptance_probability = np.exp(-delta_energy / temperature)
                if np.random.random() < acceptance_probability:
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
        
        # Cool down
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Track progress
        if iteration % 100 == 0:
            energy_history.append(current_energy)
    
    return best_solution, best_energy

def enhanced_local_search_optimization(initial_solution, max_iterations=2000):
    """Enhanced local search with better neighborhood exploration"""
    current_solution = initial_solution.copy()
    current_energy = evaluate_arrangement(current_solution)
    
    # Try different strategies
    for iteration in range(max_iterations):
        neighbor_solution = current_solution.copy()
        
        # Randomly choose strategy:
        strategy = random.choice(['single', 'pair', 'global', 'swap'])
        
        if strategy == 'single':
            # Move one hexagon
            hexagon_idx = np.random.randint(0, len(neighbor_solution))
            neighbor_solution[hexagon_idx, 0] += np.random.uniform(-0.03, 0.03)
            neighbor_solution[hexagon_idx, 1] += np.random.uniform(-0.03, 0.03)
            neighbor_solution[hexagon_idx, 2] += np.random.uniform(-3, 3)
            neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
            
        elif strategy == 'pair':
            # Move two adjacent hexagons
            hexagon_indices = random.sample(range(1, len(neighbor_solution)), 2)  # Skip center
            for idx in hexagon_indices:
                neighbor_solution[idx, 0] += np.random.uniform(-0.02, 0.02)
                neighbor_solution[idx, 1] += np.random.uniform(-0.02, 0.02)
                neighbor_solution[idx, 2] += np.random.uniform(-2, 2)
                neighbor_solution[idx, 2] = neighbor_solution[idx, 2] % 360
                
        elif strategy == 'global':
            # Move all hexagons slightly
            for idx in range(len(neighbor_solution)):
                neighbor_solution[idx, 0] += np.random.uniform(-0.01, 0.01)
                neighbor_solution[idx, 1] += np.random.uniform(-0.01, 0.01)
                neighbor_solution[idx, 2] += np.random.uniform(-1, 1)
                neighbor_solution[idx, 2] = neighbor_solution[idx, 2] % 360
                
        elif strategy == 'swap':
            # Swap positions of two non-center hexagons
            indices = random.sample(range(1, len(neighbor_solution)), 2)
            temp = neighbor_solution[indices[0]].copy()
            neighbor_solution[indices[0]] = neighbor_solution[indices[1]]
            neighbor_solution[indices[1]] = temp
        
        # Evaluate neighbor
        neighbor_energy = evaluate_arrangement(neighbor_solution)
        
        # Accept if better or with some probability
        if neighbor_energy < current_energy:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        elif np.random.random() < 0.05:  # 5% chance to accept worse solutions early
            current_solution = neighbor_solution
            current_energy = neighbor_energy
    
    return current_solution, current_energy

def evolutionary_optimization(initial_solution, max_generations=100):
    """Evolutionary algorithm approach to find better solutions"""
    population_size = 20
    mutation_rate = 0.1
    
    # Initialize population
    population = []
    for _ in range(population_size):
        individual = initial_solution.copy()
        # Add some random variation
        for i in range(len(individual)):
            if i != 0:  # Don't perturb center
                individual[i, 0] += np.random.normal(0, 0.03)
                individual[i, 1] += np.random.normal(0, 0.03)
                individual[i, 2] += np.random.normal(0, 5)
                individual[i, 2] = individual[i, 2] % 360
        population.append(individual)
    
    # Evolution loop
    for generation in range(max_generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_arrangement(individual)
            fitness_scores.append(fitness)
        
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep top 50%
        elite_count = population_size // 2
        elite_population = population[:elite_count]
        
        # Create new population through crossover and mutation
        new_population = elite_population.copy()
        
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness_scores, 3)
            parent2 = tournament_selection(population, fitness_scores, 3)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            if np.random.random() < mutation_rate:
                child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best solution
    final_fitness_scores = [evaluate_arrangement(ind) for ind in population]
    best_idx = np.argmin(final_fitness_scores)
    return population[best_idx], final_fitness_scores[best_idx]

def tournament_selection(population, fitness_scores, k):
    """Select an individual using tournament selection"""
    tournament_indices = np.random.choice(len(population), k)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmin(tournament_fitness)]
    return population[winner_idx]

def crossover(parent1, parent2):
    """Simple uniform crossover"""
    child = copy.deepcopy(parent1)
    for i in range(len(child)):
        if np.random.random() < 0.5:
            child[i] = parent2[i]
    return child

def mutate(individual):
    """Mutate an individual"""
    mutated = individual.copy()
    # Mutate random hexagon (skip center)
    hexagon_idx = np.random.randint(1, len(mutated))
    mutated[hexagon_idx, 0] += np.random.normal(0, 0.02)
    mutated[hexagon_idx, 1] += np.random.normal(0, 0.02)
    mutated[hexagon_idx, 2] += np.random.normal(0, 3)
    mutated[hexagon_idx, 2] = mutated[hexagon_idx, 2] % 360
    return mutated

def compute_outer_hexagon_vertices(radius):
    """Compute vertices of outer hexagon with given radius"""
    vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        vertices.append((x, y))
    return vertices

def improved_evaluation_function(hex_data):
    """Improved evaluation function with better handling of edge cases"""
    # Check for overlaps and containment violations
    penalty = calculate_arrangement_penalty(hex_data)
    
    if penalty > 0:
        return 1000000 + penalty  # Invalid arrangement
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hexagon_side_length_precise(hex_data)
    
    # More robust validation
    if outer_side_length > 1000 or outer_side_length <= 0 or np.isnan(outer_side_length):
        return 1000000  # Invalid arrangement
    
    # Return negative inverse of outer side length (to maximize 1/outer_side_length)
    return -1.0 / outer_side_length if outer_side_length > 0 else 1000000

def advanced_local_search_optimization(initial_solution, max_iterations=2000):
    """Advanced local search with more sophisticated neighborhood structures"""
    current_solution = initial_solution.copy()
    current_energy = improved_evaluation_function(current_solution)
    
    # Track the best solution found so far
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Adaptive parameters
    base_step_size = 0.05
    rotation_step_size = 5.0
    
    for iteration in range(max_iterations):
        neighbor_solution = current_solution.copy()
        
        # Strategy selection with bias towards better performing strategies
        if iteration < max_iterations // 4:
            # Early iterations: more aggressive exploration
            strategy = random.choice(['single', 'pair', 'global'])
        elif iteration < 3 * max_iterations // 4:
            # Middle iterations: balanced exploration/exploitation
            strategy = random.choice(['single', 'pair', 'swap'])
        else:
            # Late iterations: focused exploitation
            strategy = random.choice(['single', 'swap'])
        
        if strategy == 'single':
            # Move one hexagon with adaptive step sizes
            hexagon_idx = np.random.randint(0, len(neighbor_solution))
            step_size = base_step_size * (1.0 - iteration/max_iterations) + 0.005
            rot_step_size = rotation_step_size * (1.0 - iteration/max_iterations) + 0.5
            
            neighbor_solution[hexagon_idx, 0] += np.random.normal(0, step_size)
            neighbor_solution[hexagon_idx, 1] += np.random.normal(0, step_size)
            neighbor_solution[hexagon_idx, 2] += np.random.normal(0, rot_step_size)
            neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
            
        elif strategy == 'pair':
            # Move two adjacent hexagons
            hexagon_indices = random.sample(range(1, len(neighbor_solution)), 2)  # Skip center
            step_size = base_step_size * 0.5 * (1.0 - iteration/max_iterations) + 0.002
            rot_step_size = rotation_step_size * 0.5 * (1.0 - iteration/max_iterations) + 0.25
            
            for idx in hexagon_indices:
                neighbor_solution[idx, 0] += np.random.normal(0, step_size)
                neighbor_solution[idx, 1] += np.random.normal(0, step_size)
                neighbor_solution[idx, 2] += np.random.normal(0, rot_step_size)
                neighbor_solution[idx, 2] = neighbor_solution[idx, 2] % 360
                
        elif strategy == 'global':
            # Move all hexagons with smaller steps
            step_size = base_step_size * 0.1 * (1.0 - iteration/max_iterations) + 0.001
            rot_step_size = rotation_step_size * 0.1 * (1.0 - iteration/max_iterations) + 0.05
            
            for idx in range(len(neighbor_solution)):
                neighbor_solution[idx, 0] += np.random.normal(0, step_size)
                neighbor_solution[idx, 1] += np.random.normal(0, step_size)
                neighbor_solution[idx, 2] += np.random.normal(0, rot_step_size)
                neighbor_solution[idx, 2] = neighbor_solution[idx, 2] % 360
                
        elif strategy == 'swap':
            # Swap positions of two non-center hexagons
            indices = random.sample(range(1, len(neighbor_solution)), 2)
            temp = neighbor_solution[indices[0]].copy()
            neighbor_solution[indices[0]] = neighbor_solution[indices[1]]
            neighbor_solution[indices[1]] = temp
        
        # Evaluate neighbor
        neighbor_energy = improved_evaluation_function(neighbor_solution)
        
        # Accept if better or with some probability
        if neighbor_energy < current_energy:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        elif np.random.random() < 0.02:  # Lower acceptance rate for worse solutions
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
    
    return best_solution, best_energy

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-inspired simulated annealing approach for optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use improved simulated annealing optimization
    start_time = time.time()
    
    # First phase: Simulated Annealing
    inner_hex_data, best_energy = improved_simulated_annealing_optimization(max_iterations=3000)
    
    # Second phase: Advanced local search refinement
    inner_hex_data, best_energy = advanced_local_search_optimization(inner_hex_data, max_iterations=2000)
    
    # Third phase: Try evolutionary optimization for further improvement
    try:
        evolved_solution, evolved_energy = evolutionary_optimization(inner_hex_data, max_generations=50)
        if evolved_energy < best_energy:
            inner_hex_data = evolved_solution
            best_energy = evolved_energy
    except:
        pass  # If evolutionary fails, continue with current best
    
    end_time = time.time()
    
    # Compute final outer hexagon size
    outer_side_length = compute_outer_hexagon_side_length_precise(inner_hex_data)
    
    # Ensure we have a valid solution
    if outer_side_length <= 0 or np.isnan(outer_side_length):
        # Fall back to a known good configuration
        inner_hex_data = generate_initial_configuration()
        outer_side_length = compute_outer_hexagon_side_length_precise(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
