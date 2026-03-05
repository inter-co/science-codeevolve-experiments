# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from copy import deepcopy

# For geometric operations
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity

# Optimization imports
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import distance
from scipy.spatial.distance import cdist

# For better performance
from numba import jit, prange

@jit(nopython=True)
def hexagon_vertices_numba(center_x: float, center_y: float, radius: float, rotation: float) -> np.ndarray:
    """Compute hexagon vertices using numba for speed."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    vertices = np.empty((7, 2))
    for i in range(7):
        vertices[i, 0] = center_x + radius * np.cos(angles[i])
        vertices[i, 1] = center_y + radius * np.sin(angles[i])
    return vertices

def create_regular_hexagon(center: Tuple[float, float], radius: float, rotation: float = 0) -> Polygon:
    """Create a regular hexagon with given center, radius, and rotation."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]
    return Polygon(points)

def get_hexagon_vertices(hex_center: Tuple[float, float], radius: float, rotation: float = 0) -> List[Tuple[float, float]]:
    """Get vertices of a hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(hex_center[0] + radius * np.cos(angle), hex_center[1] + radius * np.sin(angle)) for angle in angles]

def check_containment(hexagon: Polygon, container: Polygon) -> bool:
    """Check if hexagon is fully contained within container."""
    return container.contains(hexagon) or container.covers(hexagon)

def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2) and not hex1.touches(hex2)

def calculate_outer_hexagon_radius_optimized(inner_hex_data: np.ndarray) -> float:
    """Optimized version using vectorized operations."""
    inner_hex_radius = 1.0
    
    # Vectorized computation of all vertices
    centers = inner_hex_data[:, :2]
    rotations = inner_hex_data[:, 2]
    
    # Precompute angles once
    angles = np.linspace(0, 2*np.pi, 7)  # No need to add 0 rotation here
    
    # Compute all vertices at once
    all_vertices = []
    for i in range(len(centers)):
        center = centers[i]
        rotation = rotations[i]
        # Create rotated angles
        rotated_angles = angles + np.radians(rotation)
        vertices = np.column_stack([
            center[0] + inner_hex_radius * np.cos(rotated_angles),
            center[1] + inner_hex_radius * np.sin(rotated_angles)
        ])
        all_vertices.extend(vertices)
    
    # Compute distances from origin
    all_vertices = np.array(all_vertices)
    distances = np.sqrt(all_vertices[:, 0]**2 + all_vertices[:, 1]**2)
    
    # Return maximum distance + hexagon radius for buffer
    return np.max(distances) + inner_hex_radius

def compute_hexagon_distance(hex1_center: Tuple[float, float], hex2_center: Tuple[float, float]) -> float:
    """Compute Euclidean distance between hexagon centers."""
    return np.sqrt((hex1_center[0] - hex2_center[0])**2 + (hex1_center[1] - hex2_center[1])**2)

def evaluate_solution_fast(inner_hex_data: np.ndarray) -> Tuple[float, float]:
    """
    Fast evaluation of solution with early termination for overlaps.
    Returns (penalty, inv_outer_radius)
    penalty: 0 if valid, positive otherwise
    inv_outer_radius: 1/outer_radius (higher is better)
    """
    inner_hex_radius = 1.0
    n = len(inner_hex_data)
    
    # Precompute hexagon polygons for faster access
    hexagons = []
    for i in range(n):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hexagon = create_regular_hexagon(center, inner_hex_radius, rotation)
        hexagons.append(hexagon)
    
    # Check for overlaps - early termination
    penalty = 0
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(hexagons[i], hexagons[j]):
                penalty += 1000000  # Heavy penalty for overlaps
                # Early termination if overlap found
                if penalty > 1000000:
                    # Calculate outer radius anyway for penalty calculation
                    outer_radius = calculate_outer_hexagon_radius_optimized(inner_hex_data)
                    return penalty, 1.0 / outer_radius
    
    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hexagon_radius_optimized(inner_hex_data)
    
    # Check containment - simpler approach
    outer_hexagon = create_regular_hexagon((0, 0), outer_radius)
    for hexagon in hexagons:
        if not check_containment(hexagon, outer_hexagon):
            penalty += 1000000  # Heavy penalty for containment violations
    
    # Return inverse of outer radius (we want to maximize this)
    return penalty, 1.0 / outer_radius

def evaluate_solution(inner_hex_data: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate a solution: returns (penalty, inv_outer_radius)
    penalty: 0 if valid, positive otherwise
    inv_outer_radius: 1/outer_radius (higher is better)
    """
    return evaluate_solution_fast(inner_hex_data)

def generate_initial_population() -> np.ndarray:
    """Generate a better initial population based on known good patterns."""
    # Use a known good configuration for 11 hexagons - based on mathematical analysis
    # This pattern places hexagons in a symmetric arrangement around the center
    # Based on the best known configurations for hexagon packing
    base_positions = [
        (0, 0),           # Center hexagon
        (2, 0),           # Right
        (-2, 0),          # Left
        (1, np.sqrt(3)),  # Top right
        (-1, np.sqrt(3)), # Top left
        (1, -np.sqrt(3)), # Bottom right
        (-1, -np.sqrt(3)),# Bottom left
        (3, np.sqrt(3)),  # Far right top
        (-3, np.sqrt(3)), # Far left top
        (3, -np.sqrt(3)), # Far right bottom
        (-3, -np.sqrt(3)) # Far left bottom
    ]
    
    # Add some randomness to avoid local minima but keep good structure
    individual = np.zeros((11, 3))
    for i, (x, y) in enumerate(base_positions):
        # Add less noise to center positions to maintain good structure
        if i == 0:  # Center hexagon
            individual[i][0] = x + random.uniform(-0.1, 0.1)
            individual[i][1] = y + random.uniform(-0.1, 0.1)
        else:
            individual[i][0] = x + random.uniform(-0.2, 0.2)
            individual[i][1] = y + random.uniform(-0.2, 0.2)
        individual[i][2] = random.uniform(0, 360)  # rotation angle
    
    return individual

def generate_random_individual() -> np.ndarray:
    """Generate a random individual (11 hexagons with positions and rotations)."""
    individual = np.zeros((11, 3))
    # Generate random positions and rotations for 11 hexagons
    for i in range(11):
        # Position: within a reasonable range, closer to center to improve chances of valid solution
        individual[i][0] = random.uniform(-10, 10)  # Wider range for exploration
        individual[i][1] = random.uniform(-10, 10)  # Wider range for exploration
        individual[i][2] = random.uniform(0, 360)  # rotation angle
    return individual

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position or rotation
            if random.random() < 0.7:  # 70% chance to mutate position
                # Mutate position with adaptive step sizes based on generation
                step_size = 0.2 if random.random() < 0.5 else 0.3
                mutated[i][0] += random.gauss(0, step_size)
                mutated[i][1] += random.gauss(0, step_size)
            else:
                # Mutate rotation
                mutated[i][2] += random.gauss(0, 30)  # Increased rotation change
                mutated[i][2] = mutated[i][2] % 360  # Keep within [0, 360)
    return mutated

def crossover_parents(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crossover two parents to produce two children."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Uniform crossover instead of single point for better diversity
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    
    return child1, child2

def optimize_with_scipy(inner_hex_data: np.ndarray, max_time: float) -> np.ndarray:
    """Use scipy optimization to refine the solution."""
    start_time = time.time()
    
    # Flatten parameters for scipy optimization
    def objective(params):
        # Reshape params back to hex data
        hex_data = params.reshape((11, 3))
        penalty, inv_radius = evaluate_solution(hex_data)
        # We want to maximize inv_radius, so minimize -inv_radius
        # But penalize invalid solutions heavily
        if penalty > 0:
            return 1000000 + penalty  # Very high penalty for invalid solutions
        return -inv_radius  # Minimize negative inverse radius
    
    # Bounds for positions (x, y) and rotation (angle)
    bounds = []
    for i in range(11):
        # x and y positions - expanded bounds for better exploration
        bounds.extend([(-15, 15), (-15, 15)])
        # rotation angle
        bounds.append((0, 360))
    
    try:
        # Use differential evolution with better settings for faster convergence
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200,  # Reduced iterations to save time
            popsize=20,   # Smaller population for faster convergence
            seed=42,
            disp=False,
            timeout=max_time - (time.time() - start_time),
            strategy='best1exp',  # Use exponential crossover which works better for this problem
            atol=1e-12,   # Tighter tolerances
            rtol=1e-12
        )
        
        if result.success:
            optimized_params = result.x
            return optimized_params.reshape((11, 3))
    except Exception as e:
        # If optimization fails, return original
        pass
    
    return inner_hex_data

def optimize_with_local_search(inner_hex_data: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Use local search to fine-tune the solution with enhanced strategies."""
    current = inner_hex_data.copy()
    best_penalty, best_inv_radius = evaluate_solution(current)
    best_solution = current.copy()
    
    # More sophisticated local search with multiple move types
    for iteration in range(max_iter):
        # Adaptive step sizes based on iteration
        if iteration < max_iter // 3:
            step_size = 0.15
        elif iteration < 2 * max_iter // 3:
            step_size = 0.08
        else:
            step_size = 0.04
            
        mutated = current.copy()
        
        # Try different mutation strategies
        if random.random() < 0.3:  # 30% chance for global search
            # Random perturbation of all hexagons
            for i in range(11):
                mutated[i][0] += random.gauss(0, step_size * 2)
                mutated[i][1] += random.gauss(0, step_size * 2)
                mutated[i][2] += random.gauss(0, 15)  # Less rotation change
                mutated[i][2] = mutated[i][2] % 360
        else:  # 70% chance for focused search
            # Perturb only a few random hexagons
            hexagon_indices = random.sample(range(11), min(4, 11))  # Perturb fewer hexagons
            for idx in hexagon_indices:
                mutated[idx][0] += random.gauss(0, step_size)
                mutated[idx][1] += random.gauss(0, step_size)
                mutated[idx][2] += random.gauss(0, 10)  # Moderate rotation change
                mutated[idx][2] = mutated[idx][2] % 360
        
        penalty, inv_radius = evaluate_solution(mutated)
        
        # Accept better solutions or accept with probability based on how much worse they are
        if penalty == 0 and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_solution = mutated.copy()
            current = mutated.copy()
        elif penalty == 0 and inv_radius > best_inv_radius * 0.99999:  # Accept slightly worse solutions early on
            # With low probability accept slightly worse solutions to escape local minima
            if random.random() < 0.01:
                current = mutated.copy()
    
    return best_solution

def improved_genetic_algorithm_hexagon_packing(max_generations: int = 1000, population_size: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """Improved genetic algorithm with better initialization and optimization."""
    # Initialize population with better starting points
    population = [generate_initial_population()]
    population.extend([generate_random_individual() for _ in range(population_size - 1)])
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Track progress
    start_time = time.time()
    
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            penalty, inv_radius = evaluate_solution(individual)
            # Fitness is negative penalty plus inverse radius (higher is better)
            fitness = -penalty + inv_radius
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Update best solution
        current_best_fitness, current_best_individual = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()
        
        # Early stopping if we've found a very good solution
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        # Selection: keep top 25% 
        elite_count = max(20, population_size // 4)
        elites = [ind for _, ind in fitness_scores[:elite_count]]
        
        # Generate new population through crossover and mutation
        new_population = elites.copy()
        
        # Fill remaining slots with offspring
        while len(new_population) < population_size:
            # Tournament selection with size 3
            parent1_idx = random.randint(0, len(elites) - 1)
            parent2_idx = random.randint(0, len(elites) - 1)
            
            parent1 = elites[parent1_idx]
            parent2 = elites[parent2_idx]
            
            child1, child2 = crossover_parents(parent1, parent2)
            
            # Apply mutations with adaptive rates
            mutation_rate = 0.3 if generation < max_generations//2 else 0.15
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Refine best solution with both scipy optimization and local search
    if best_individual is not None:
        # First try scipy optimization with more aggressive settings
        refined_best = optimize_with_scipy(best_individual, 55 - (time.time() - start_time))
        
        # Then try local search with more iterations
        refined_best = optimize_with_local_search(refined_best, 1500)
        
        penalty, inv_radius = evaluate_solution(refined_best)
        if penalty == 0:  # Valid solution
            best_individual = refined_best
    
    # Final evaluation of best individual
    penalty, inv_radius = evaluate_solution(best_individual)
    
    # Create outer hexagon data (centered at origin with appropriate rotation)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 1.0 / inv_radius if inv_radius > 0 else 100
    
    return best_individual, outer_hex_data, outer_hex_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses genetic algorithm with refinement for optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use improved genetic algorithm to find optimal solution
    inner_hex_data, outer_hex_data, outer_hex_side_length = improved_genetic_algorithm_hexagon_packing()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# Enhanced optimization approach specifically targeting better solutions
def enhanced_hexagon_packing_11():
    """
    Enhanced approach using hybrid optimization techniques to achieve better results.
    """
    # Start with multiple random initializations to increase chance of finding better solutions
    best_result = None
    best_inv_radius = 0
    
    # Try multiple random starts
    for attempt in range(5):
        # Set different random seeds for each attempt
        random.seed(attempt * 1000 + 42)
        np.random.seed(attempt * 1000 + 42)
        
        # Generate initial population with better starting points
        population = [generate_initial_population()]
        population.extend([generate_random_individual() for _ in range(50)])
        
        # Run GA with reduced generations for speed
        for gen in range(100):  # Fewer generations but more focused
            fitness_scores = []
            for individual in population:
                penalty, inv_radius = evaluate_solution(individual)
                fitness = -penalty + inv_radius
                fitness_scores.append((fitness, individual))
            
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            current_best_fitness, current_best_individual = fitness_scores[0]
            
            if current_best_fitness > best_inv_radius:
                best_inv_radius = current_best_fitness
                best_result = current_best_individual.copy()
                
            # Selection and reproduction
            elite_count = 15
            elites = [ind for _, ind in fitness_scores[:elite_count]]
            
            new_population = elites.copy()
            while len(new_population) < 51:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                child1, child2 = crossover_parents(parent1, parent2)
                child1 = mutate_individual(child1, 0.2)
                child2 = mutate_individual(child2, 0.2)
                new_population.extend([child1, child2])
            
            population = new_population[:51]
        
        # Refinement with local search
        if best_result is not None:
            refined = optimize_with_local_search(best_result, 1000)
            penalty, inv_radius = evaluate_solution(refined)
            if penalty == 0 and inv_radius > best_inv_radius:
                best_inv_radius = inv_radius
                best_result = refined
    
    # Final refinement with scipy optimization
    if best_result is not None:
        final_refined = optimize_with_scipy(best_result, 50)
        penalty, inv_radius = evaluate_solution(final_refined)
        if penalty == 0:
            best_result = final_refined
    
    # Create output
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 1.0 / best_inv_radius if best_inv_radius > 0 else 100
    
    return best_result, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
