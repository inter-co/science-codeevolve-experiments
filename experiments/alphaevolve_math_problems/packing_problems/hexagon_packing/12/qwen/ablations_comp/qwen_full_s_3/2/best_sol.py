# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import math
import random
import time
from itertools import permutations

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(theta)
        y = center_y + side_length * math.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def create_outer_hexagon(side_length):
    """Create vertices of outer hexagon centered at origin."""
    return generate_hexagon_vertices(0, 0, 0, side_length)

def point_in_polygon(point, polygon_vertices):
    """Check if a point is inside a polygon using Shapely."""
    poly = Polygon(polygon_vertices)
    pt = Point(point)
    return poly.contains(pt)

def hexagon_contains_point(hex_vertices, point):
    """Check if a point is inside a hexagon."""
    return point_in_polygon(point, hex_vertices)

def hexagon_intersects(hex1_vertices, hex2_vertices):
    """Check if two hexagons intersect using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hexagon_side_length(inner_hex_data, outer_radius_guess=None):
    """Calculate the minimum outer hexagon side length that contains all inner hexagons."""
    if outer_radius_guess is None:
        # Estimate based on maximum distance from center
        max_dist = 0
        for x, y, angle in inner_hex_data:
            # Get vertices of this hexagon
            vertices = generate_hexagon_vertices(x, y, angle)
            # Find maximum distance from center (0,0) to any vertex
            for vx, vy in vertices:
                dist = math.sqrt(vx*vx + vy*vy)
                max_dist = max(max_dist, dist)
        # Add some buffer for safety
        outer_radius_guess = max_dist + 1.0
    
    # Binary search for tightest fit with better precision
    min_radius = 0.1
    max_radius = 10.0
    
    # Binary search for tightest fit - limited iterations for time constraint
    for _ in range(20):  # Fewer iterations to save time
        mid_radius = (min_radius + max_radius) / 2
        outer_hex = create_outer_hexagon(mid_radius)
        
        # Check if all inner hexagons fit
        all_fit = True
        for x, y, angle in inner_hex_data:
            inner_hex = generate_hexagon_vertices(x, y, angle)
            # Check if all vertices of inner hex are within outer hex
            for vertex in inner_hex:
                if not hexagon_contains_point(outer_hex, vertex):
                    all_fit = False
                    break
            if not all_fit:
                break
        
        if all_fit:
            max_radius = mid_radius
        else:
            min_radius = mid_radius
    
    return max_radius

def evaluate_solution(inner_hex_data, outer_radius=None):
    """Comprehensive evaluation of a solution."""
    if outer_radius is None:
        outer_radius = calculate_outer_hexagon_side_length(inner_hex_data)
    
    # Create outer hexagon
    outer_hex = create_outer_hexagon(outer_radius)
    
    # Check containment - more thorough check
    for x, y, angle in inner_hex_data:
        inner_hex = generate_hexagon_vertices(x, y, angle)
        # Check all vertices are within outer hexagon
        for vertex in inner_hex:
            if not hexagon_contains_point(outer_hex, vertex):
                return False, outer_radius, float('inf')
    
    # Check overlaps - more thorough check
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]
            hex1 = generate_hexagon_vertices(x1, y1, angle1)
            hex2 = generate_hexagon_vertices(x2, y2, angle2)
            
            if hexagon_intersects(hex1, hex2):
                return False, outer_radius, float('inf')
    
    return True, outer_radius, 1.0 / outer_radius

def rotate_hexagon_around_center(hex_data, center_x, center_y, angle_deg):
    """Rotate all hexagons around a center point."""
    rad_angle = math.radians(angle_deg)
    cos_a = math.cos(rad_angle)
    sin_a = math.sin(rad_angle)
    
    rotated = []
    for x, y, angle in hex_data:
        # Translate to origin
        tx = x - center_x
        ty = y - center_y
        
        # Rotate
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        
        # Translate back
        new_x = rx + center_x
        new_y = ry + center_y
        
        # Add rotation to existing angle
        new_angle = (angle + angle_deg) % 360
        
        rotated.append([new_x, new_y, new_angle])
    
    return rotated

def create_symmetric_configurations():
    """Generate multiple symmetric starting configurations based on D6 symmetry group."""
    configs = []
    
    # Configuration 1: Central hexagon with 11 others arranged in rings
    config1 = [
        [0.0, 0.0, 0.0],  # center
        [0.0, 2.0, 0.0],  # top
        [0.0, -2.0, 0.0],  # bottom
        [1.732, 1.0, 0.0],  # top-right
        [-1.732, 1.0, 0.0],  # top-left
        [1.732, -1.0, 0.0],  # bottom-right
        [-1.732, -1.0, 0.0],  # bottom-left
        [3.464, 0.0, 0.0],  # far right
        [-3.464, 0.0, 0.0],  # far left
        [1.732, 2.0, 0.0],  # top far-right
        [-1.732, 2.0, 0.0],  # top far-left
        [1.732, -2.0, 0.0],  # bottom far-right
    ]
    configs.append(config1)
    
    # Configuration 2: More compact arrangement
    config2 = [
        [0.0, 0.0, 0.0],  # center
        [0.0, 1.8, 0.0],  # top
        [0.0, -1.8, 0.0],  # bottom
        [1.55, 0.9, 0.0],  # top-right
        [-1.55, 0.9, 0.0],  # top-left
        [1.55, -0.9, 0.0],  # bottom-right
        [-1.55, -0.9, 0.0],  # bottom-left
        [3.1, 0.0, 0.0],  # far right
        [-3.1, 0.0, 0.0],  # far left
        [1.55, 2.7, 0.0],  # top far-right
        [-1.55, 2.7, 0.0],  # top far-left
        [1.55, -2.7, 0.0],  # bottom far-right
    ]
    configs.append(config2)
    
    # Configuration 3: Highly symmetric with rotations
    config3 = [
        [0.0, 0.0, 0.0],
        [0.0, 2.1, 0.0],
        [1.81, 1.05, 0.0],
        [1.81, -1.05, 0.0],
        [0.0, -2.1, 0.0],
        [-1.81, -1.05, 0.0],
        [-1.81, 1.05, 0.0],
        [3.62, 0.0, 0.0],
        [2.715, 1.57, 0.0],
        [2.715, -1.57, 0.0],
        [-2.715, -1.57, 0.0],
        [-2.715, 1.57, 0.0]
    ]
    configs.append(config3)
    
    return configs

def fitness_function(hex_data):
    """Fitness function that rewards tight packing and penalizes overlaps."""
    # Check validity first
    valid, outer_radius, inv_radius = evaluate_solution(hex_data)
    
    if not valid:
        # Large penalty for invalid configurations
        return -1000000
    
    # Reward tight packing (higher inverse radius)
    fitness = inv_radius
    
    # Additional reward for symmetric arrangements
    # Simple measure of how close positions are to symmetric placement
    symmetry_penalty = 0
    
    # For a good symmetric packing, hexagons should be roughly evenly distributed
    # We compute the variance of distances from center
    distances = []
    for x, y, _ in hex_data:
        dist = math.sqrt(x*x + y*y)
        distances.append(dist)
    
    # Low variance in distances indicates more uniform distribution
    mean_dist = sum(distances) / len(distances)
    variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
    
    # Penalize high variance (non-uniform distribution)
    symmetry_penalty = -variance * 0.1
    
    return fitness + symmetry_penalty

def crossover_operator(parent1, parent2):
    """Custom crossover operator that preserves hexagon properties."""
    # Create offspring by mixing positions and angles
    offspring = []
    
    # Use uniform crossover for positions and angles
    for i in range(len(parent1)):
        # 50% chance to take from parent1, 50% from parent2
        if random.random() < 0.5:
            offspring.append(parent1[i].copy())
        else:
            offspring.append(parent2[i].copy())
    
    # Apply small mutations to prevent premature convergence
    for i in range(len(offspring)):
        if random.random() < 0.1:  # 10% mutation rate
            # Mutate position slightly
            offspring[i][0] += random.uniform(-0.1, 0.1)
            offspring[i][1] += random.uniform(-0.1, 0.1)
            # Mutate angle
            offspring[i][2] += random.uniform(-5, 5)
            offspring[i][2] = offspring[i][2] % 360
    
    return offspring

def mutate_individual(individual):
    """Apply mutation to an individual."""
    mutated = [list(item) for item in individual]
    
    # Randomly select which hexagon to mutate
    hex_idx = random.randint(0, 11)
    
    # Apply small random perturbations
    mutated[hex_idx][0] += random.uniform(-0.05, 0.05)  # x position
    mutated[hex_idx][1] += random.uniform(-0.05, 0.05)  # y position
    mutated[hex_idx][2] += random.uniform(-3, 3)       # angle
    
    # Keep angle within bounds
    mutated[hex_idx][2] = mutated[hex_idx][2] % 360
    
    return mutated

def evolutionary_hexagon_packing(max_time=50):
    """Evolutionary algorithm approach to hexagon packing."""
    start_time = time.time()
    
    # Generate multiple starting configurations
    initial_configs = create_symmetric_configurations()
    
    # Initialize population with diverse configurations
    population = []
    for config in initial_configs:
        population.append(config)
    
    # Add some random variations to increase diversity
    for _ in range(10):
        # Create random variation of one of the initial configurations
        base_config = random.choice(initial_configs)
        variant = []
        for x, y, angle in base_config:
            # Add small random noise
            new_x = x + random.uniform(-0.2, 0.2)
            new_y = y + random.uniform(-0.2, 0.2)
            new_angle = (angle + random.uniform(-10, 10)) % 360
            variant.append([new_x, new_y, new_angle])
        population.append(variant)
    
    best_solution = None
    best_fitness = float('-inf')
    
    generation = 0
    max_generations = 1000
    
    while (time.time() - start_time) < max_time and generation < max_generations:
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            score = fitness_function(individual)
            fitness_scores.append(score)
            
            if score > best_fitness:
                best_fitness = score
                best_solution = [list(item) for item in individual]
        
        # Selection: keep top 50% of population
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        selected_indices = sorted_indices[:len(population)//2]
        selected_population = [population[i] for i in selected_indices]
        
        # Create new population through crossover and mutation
        new_population = selected_population[:]
        
        # Generate offspring through crossover
        while len(new_population) < len(population):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            offspring = crossover_operator(parent1, parent2)
            
            # Apply mutation with higher probability for younger generations
            if random.random() < 0.3:
                offspring = mutate_individual(offspring)
            
            new_population.append(offspring)
        
        population = new_population[:len(population)]
        generation += 1
    
    # Final refinement using local search on best solution
    if best_solution is not None:
        # Perform fine-tuning on the best solution
        refined_solution = best_solution.copy()
        
        # Simple local search: try small adjustments to each hexagon
        for _ in range(1000):
            if (time.time() - start_time) >= max_time:
                break
                
            # Try small random adjustments
            test_solution = [list(item) for item in refined_solution]
            hex_idx = random.randint(0, 11)
            
            # Small perturbation
            test_solution[hex_idx][0] += random.uniform(-0.02, 0.02)
            test_solution[hex_idx][1] += random.uniform(-0.02, 0.02)
            test_solution[hex_idx][2] += random.uniform(-1, 1)
            test_solution[hex_idx][2] = test_solution[hex_idx][2] % 360
            
            # Check if this improves fitness
            new_fitness = fitness_function(test_solution)
            if new_fitness > fitness_function(refined_solution):
                refined_solution = test_solution
        
        best_solution = refined_solution
    
    return best_solution, best_fitness

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary algorithm with symmetry-aware operators.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Run evolutionary optimization
    best_positions, final_score = evolutionary_hexagon_packing(max_time=55)
    
    # Calculate final outer hexagon size
    outer_side_length = calculate_outer_hexagon_side_length(best_positions)
    
    # Create outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Return the optimized configuration
    return np.array(best_positions), outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
