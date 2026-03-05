# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from typing import Tuple, List
import time

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon centered at center with given rotation."""
    angle = rotation * np.pi / 180
    # Vertices of unit hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_vertices.append((x + center[0], y + center[1]))
    return Polygon(hex_vertices)

def get_outer_hexagon_vertices(side_length, center=(0,0), rotation=0):
    """Get vertices of outer hexagon."""
    angle = rotation * np.pi / 180
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = side_length * np.cos(theta)
        y = side_length * np.sin(theta)
        vertices.append((x + center[0], y + center[1]))
    return vertices

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def calculate_total_packing_area(inner_hexagons):
    """Calculate total area of inner hexagons."""
    total_area = 0
    for hex in inner_hexagons:
        total_area += hex.area
    return total_area

def evaluate_solution(inner_positions, inner_rotations, outer_side_length):
    """Evaluate solution quality."""
    # Create outer hexagon
    outer_hex = Polygon(get_outer_hexagon_vertices(outer_side_length))
    
    # Create inner hexagons
    inner_hexagons = []
    for pos, rot in zip(inner_positions, inner_rotations):
        hex = create_unit_hexagon(pos, rot)
        inner_hexagons.append(hex)
    
    # Check containment and overlap
    valid = True
    total_area = 0
    
    for i, hex in enumerate(inner_hexagons):
        # Check containment
        if not check_containment(hex, outer_hex):
            valid = False
            break
            
        # Check overlaps with others
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(hex, inner_hexagons[j]):
                valid = False
                break
                
        if not valid:
            break
            
        total_area += hex.area
    
    if valid:
        # Calculate how well we're utilizing the space
        outer_area = outer_hex.area
        efficiency = total_area / outer_area if outer_area > 0 else 0
        return efficiency, True
    else:
        return 0, False

def generate_initial_population(pop_size, num_hexagons=11):
    """Generate initial population for GA."""
    population = []
    for _ in range(pop_size):
        # Random positions and rotations
        positions = []
        rotations = []
        for _ in range(num_hexagons):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            rot = random.uniform(0, 360)
            positions.append([x, y])
            rotations.append(rot)
        population.append((positions, rotations))
    return population

def mutate_individual(positions, rotations, mutation_rate=0.1):
    """Mutate individual solution."""
    new_positions = []
    new_rotations = []
    
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        if random.random() < mutation_rate:
            # Mutate position
            new_x = pos[0] + random.gauss(0, 0.5)
            new_y = pos[1] + random.gauss(0, 0.5)
            new_positions.append([new_x, new_y])
        else:
            new_positions.append(pos.copy())
            
        if random.random() < mutation_rate:
            # Mutate rotation
            new_rot = (rot + random.gauss(0, 10)) % 360
            new_rotations.append(new_rot)
        else:
            new_rotations.append(rot)
            
    return new_positions, new_rotations

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    p1_pos, p1_rot = parent1
    p2_pos, p2_rot = parent2
    
    # Simple average crossover
    child_pos = []
    child_rot = []
    
    for i in range(len(p1_pos)):
        # Blend positions
        child_pos.append([(p1_pos[i][0] + p2_pos[i][0]) / 2, (p1_pos[i][1] + p2_pos[i][1]) / 2])
        # Blend rotations
        child_rot.append((p1_rot[i] + p2_rot[i]) / 2)
        
    return child_pos, child_rot

def optimize_hexagon_packing():
    """Use genetic algorithm to find optimal packing."""
    # Parameters
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    # Initialize population
    population = generate_initial_population(pop_size)
    
    best_fitness = 0
    best_solution = None
    best_side_length = 100
    
    # Start timer
    start_time = time.time()
    
    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for positions, rotations in population:
            # Try different outer hexagon sizes to find minimal one
            side_length = 8.0  # Start with reasonable size
            found_valid = False
            
            # Binary search for minimum valid side length
            min_side = 2.0
            max_side = 15.0
            
            while max_side - min_side > 0.01 and time.time() - start_time < 55:  # Leave 5 seconds for final refinement
                test_side = (min_side + max_side) / 2
                fitness, valid = evaluate_solution(positions, rotations, test_side)
                
                if valid:
                    max_side = test_side
                    found_valid = True
                else:
                    min_side = test_side
                    
            if found_valid:
                final_side = (min_side + max_side) / 2
                if final_side < best_side_length:
                    best_side_length = final_side
                    best_solution = (positions.copy(), rotations.copy())
                    best_fitness = fitness
                    
        # Print progress
        if gen % 20 == 0:
            print(f"Generation {gen}: Best side length = {best_side_length:.4f}")
            
        # Selection and reproduction
        # Sort population by fitness
        sorted_pop = sorted(zip(population, [evaluate_solution(pos, rot, 8)[0] for pos, rot in population]), 
                           key=lambda x: x[1], reverse=True)
        
        # Keep elite
        elite = [ind[0] for ind in sorted_pop[:elite_size]]
        
        # Generate new population
        new_population = elite[:]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = random.randint(0, elite_size - 1)
            parent2_idx = random.randint(0, elite_size - 1)
            
            parent1 = elite[parent1_idx]
            parent2 = elite[parent2_idx]
            
            # Crossover
            child_pos, child_rot = crossover(parent1, parent2)
            
            # Mutation
            child_pos, child_rot = mutate_individual(child_pos, child_rot, mutation_rate)
            
            new_population.append((child_pos, child_rot))
            
        population = new_population
        
        # Early termination check
        if time.time() - start_time > 55:
            break
    
    # Final refinement of best solution
    if best_solution:
        final_positions, final_rotations = best_solution
        side_length = best_side_length
        
        # Refine by fine-tuning
        refined_positions = []
        refined_rotations = []
        
        # Simple local search around best solution
        for i, (pos, rot) in enumerate(zip(final_positions, final_rotations)):
            # Small random perturbations to improve solution
            new_pos = [pos[0] + random.uniform(-0.1, 0.1), pos[1] + random.uniform(-0.1, 0.1)]
            new_rot = rot + random.uniform(-5, 5)
            refined_positions.append(new_pos)
            refined_rotations.append(new_rot % 360)
            
        # Final validation and optimization
        best_final_side = side_length
        final_positions = refined_positions
        final_rotations = refined_rotations
        
        # Verify and get final result
        _, valid = evaluate_solution(final_positions, final_rotations, best_final_side)
        if not valid:
            # If invalid, try to find a better configuration
            best_final_side = 10.0  # Default large value
            for test_side in [5.0, 6.0, 7.0, 8.0, 9.0]:
                _, valid = evaluate_solution(final_positions, final_rotations, test_side)
                if valid:
                    best_final_side = test_side
                    break
        
        return final_positions, final_rotations, best_final_side
    
    # Fallback to simple optimized arrangement
    return simple_optimized_arrangement()

def simple_optimized_arrangement():
    """Return a more optimized simple arrangement."""
    # Hexagonal packing pattern - more efficient than grid
    positions = [
        [0, 0],           # center
        [0, 2.17],        # top
        [0, -2.17],       # bottom  
        [1.87, 1.08],     # top-right
        [-1.87, 1.08],    # top-left
        [1.87, -1.08],    # bottom-right
        [-1.87, -1.08],   # bottom-left
        [3.75, 0],        # far right
        [-3.75, 0],       # far left
        [1.87, 3.25],     # top far
        [-1.87, 3.25],    # top far left
    ]
    
    rotations = [0] * 11  # All horizontal
    
    return positions, rotations, 6.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use genetic algorithm optimization
    positions, rotations, side_length = optimize_hexagon_packing()
    
    # Convert to required format
    inner_hex_data = np.array([[pos[0], pos[1], rot] for pos, rot in zip(positions, rotations)])
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, side_length


# EVOLVE-BLOCK-END
