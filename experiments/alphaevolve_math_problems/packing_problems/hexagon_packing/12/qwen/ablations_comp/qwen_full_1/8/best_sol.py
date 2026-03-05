# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import math
import time
from itertools import combinations

def get_hexagon_vertices(center_x, center_y, side_length=1, rotation=0):
    """Get vertices of a regular hexagon given center, side length, and rotation."""
    vertices = []
    rotation_rad = math.radians(rotation)
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def get_hexagon_polygon(center_x, center_y, angle_deg, side_length=1):
    """Get Shapely polygon representation of hexagon"""
    vertices = get_hexagon_vertices(center_x, center_y, side_length, angle_deg)
    return Polygon(vertices)


def check_containment_all_vertices(hexagon_poly, outer_hex_poly):
    """Check if ALL vertices of hexagon are fully contained within outer hexagon"""
    # Check all 6 vertices of the inner hexagon
    vertices = list(hexagon_poly.exterior.coords)
    for x, y in vertices[:-1]:  # Exclude last point which duplicates first
        point = Point(x, y)
        if not outer_hex_poly.contains(point):
            return False
    return True


def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)


def calculate_outer_hexagon_side_length_from_positions(positions, rotations, outer_center=(0,0)):
    """Calculate the minimum outer hexagon side length required"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(positions)):
        pos = positions[i]
        rot = rotations[i]
        vertices = get_hexagon_vertices(pos[0], pos[1], 1.0, rot)
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return 1.0
    
    # Calculate distance from outer center to each vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - outer_center[0])**2 + (y - outer_center[1])**2)
        max_dist = max(max_dist, dist)
    
    # For a regular hexagon, the relationship between circumradius and side length:
    # Circumradius = side_length * sqrt(3) / 2
    # So side_length = circumradius * 2 / sqrt(3)
    side_length = max_dist * 2 / math.sqrt(3)
    
    return side_length


def evaluate_packing_with_constraints(inner_positions, inner_rotations, outer_side_length):
    """
    Evaluate if a configuration is valid and return the inverse side length.
    Returns (inverse_side_length, valid) tuple.
    """
    # Create hexagon polygons for all inner hexagons
    inner_hexagons = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        rot = inner_rotations[i]
        hex_poly = get_hexagon_polygon(pos[0], pos[1], rot)
        inner_hexagons.append(hex_poly)
    
    # Check containment - ALL vertices must be inside outer hexagon
    outer_center = (0, 0)
    outer_vertices = get_hexagon_vertices(outer_center[0], outer_center[1], outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    for hex_poly in inner_hexagons:
        if not check_containment_all_vertices(hex_poly, outer_polygon):
            return None, False  # Not fully contained
    
    # Check overlaps - no pair should overlap
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if check_overlap(inner_hexagons[i], inner_hexagons[j]):
            return None, False  # Overlapping
    
    return 1.0 / outer_side_length, True


def create_symmetric_hexagon_arrangement():
    """
    Create a highly symmetric arrangement of 12 hexagons based on mathematical principles
    This approach uses a known optimal configuration pattern with rotational symmetry
    """
    # Based on known dense packings, we'll construct a pattern that's more likely to be optimal
    # Using 3 concentric rings with strategic positioning
    
    # Ring 1: Center hexagon
    positions = [(0.0, 0.0)]
    
    # Ring 2: 6 hexagons around the center (at distance 2)
    ring2_angles = [i * 60 for i in range(6)]
    for angle in ring2_angles:
        rad = math.radians(angle)
        x = 2.0 * math.cos(rad)
        y = 2.0 * math.sin(rad)
        positions.append((x, y))
    
    # Ring 3: 5 hexagons in a ring (at distance 3.464 approximately)
    ring3_angles = [i * 72 for i in range(5)]  # 5-fold symmetry
    for angle in ring3_angles:
        rad = math.radians(angle)
        x = 3.464 * math.cos(rad)
        y = 3.464 * math.sin(rad)
        positions.append((x, y))
    
    # Add one more to complete 12 hexagons
    positions.append((0.0, 3.464))
    
    # Initial rotations - some may be rotated for better packing
    rotations = [0.0] * 12
    
    return positions, rotations


def generate_valid_configurations():
    """
    Generate multiple valid configurations to explore the solution space
    Uses combinatorial approach with symmetry constraints
    """
    # Generate a few candidate configurations
    configs = []
    
    # Configuration 1: Hexagonal close packing pattern with slight variations
    pos1 = [
        (0, 0), (0, 2), (1.732, 1), (1.732, -1), (0, -2), (-1.732, -1),
        (-1.732, 1), (3.464, 0), (1.732, 3), (-1.732, 3), (-3.464, 0), (-1.732, -3)
    ]
    rot1 = [0] * 12
    configs.append((pos1, rot1))
    
    # Configuration 2: Alternative arrangement with different spacing
    pos2 = [
        (0, 0), (0, 2.2), (1.8, 1.1), (1.8, -1.1), (0, -2.2), (-1.8, -1.1),
        (-1.8, 1.1), (3.6, 0), (1.8, 3.3), (-1.8, 3.3), (-3.6, 0), (-1.8, -3.3)
    ]
    rot2 = [0] * 12
    configs.append((pos2, rot2))
    
    # Configuration 3: More compact arrangement
    pos3 = [
        (0, 0), (0, 1.8), (1.5, 0.9), (1.5, -0.9), (0, -1.8), (-1.5, -0.9),
        (-1.5, 0.9), (3.0, 0), (1.5, 2.7), (-1.5, 2.7), (-3.0, 0), (-1.5, -2.7)
    ]
    rot3 = [0] * 12
    configs.append((pos3, rot3))
    
    return configs


def constraint_satisfaction_approach():
    """
    Use a constraint satisfaction approach to find valid configurations
    This approach systematically explores geometric constraints
    """
    # Define geometric constraints
    min_distance = 2.0  # Minimum distance between centers for non-overlap
    max_radius = 4.5   # Reasonable upper bound for outer hexagon radius
    
    # Try multiple configurations using systematic sampling
    best_result = None
    best_inv_side_length = 0
    
    # Generate configurations with different symmetries
    configs = generate_valid_configurations()
    
    for i, (positions, rotations) in enumerate(configs):
        # Try different outer hexagon sizes
        for outer_size in np.linspace(3.8, 4.2, 10):
            inv_side_length, valid = evaluate_packing_with_constraints(positions, rotations, outer_size)
            if valid and inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_result = (positions, rotations, outer_size)
    
    return best_result, best_inv_side_length


def evolutionary_hexagon_packing():
    """
    Evolutionary approach to hexagon packing optimization
    This implements a novel genetic algorithm tailored for hexagon arrangements
    """
    # Population of hexagon arrangements
    population_size = 20
    generations = 30
    
    # Initialize population with random valid configurations
    population = []
    
    # Create diverse initial configurations
    for i in range(population_size):
        # Randomly perturb positions and rotations
        positions = []
        rotations = []
        
        # Generate 12 hexagon positions with some clustering
        for j in range(12):
            # Perturb around a base arrangement
            base_x = (j % 4 - 1.5) * 2.0
            base_y = (j // 4 - 1.0) * 2.0
            x = base_x + np.random.normal(0, 0.5)
            y = base_y + np.random.normal(0, 0.5)
            positions.append((x, y))
            rotations.append(np.random.uniform(-180, 180))
        
        population.append((positions, rotations))
    
    # Evolution loop
    best_individual = None
    best_fitness = 0
    
    for generation in range(generations):
        fitness_scores = []
        
        # Evaluate fitness for each individual
        for positions, rotations in population:
            # Try to find optimal outer size for this configuration
            # Binary search for the minimal outer size
            low, high = 3.8, 4.5
            best_outer_size = 4.5
            
            for _ in range(10):  # Binary search iterations
                mid = (low + high) / 2
                inv_side_length, valid = evaluate_packing_with_constraints(positions, rotations, mid)
                if valid and inv_side_length > best_fitness:
                    best_fitness = inv_side_length
                    best_outer_size = mid
                    high = mid
                else:
                    low = mid
            
            # Use the best valid size found
            inv_side_length, valid = evaluate_packing_with_constraints(positions, rotations, best_outer_size)
            fitness = inv_side_length if valid else 0
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = (positions, rotations, best_outer_size)
        
        # Selection and reproduction (simple tournament selection)
        selected_indices = np.argsort(fitness_scores)[-population_size//2:]
        selected_population = [population[i] for i in selected_indices]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        while len(new_population) < population_size:
            parent1 = selected_population[np.random.randint(0, len(selected_population))]
            parent2 = selected_population[np.random.randint(0, len(selected_population))]
            
            # Crossover: combine positions and rotations
            child_positions = []
            child_rotations = []
            
            for i in range(12):
                if np.random.random() < 0.5:
                    child_positions.append(parent1[0][i])
                    child_rotations.append(parent1[1][i])
                else:
                    child_positions.append(parent2[0][i])
                    child_rotations.append(parent2[1][i])
            
            # Mutation: small random perturbations
            for i in range(12):
                if np.random.random() < 0.1:  # 10% chance of mutation
                    child_positions[i] = (
                        child_positions[i][0] + np.random.normal(0, 0.1),
                        child_positions[i][1] + np.random.normal(0, 0.1)
                    )
                if np.random.random() < 0.1:
                    child_rotations[i] += np.random.normal(0, 10)
            
            new_population.append((child_positions, child_rotations))
        
        population = new_population[:population_size]
    
    return best_individual, best_fitness


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a combinatorial constraint satisfaction approach with evolutionary optimization.
    """
    
    # Start with a well-known symmetric configuration
    initial_positions, initial_rotations = create_symmetric_hexagon_arrangement()
    
    # Try constraint satisfaction approach first
    best_result, best_inv_side_length = constraint_satisfaction_approach()
    
    # Also try evolutionary approach
    evol_result, evol_fitness = evolutionary_hexagon_packing()
    
    # Choose the better of the two approaches
    if evol_fitness > best_inv_side_length:
        best_result = evol_result
        best_inv_side_length = evol_fitness
    
    # If no good result was found, fall back to initial configuration
    if best_result is None:
        # Use the symmetric configuration directly
        positions = initial_positions
        rotations = initial_rotations
        outer_radius = 4.0  # Conservative estimate
        
        # Try to find the actual minimum outer radius
        min_outer_radius = 3.8
        max_outer_radius = 4.5
        best_outer_radius = 4.0
        
        # Binary search for optimal radius
        for _ in range(15):
            test_radius = (min_outer_radius + max_outer_radius) / 2
            inv_side_length, valid = evaluate_packing_with_constraints(positions, rotations, test_radius)
            if valid and inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_outer_radius = test_radius
                max_outer_radius = test_radius
            else:
                min_outer_radius = test_radius
        
        best_result = (positions, rotations, best_outer_radius)
    
    # Final refinement using local optimization around the best solution
    positions, rotations, outer_radius = best_result
    
    # Use a more targeted optimization approach
    def objective(x):
        # Extract variables: [x1,y1,r1,...,x12,y12,r12,R]
        positions = [(x[3*i], x[3*i+1]) for i in range(12)]
        rotations = [x[3*i+2] for i in range(12)]
        outer_radius = x[-1]
        
        # Check validity and return negative inverse side length
        inv_side_length, valid = evaluate_packing_with_constraints(positions, rotations, outer_radius)
        if not valid:
            # Heavy penalty for invalid configurations
            return 1000000
        return -inv_side_length
    
    # Set up bounds
    bounds = []
    for i in range(12):
        bounds.extend([(-6, 6), (-6, 6)])  # x, y bounds
        bounds.append((-180, 180))  # rotation bounds
    bounds.append((3.8, 4.2))  # outer radius bounds
    
    # Start with current best solution
    x0 = []
    for i in range(12):
        x0.extend([positions[i][0], positions[i][1], rotations[i]])
    x0.append(outer_radius)
    
    # Local optimization to refine
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
        if result.success:
            refined_positions = [(result.x[3*i], result.x[3*i+1]) for i in range(12)]
            refined_rotations = [result.x[3*i+2] for i in range(12)]
            refined_outer_radius = result.x[-1]
            
            inv_side_length, valid = evaluate_packing_with_constraints(refined_positions, refined_rotations, refined_outer_radius)
            if valid and inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_result = (refined_positions, refined_rotations, refined_outer_radius)
    except:
        pass
    
    # Return final results
    positions, rotations, outer_radius = best_result
    inner_hex_data = np.array([
        [pos[0], pos[1], rot] for pos, rot in zip(positions, rotations)
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
