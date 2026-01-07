# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
import random
from typing import Tuple
import time
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach with differential evolution and local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Define bounds for optimization: [x, y, r] for each circle
    # x, y in [r, 1-r], r in [0, 0.5] (reasonable upper bound)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    # Fitness function that evaluates a candidate solution
    def evaluate(individual):
        # Convert individual to positions and radii
        positions = np.array(individual[::3]).reshape(n, 1)  # x coordinates
        positions = np.hstack([positions, np.array(individual[1::3]).reshape(n, 1)])  # y coordinates
        radii = np.array(individual[2::3])  # radii
        
        # Check containment constraints
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return (-1000000,)  # Invalid configuration
        
        # Check overlap constraints using spatial indexing for efficiency
        tree = cKDTree(positions)
        penalty = 0
        
        for i in range(n):
            # Query nearby points (within 2*(max_radius))
            neighbors = tree.query_ball_point(positions[i], 2 * max(radii), p=2)
            for j in neighbors:
                if i != j:
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist_sq = dx*dx + dy*dy
                    r1, r2 = radii[i], radii[j]
                    
                    # Check if circles overlap
                    if dist_sq < (r1 + r2)**2:
                        # Penalty based on how much they overlap
                        overlap = (r1 + r2)**2 - dist_sq
                        penalty += overlap * 1000000
        
        # Objective: maximize sum of radii (negative because DEAP minimizes)
        total_radius = np.sum(radii)
        return (total_radius - penalty,)
    
    # Improved local optimization with better convergence
    def local_optimization(positions, radii):
        """Improve configuration using gradient-based approach with adaptive steps"""
        # Create a copy to work with
        pos = positions.copy()
        rad = radii.copy()
        
        # Parameters for optimization
        max_iterations = 500
        learning_rate = 0.005
        repulsion_strength = 0.001
        boundary_strength = 10.0
        
        for iteration in range(max_iterations):
            # Calculate forces on each circle
            forces = np.zeros_like(pos)
            
            # Repulsion forces from other circles
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        if dist > 0 and dist < (rad[i] + rad[j]) * 2:
                            # Repulsion force (inverse square law)
                            force_magnitude = repulsion_strength / (dist * dist + 1e-8)
                            forces[i][0] += force_magnitude * dx / dist
                            forces[i][1] += force_magnitude * dy / dist
            
            # Boundary forces - stronger near boundaries
            for i in range(n):
                # Force away from boundaries (stronger near edges)
                boundary_force_x = boundary_strength * max(0, rad[i] - pos[i][0]) + \
                                  boundary_strength * max(0, pos[i][0] - (1 - rad[i]))
                boundary_force_y = boundary_strength * max(0, rad[i] - pos[i][1]) + \
                                  boundary_strength * max(0, pos[i][1] - (1 - rad[i]))
                
                forces[i][0] += (boundary_force_x if pos[i][0] < rad[i] else -boundary_force_x)
                forces[i][1] += (boundary_force_y if pos[i][1] < rad[i] else -boundary_force_y)
            
            # Update positions
            for i in range(n):
                pos[i][0] += learning_rate * forces[i][0]
                pos[i][1] += learning_rate * forces[i][1]
                
                # Keep within bounds
                pos[i][0] = max(rad[i], min(1 - rad[i], pos[i][0]))
                pos[i][1] = max(rad[i], min(1 - rad[i], pos[i][1]))
            
            # Update radii to maximize total while maintaining constraints
            updated_radii = rad.copy()
            for i in range(n):
                # Calculate maximum radius allowed by boundaries
                max_radius_bound = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
                
                # Calculate minimum distance to neighbors
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dist = math.sqrt((pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                
                # Maximum radius is constrained by both boundaries and neighbors
                if min_dist < float('inf'):
                    max_radius_neighbors = min_dist / 2.0
                    max_radius = min(max_radius_bound, max_radius_neighbors)
                else:
                    max_radius = max_radius_bound
                
                # Increase radius if beneficial
                updated_radii[i] = min(max_radius, rad[i] * 1.05)
            
            # Check for convergence
            if np.allclose(updated_radii, rad, rtol=1e-5):
                break
                
            rad = updated_radii
        
        return pos, rad
    
    # Better initialization using a more sophisticated approach
    def generate_initial_population(size):
        population = []
        for _ in range(size):
            # Create initial configuration with some structure
            individual = []
            
            # Place circles in a grid-like pattern with small random perturbations
            rows = 6
            cols = 6
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)
            
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n:
                        break
                    x = (j + 1) * spacing_x + random.uniform(-0.02, 0.02)
                    y = (i + 1) * spacing_y + random.uniform(-0.02, 0.02)
                    r = random.uniform(0.02, 0.15)  # Initial radii between 0.02 and 0.15
                    
                    # Clip to valid ranges
                    x = max(r, min(1 - r, x))
                    y = max(r, min(1 - r, y))
                    r = max(0.001, min(0.5, r))
                    
                    individual.extend([x, y, r])
                    idx += 1
                    
            population.append(individual)
        return population
    
    # Set up DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: random.uniform(bounds[0][0], bounds[0][1]), n*3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolutionary algorithm
    pop = generate_initial_population(50)
    
    # Run with different parameters to get better results
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                          ngen=50, stats=stats, halloffame=hof, 
                                          verbose=False)
    except Exception:
        # Fallback to basic approach if evolution fails
        pass
    
    # Get best solution from evolution
    best_individual = None
    if hof:
        best_individual = hof[0]
    
    # If evolution failed or produced poor results, use local optimization approach
    if best_individual is None:
        # Try a more focused approach with better initialization
        best_sum = 0
        best_circles = None
        
        # Multiple random restarts
        for restart in range(10):
            # Generate better initial configuration
            positions = np.zeros((n, 2))
            radii = np.zeros(n)
            
            # Create a more balanced distribution
            for i in range(n):
                # Distribute more evenly
                row = i // 6
                col = i % 6
                positions[i][0] = (col + 1) * (1.0 / 7.0) + random.uniform(-0.01, 0.01)
                positions[i][1] = (row + 1) * (1.0 / 7.0) + random.uniform(-0.01, 0.01)
                radii[i] = random.uniform(0.05, 0.15)
            
            # Apply local optimization
            optimized_pos, optimized_rad = local_optimization(positions, radii)
            
            # Validate and check quality
            valid = True
            for i in range(n):
                x, y = optimized_pos[i]
                r = optimized_rad[i]
                if r > x or r > (1-x) or r > y or r > (1-y):
                    valid = False
                    break
            
            if valid:
                total_radius = np.sum(optimized_rad)
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_circles = np.column_stack([optimized_pos, optimized_rad])
        
        if best_circles is not None:
            return best_circles
    
    # If we still don't have a good solution, create a fallback
    if best_individual is None:
        # Use a structured approach with known good patterns
        circles = np.zeros((n, 3))
        row_count = 6
        col_count = 6
        spacing_x = 1.0 / (col_count + 1)
        spacing_y = 1.0 / (row_count + 1)
        
        idx = 0
        for i in range(row_count):
            for j in range(col_count):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x + random.uniform(-0.01, 0.01)
                y = (i + 1) * spacing_y + random.uniform(-0.01, 0.01)
                circles[idx] = [x, y, 0.05]
                idx += 1
        
        # Local optimization on this configuration
        positions = circles[:, :2]
        radii = circles[:, 2]
        optimized_pos, optimized_rad = local_optimization(positions, radii)
        return np.column_stack([optimized_pos, optimized_rad])
    
    # Convert best individual to circles array
    positions = np.array(best_individual[::3]).reshape(n, 1)
    positions = np.hstack([positions, np.array(best_individual[1::3]).reshape(n, 1)])
    radii = np.array(best_individual[2::3])
    
    # Final local optimization for refinement
    final_pos, final_rad = local_optimization(positions, radii)
    
    # Validation
    valid = True
    for i in range(n):
        x, y = final_pos[i]
        r = final_rad[i]
        if r > x or r > (1-x) or r > y or r > (1-y):
            valid = False
            break
    
    if not valid:
        # Fallback to a simpler configuration
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        radius = spacing / 2.0
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (j + 1) * spacing
                y = (i + 1) * spacing
                circles[idx] = [x, y, radius]
                idx += 1
        
        return circles
    
    return np.column_stack([final_pos, final_rad])


# EVOLVE-BLOCK-END
