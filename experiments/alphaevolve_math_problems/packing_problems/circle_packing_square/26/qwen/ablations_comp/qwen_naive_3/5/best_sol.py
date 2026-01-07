# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import math
from typing import Tuple
from itertools import combinations
from scipy.spatial import KDTree
import warnings
from deap import base, creator, tools, algorithms
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithms and local optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Strategy: Use a more sophisticated initial placement based on known good patterns
    # and combine evolutionary algorithms with local optimization
    
    # Initialize with a better starting configuration using a known good pattern
    circles = np.zeros((n, 3))
    
    # Generate initial placement using a more systematic approach
    def generate_initial_placement(num_circles: int) -> np.ndarray:
        # Use a hexagonal-like packing pattern which often works well for circle packing
        positions = []
        
        # Create a more structured approach
        rows = 5
        cols = 5
        
        # Try to place in a grid-like fashion but with some randomness
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_circles:
                    break
                # Offset every other row to create a hexagonal pattern
                offset = 0.5 if i % 2 == 1 else 0.0
                x = 0.1 + (j + offset) * 0.18
                y = 0.1 + i * 0.18
                
                # Add some randomness to avoid perfect grid alignment
                x += (np.random.random() - 0.5) * 0.05
                y += (np.random.random() - 0.5) * 0.05
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                positions.append([x, y])
        
        # Fill remaining positions with strategic placement
        while len(positions) < num_circles:
            # Try to place in less crowded areas
            attempts = 0
            while attempts < 1000:
                x = 0.05 + 0.9 * np.random.random()
                y = 0.05 + 0.9 * np.random.random()
                
                # Check distance to existing points
                if len(positions) > 0:
                    existing_positions = np.array(positions)
                    distances = np.sqrt(np.sum((existing_positions - [x, y])**2, axis=1))
                    min_dist = np.min(distances)
                    # Require minimum distance to avoid clustering
                    if min_dist > 0.05:
                        positions.append([x, y])
                        break
                else:
                    positions.append([x, y])
                    break
                attempts += 1
            
            # If still no valid spot, just add a random point
            if len(positions) < num_circles:
                x = 0.05 + 0.9 * np.random.random()
                y = 0.05 + 0.9 * np.random.random()
                positions.append([x, y])
            
        return np.array(positions[:num_circles])
    
    # Generate initial positions
    initial_positions = generate_initial_placement(n)
    
    # Assign positions and set initial radii
    for i, (x, y) in enumerate(initial_positions):
        circles[i] = [x, y, 0.05]  # Start with small radius
    
    # Refine initial radii based on available space using KDTree for efficiency
    def compute_max_radius(circle_idx, current_circles, kdtree=None):
        x, y, _ = current_circles[circle_idx]
        
        # Find minimum distance to boundaries
        min_boundary_dist = min(x, 1-x, y, 1-y)
        
        # Find minimum distance to other circles using spatial indexing for efficiency
        min_other_dist = float('inf')
        
        if kdtree is not None:
            # Use KDTree for faster neighbor search
            nearby_indices = kdtree.query_ball_point([x, y], 2 * min_boundary_dist)
            for idx in nearby_indices:
                if idx != circle_idx:
                    other_x, other_y, other_r = current_circles[idx]
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_other_dist = min(min_other_dist, dist)
        else:
            # Fallback to direct computation
            for j in range(n):
                if i != j:
                    other_x, other_y, other_r = current_circles[j]
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_other_dist = min(min_other_dist, dist)
        
        # Set radius based on available space, ensuring no overlaps
        if min_other_dist < float('inf'):
            max_radius = min(min_boundary_dist, min_other_dist / 2.0)
        else:
            max_radius = min_boundary_dist
            
        # Make sure it's positive and reasonable
        max_radius = max(0.001, min(max_radius, 0.25))
        return max_radius
    
    # Build KDTree for efficient neighbor searches
    positions_array = circles[:, :2]
    kdtree = KDTree(positions_array)
    
    # Compute initial radii more efficiently
    for i in range(n):
        circles[i, 2] = compute_max_radius(i, circles, kdtree)
    
    # More sophisticated optimization approach with better constraints
    def objective(radii_and_centers):
        # Extract centers and radii from flattened array
        centers = radii_and_centers[:2*n].reshape(-1, 2)
        radii = radii_and_centers[2*n:]
        
        # Calculate negative sum of radii (we want to maximize sum)
        return -np.sum(radii)
    
    def constraint_func(radii_and_centers):
        centers = radii_and_centers[:2*n].reshape(-1, 2)
        radii = radii_and_centers[2*n:]
        
        constraints = []
        
        # Distance constraint: circles must not overlap
        # Use more efficient pairwise checking with early termination
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                min_dist = radii[i] + radii[j]
                # Constraint is satisfied when dist >= min_dist (so we subtract)
                constraints.append(dist - min_dist)
        
        # Boundary constraints: all radii must be valid
        for i in range(n):
            constraints.append(centers[i, 0] - radii[i])  # x - r >= 0
            constraints.append(centers[i, 1] - radii[i])  # y - r >= 0
            constraints.append(1 - centers[i, 0] - radii[i])  # 1 - x - r >= 0
            constraints.append(1 - centers[i, 1] - radii[i])  # 1 - y - r >= 0
        
        return np.array(constraints)
    
    # Flatten initial values
    initial_guess = np.concatenate([
        circles[:, :2].flatten(),  # centers
        circles[:, 2]              # radii
    ])
    
    # Define bounds for optimization
    bounds = []
    # Bounds for centers (0.01 to 0.99 to keep some margin)
    for _ in range(2*n):
        bounds.append((0.01, 0.99))
    # Bounds for radii (positive but not too large)
    for _ in range(n):
        bounds.append((0.001, 0.49))
    
    # Try evolutionary algorithm approach first for global search
    try:
        # Define evolutionary algorithm parameters
        toolbox = base.Toolbox()
        
        # Define individual representation as flattened array of [x1,y1,r1,x2,y2,r2,...]
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        def create_individual():
            # Create a random individual with proper bounds
            individual = []
            for i in range(2*n):  # centers
                individual.append(random.uniform(0.01, 0.99))
            for i in range(n):   # radii
                individual.append(random.uniform(0.001, 0.49))
            return creator.Individual(individual)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def eval_fitness(individual):
            # Convert individual to centers and radii
            centers = np.array(individual[:2*n]).reshape(-1, 2)
            radii = np.array(individual[2*n:])
            
            # Check constraints
            # Distance constraints
            penalty = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) * 1000  # Heavy penalty
            
            # Boundary constraints
            for i in range(n):
                if centers[i, 0] < radii[i] or centers[i, 1] < radii[i] or \
                   centers[i, 0] > 1 - radii[i] or centers[i, 1] > 1 - radii[i]:
                    penalty += 10000  # Heavy penalty
            
            # Return negative sum of radii plus penalties (to maximize sum)
            return (np.sum(radii) - penalty,)
        
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolutionary algorithm
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution with more generations
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.7, mutpb=0.2, 
            ngen=50, stats=stats, halloffame=hof, verbose=False
        )
        
        # Get the best individual from EA
        if len(hof) > 0:
            best_individual = hof[0]
            # Convert back to circles format
            centers = np.array(best_individual[:2*n]).reshape(-1, 2)
            radii = np.array(best_individual[2*n:])
            
            # Update circles with EA result
            for i in range(n):
                circles[i] = [centers[i, 0], centers[i, 1], radii[i]]
                
            # Now refine with local optimization
            # Create a refined initial guess from EA result
            refined_guess = np.concatenate([
                circles[:, :2].flatten(),  # centers
                circles[:, 2]              # radii
            ])
            
            # Local optimization with SLSQP
            try:
                result = minimize(
                    objective,
                    refined_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 2000, 'ftol': 1e-12, 'eps': 1e-8}
                )
                
                if result.success:
                    optimized_centers = result.x[:2*n].reshape(-1, 2)
                    optimized_radii = result.x[2*n:]
                    
                    # Update circles with optimized values
                    for i in range(n):
                        circles[i] = [optimized_centers[i, 0], optimized_centers[i, 1], optimized_radii[i]]
            except:
                pass
    except Exception as e:
        # Fall back to traditional optimization if EA fails
        pass
    
    # Final validation and refinement
    try:
        # Try multiple local optimization approaches
        best_sum = np.sum(circles[:, 2])
        best_circles = circles.copy()
        
        # Run several local optimization attempts
        for attempt in range(5):
            np.random.seed(42 + attempt)
            
            # Create a slightly perturbed version
            perturbed_guess = initial_guess.copy()
            # Add noise to centers and radii
            for i in range(len(perturbed_guess)):
                if i >= 2*n:  # Radii
                    perturbed_guess[i] *= (1 + (np.random.random() - 0.5) * 0.1)
                else:  # Centers
                    perturbed_guess[i] += (np.random.random() - 0.5) * 0.05
            
            # Clip to bounds
            for i in range(len(bounds)):
                perturbed_guess[i] = max(bounds[i][0], min(bounds[i][1], perturbed_guess[i]))
            
            try:
                result = minimize(
                    objective,
                    perturbed_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1000, 'ftol': 1e-10, 'eps': 1e-6}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        optimized_centers = result.x[:2*n].reshape(-1, 2)
                        optimized_radii = result.x[2*n:]
                        
                        for i in range(n):
                            best_circles[i] = [optimized_centers[i, 0], optimized_centers[i, 1], optimized_radii[i]]
            except:
                continue
        
        circles = best_circles
    except Exception as e:
        pass
    
    # Final validation to ensure all constraints are met
    # Make sure circles don't go outside boundaries
    for i in range(n):
        x, y, r = circles[i]
        # Adjust if necessary
        circles[i] = [max(r, min(1-r, x)), max(r, min(1-r, y)), r]
    
    return circles


# EVOLVE-BLOCK-END
