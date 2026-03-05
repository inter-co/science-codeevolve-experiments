# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import ConvexHull
import random
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings('ignore')
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced initialization and optimization techniques for improved results.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Better initialization strategies
    def initialize_regular_hexagon():
        """Initialize points in a regular hexagon pattern"""
        points = []
        # Place 16 points in a hexagonal pattern
        # 1st layer: 6 points around a circle
        # 2nd layer: 10 points in a hexagonal pattern
        # For simplicity, use a more uniform distribution
        
        # Use a circular arrangement with some randomness
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = 0.4 + 0.3 * np.random.random(n)  # Random radii to avoid perfect symmetry
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        return np.array(points)
    
    def initialize_grid_with_jitter():
        """Initialize points in a grid with jitter"""
        # Create a 4x4 grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25
                y = i * 0.25
                grid_points.append([x, y])
        
        points = np.array(grid_points[:n])
        # Add jitter to break symmetry
        jitter = np.random.normal(0, 0.015, (n, 2))
        points = points + jitter
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    def initialize_concentric_rings():
        """Initialize points in concentric rings"""
        points = []
        # 4 rings with different radii
        ring_radii = [0.15, 0.35, 0.55, 0.75]
        points_per_ring = [4, 4, 4, 4]  # Total 16 points
        
        for ring_idx, (radius, num_points) in enumerate(zip(ring_radii, points_per_ring)):
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            for angle in angles:
                # Add some randomness to avoid perfect patterns
                angle += np.random.normal(0, 0.1)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        return np.array(points)
    
    def initialize_poisson_disk():
        """Initialize points using a Poisson disk sampling approach"""
        # Simplified version - place points in a way that avoids clustering
        points = []
        # Use a simple grid with some randomness
        for i in range(4):
            for j in range(4):
                # Add some jitter to make it non-uniform
                x = j * 0.25 + np.random.normal(0, 0.02)
                y = i * 0.25 + np.random.normal(0, 0.02)
                points.append([x, y])
        
        points = np.array(points[:n])
        points = np.clip(points, 0, 1)
        return points
    
    def initialize_random_uniform():
        """Initialize points uniformly at random"""
        return np.random.uniform(0, 1, (n, 2))
    
    def initialize_fibonacci_sphere():
        """Initialize points using Fibonacci sphere algorithm for better distribution"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Map to 2D unit square
            points.append([(x + 1) / 2, (z + 1) / 2])
        
        return np.array(points)
    
    # Calculate min/max distance ratio
    def calculate_ratio(points):
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    # Enhanced objective function with better handling
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Apply boundary constraints by clipping
        points = np.clip(points, 0, 1)
        ratio = calculate_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    # Try multiple initialization strategies and use the best one
    initial_strategies = [
        initialize_regular_hexagon,
        initialize_grid_with_jitter,
        initialize_concentric_rings,
        initialize_poisson_disk,
        initialize_random_uniform,
        initialize_fibonacci_sphere
    ]
    
    best_initial_points = None
    best_ratio = 0
    
    for init_func in initial_strategies:
        try:
            points = init_func()
            ratio = calculate_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_initial_points = points.copy()
        except Exception as e:
            continue
    
    # If no good initialization found, fall back to grid with jitter
    if best_initial_points is None:
        best_initial_points = initialize_grid_with_jitter()
    
    # Multi-start optimization with different methods
    best_result_points = best_initial_points.copy()
    best_result_ratio = best_ratio
    
    # Use a hybrid optimization approach with multiple restarts
    # Strategy 1: Multiple restarts with different optimizers
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    # Strategy 2: Genetic Algorithm for global search
    def create_individual():
        return list(np.random.uniform(0, 1, n * 2))
    
    def evaluate(individual):
        points = np.array(individual).reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = calculate_ratio(points)
        return (-ratio,)  # Return negative since we want to maximize
    
    # Set up genetic algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA for global exploration
    try:
        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run GA for limited generations to explore global space
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                     ngen=30, stats=stats, halloffame=hof, verbose=False)
        
        if len(hof) > 0:
            ga_points = np.array(hof[0]).reshape(-1, 2)
            ga_points = np.clip(ga_points, 0, 1)
            ga_ratio = calculate_ratio(ga_points)
            if ga_ratio > best_result_ratio:
                best_result_points = ga_points
                best_result_ratio = ga_ratio
    except Exception as e:
        pass
    
    # Refine with local optimization from best GA result
    try:
        # Use scipy optimization with multiple restarts
        for restart in range(5):
            # Add small jitter to GA result
            jittered_points = best_result_points + np.random.normal(0, 0.005, (n, 2))
            jittered_points = np.clip(jittered_points, 0, 1)
            
            x0 = jittered_points.flatten()
            
            # Try different optimization methods
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        options={'maxiter': 200, 'ftol': 1e-8},
                        tol=1e-8
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        ratio = calculate_ratio(final_points)
                        
                        if ratio > best_result_ratio:
                            best_result_points = final_points
                            best_result_ratio = ratio
                except Exception as e:
                    continue
    except Exception as e:
        pass
    
    # Local search refinement with more targeted approach
    def local_refinement(initial_points, max_iter=200):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)
        
        for i in range(max_iter):
            # Try moving each point slightly
            for j in range(n):
                # Save current point
                original_point = current_points[j].copy()
                
                # Try a small random perturbation
                perturbation = np.random.normal(0, 0.002, 2)
                new_point = original_point + perturbation
                new_point = np.clip(new_point, 0, 1)
                
                # Temporarily update
                current_points[j] = new_point
                new_ratio = calculate_ratio(current_points)
                
                if new_ratio <= current_ratio:
                    # Revert if not better
                    current_points[j] = original_point
                else:
                    current_ratio = new_ratio
                    
        return current_points, current_ratio
    
    # Apply local refinement
    try:
        refined_points, refined_ratio = local_refinement(best_result_points, 100)
        if refined_ratio > best_result_ratio:
            best_result_points = refined_points
            best_result_ratio = refined_ratio
    except Exception as e:
        pass
    
    # Final simulated annealing approach with better parameters
    try:
        current_points = best_result_points.copy()
        current_ratio = best_result_ratio
        temperature = 0.02
        cooling_rate = 0.95
        min_temperature = 1e-6
        
        for iteration in range(300):  # Increased iterations for better convergence
            # Generate neighbor solution
            neighbor_points = current_points.copy()
            # Move one random point
            move_idx = np.random.randint(0, n)
            neighbor_points[move_idx] += np.random.normal(0, temperature, 2)
            neighbor_points[move_idx] = np.clip(neighbor_points[move_idx], 0, 1)
            
            # Calculate ratio for neighbor
            neighbor_ratio = calculate_ratio(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if neighbor_ratio > current_ratio or np.random.random() < np.exp((neighbor_ratio - current_ratio) / temperature):
                current_points = neighbor_points
                current_ratio = neighbor_ratio
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
        if current_ratio > best_result_ratio:
            best_result_points = current_points
            best_result_ratio = current_ratio
            
    except Exception as e:
        pass
    
    return best_result_points


# EVOLVE-BLOCK-END
