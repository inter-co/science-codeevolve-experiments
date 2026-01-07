# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from deap import base, creator, tools, algorithms
from functools import partial
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithms and local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    max_evaluations = 10000
    population_size = 50
    generations = 200
    
    # Initialize with a hexagonal packing approach for better density
    def initialize_circles():
        # Hexagonal packing approach - more efficient than regular grid
        circles = []
        
        # Try to arrange in hexagonal pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Hexagonal spacing
        hex_radius = 0.15  # Initial guess
        spacing_x = hex_radius * 2
        spacing_y = hex_radius * np.sqrt(3)
        
        # Calculate how many circles fit in the square
        actual_rows = int(np.floor(1.0 / spacing_y))
        actual_cols = int(np.floor(1.0 / spacing_x))
        
        # Adjust spacing to fit within unit square
        if actual_rows * spacing_y > 1.0:
            spacing_y = 1.0 / actual_rows
        if actual_cols * spacing_x > 1.0:
            spacing_x = 1.0 / actual_cols
            
        # Generate hexagonal pattern
        count = 0
        for i in range(actual_rows):
            for j in range(actual_cols):
                if count >= n:
                    break
                    
                # Offset odd rows
                x_offset = (i % 2) * spacing_x / 2
                x = x_offset + spacing_x * j + spacing_x / 2
                y = spacing_y * i + spacing_y / 2
                
                # Ensure we stay within bounds
                x = max(spacing_x/2, min(1.0 - spacing_x/2, x))
                y = max(spacing_y/2, min(1.0 - spacing_y/2, y))
                
                # Add slight randomness to avoid perfect patterns
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                
                # Keep radius small initially
                circles.append([x, y, min(0.05, spacing_x/2)])
                count += 1
                
            if count >= n:
                break
        
        # Fill any remaining slots with random positions
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.02])
            
        return np.array(circles[:n])
    
    # Check if circles are valid (inside bounds and non-overlapping)
    def is_valid(circles):
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                if dist_sq < (r1 + r2)**2:
                    return False
        return True
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        # Extract radii
        radii = circles_flat[2::3]
        # Return negative sum (for minimization)
        return -np.sum(radii)
    
    # Constraint penalty function
    def penalty_function(circles_flat):
        penalty = 0
        circles = circles_flat.reshape(-1, 3)
        
        # Boundary penalties
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Penalize if circle is outside bounds
            penalty += max(0, r - x)  # r > x
            penalty += max(0, r - (1 - x))  # r > 1-x
            penalty += max(0, r - y)  # r > y
            penalty += max(0, r - (1 - y))  # r > 1-y
        
        # Overlap penalties
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    # Penalty proportional to overlap amount
                    overlap = min_dist_sq - dist_sq
                    penalty += overlap * 1000
        
        return penalty
    
    # Local optimization function
    def local_optimize(initial_circles):
        # Use scipy optimization for refinement
        def objective_local(circles_flat):
            return -np.sum(circles_flat[2::3])
        
        def boundary_constraints(circles_flat):
            constraints = []
            for i in range(n):
                r = circles_flat[i*3 + 2]
                x = circles_flat[i*3]
                y = circles_flat[i*3 + 1]
                # r <= x, r <= 1-x, r <= y, r <= 1-y
                constraints.extend([
                    x - r,           # x >= r
                    1 - x - r,       # 1-x >= r
                    y - r,           # y >= r
                    1 - y - r        # 1-y >= r
                ])
            return np.array(constraints)
        
        def overlap_constraints(circles_flat):
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_flat[i*3], circles_flat[i*3+1], circles_flat[i*3+2]
                    x2, y2, r2 = circles_flat[j*3], circles_flat[j*3+1], circles_flat[j*3+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    # We want dist_sq >= (r1 + r2)^2 for non-overlap
                    constraints.append(dist_sq - (r1 + r2)**2)
            return np.array(constraints)
        
        initial_guess = initial_circles.flatten()
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        try:
            result = minimize(
                objective_local,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)},
                    {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
                ],
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                return optimized_circles
        except Exception:
            pass
        
        return initial_circles
    
    # Evolutionary algorithm approach
    def evolutionary_approach():
        # Create fitness class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define individual creation
        def create_individual():
            circles = initialize_circles()
            # Flatten for GA representation
            return list(circles.flatten())
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Define evaluation function
        def evaluate(individual):
            circles = np.array(individual).reshape(-1, 3)
            
            # Check validity and calculate fitness
            if not is_valid(circles):
                return (-10000,)  # Invalid solution
            
            # Calculate sum of radii as fitness
            total_radius = np.sum(circles[:, 2])
            return (total_radius,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        
        try:
            pop, logbook = algorithms.eaSimple(
                pop, toolbox, cxpb=0.5, mutpb=0.2, 
                ngen=generations, 
                halloffame=hof, verbose=False
            )
        except Exception:
            pass
        
        best_individual = hof[0]
        best_circles = np.array(best_individual).reshape(-1, 3)
        
        # Apply local optimization to best solution
        refined_solution = local_optimize(best_circles)
        return refined_solution
    
    # Multi-start approach: try several random starts
    best_solution = None
    best_sum = -float('inf')
    
    # Try evolutionary approach first
    try:
        start_time = time.time()
        evol_solution = evolutionary_approach()
        evol_sum = np.sum(evol_solution[:, 2])
        
        if evol_sum > best_sum:
            best_sum = evol_sum
            best_solution = evol_solution
            
        elapsed = time.time() - start_time
        if elapsed > 50:  # Early exit if taking too long
            pass
    except Exception:
        pass
    
    # If no evolutionary solution found, fall back to local optimization with better initializations
    if best_solution is None:
        # Try several random initializations with local optimization
        for _ in range(5):
            try:
                initial = initialize_circles()
                optimized = local_optimize(initial)
                current_sum = np.sum(optimized[:, 2])
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = optimized
            except Exception:
                continue
    
    # Final fallback to initial configuration if nothing worked
    if best_solution is None:
        initial = initialize_circles()
        return initial
    
    return best_solution


# EVOLVE-BLOCK-END
