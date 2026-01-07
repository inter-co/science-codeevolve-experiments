# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random
from deap import base, creator, tools, algorithms
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining evolutionary algorithms with local optimization for better results.
    """
    n = 32
    
    # Create initial configuration using a more sophisticated approach
    def create_initial_placement():
        # Strategy: Hexagonal close packing pattern with some randomness
        circles = []
        
        # Create hexagonal grid pattern
        rows = 6
        cols = 6
        spacing_x = 0.15
        spacing_y = 0.15 * math.sqrt(3) / 2
        radius = 0.05
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, radius])
        
        # Fill remaining slots if needed with random positions
        random.seed(42)  # Fixed seed for reproducibility
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Use smaller initial radius to allow room for growth
            circles.append([x, y, 0.03])
            
        return np.array(circles[:n])
    
    # More efficient constraint evaluation using vectorized operations
    def constraint_func(vars):
        # Return array of constraint values (positive means violated)
        constraints = []
        n = 32
        
        # Reshape variables
        circles = vars.reshape(-1, 3)
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Boundary constraints
        for i in range(n):
            x, y, r = x_coords[i], y_coords[i], radii[i]
            constraints.extend([
                x - r,           # left boundary
                1 - x - r,       # right boundary
                y - r,           # bottom boundary
                1 - y - r        # top boundary
            ])
        
        # Non-overlap constraints - use efficient pairwise computation
        # Vectorized distance calculation
        for i in range(n):
            for j in range(i+1, n):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                constraints.append(dist_sq - r_sum*r_sum)
                
        return np.array(constraints)
    
    # Efficient objective function
    def objective(vars):
        # Sum of radii (we want to maximize this)
        circles = vars.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Improved constraint checking with early termination
    def check_constraints(vars):
        """Check if all constraints are satisfied"""
        circles = vars.reshape(-1, 3)
        n = len(circles)
        
        # Check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                r_sum = r1 + r2
                if dist_sq < r_sum * r_sum:
                    return False
                    
        return True
    
    # Create initial placement
    circles = create_initial_placement()
    
    # Flatten into variables [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = circles.flatten()
    
    # Set bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Try evolutionary algorithm first for global search
    try:
        # Define DEAP problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.001, 0.999)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n*3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def eval_circle_pack(individual):
            # Convert individual to circles
            circles = np.array(individual).reshape(-1, 3)
            
            # Check if constraints are satisfied
            valid = True
            for i in range(n):
                x, y, r = circles[i]
                # Boundary check
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    valid = False
                    break
            
            if not valid:
                return -1000000,  # Very bad fitness if invalid
            
            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    r_sum = r1 + r2
                    if dist_sq < r_sum * r_sum:
                        valid = False
                        break
                if not valid:
                    return -1000000,  # Bad fitness if overlaps exist
            
            # Return negative sum of radii (since we maximize)
            total_radius = np.sum(circles[:, 2])
            return total_radius,
        
        toolbox.register("evaluate", eval_circle_pack)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        # Run for limited time or generations
        start_time = time.time()
        for gen in range(100):
            if time.time() - start_time > 45:  # Leave 15 seconds for final optimization
                break
                
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
            hof.update(population)
            
            # Early stopping if we're getting good solutions
            if gen % 10 == 0 and hof[0].fitness.values[0] > 2.8:
                break
        
        # Get best solution from evolution
        if len(hof) > 0:
            evolved_vars = np.array(hof[0])
            if check_constraints(evolved_vars):
                initial_vars = evolved_vars.copy()
        
    except Exception as e:
        # If evolutionary fails, continue with optimization
        pass
    
    # Optimization with multiple strategies
    try:
        # First attempt with SLSQP
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_vars = result.x
            circles = optimized_vars.reshape(-1, 3)
        else:
            # If first optimization fails, try L-BFGS-B with custom bounds
            result = minimize(
                objective,
                initial_vars,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_vars = result.x
                circles = optimized_vars.reshape(-1, 3)
            
    except Exception as e:
        # Fallback to initial placement if optimization fails
        pass
    
    # Final validation and adjustment
    # Make sure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Ensure valid bounds
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
        circles[i][2] = max(0.001, min(0.499, r))
    
    # Additional post-processing with local optimization
    try:
        # Better local optimization approach
        circles_copy = circles.copy()
        best_score = -objective(circles_copy.flatten())
        
        # Try to improve by adjusting positions slightly
        for iteration in range(50):  # More iterations than before
            improved = False
            for i in range(n):
                old_x, old_y, old_r = circles_copy[i]
                best_x, best_y, best_r = old_x, old_y, old_r
                best_score = -objective(circles_copy.flatten())
                
                # Try small adjustments to position and radius
                for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                        new_x = max(old_r, min(1-old_r, old_x + dx))
                        new_y = max(old_r, min(1-old_r, old_y + dy))
                        
                        # Temporarily update
                        circles_copy[i] = [new_x, new_y, old_r]
                        score = -objective(circles_copy.flatten())
                        
                        if score > best_score:
                            best_score = score
                            best_x, best_y = new_x, new_y
                            improved = True
                
                circles_copy[i] = [best_x, best_y, old_r]
            
            if not improved:
                break
                
        circles = circles_copy
        
    except:
        pass
    
    return circles


# EVOLVE-BLOCK-END
