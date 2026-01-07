# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm with local refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using a more sophisticated approach based on known good packings
    def initialize_better():
        # Use a more advanced initialization strategy inspired by good known packings
        circles = []
        
        # Create a better initial pattern using a more systematic approach
        # Based on hexagonal packing principles adapted for square container
        
        # Try to place circles in a hexagonal lattice pattern
        # First, determine how many rows/columns we need
        sqrt_n = math.sqrt(n)
        rows = int(math.ceil(sqrt_n))
        cols = int(math.ceil(n / rows))
        
        # Adjust to ensure we have enough slots
        if rows * cols < n:
            cols += 1
            
        # Calculate spacing to fit in unit square with some margin
        spacing_x = 0.95 / cols
        spacing_y = 0.95 / rows
        
        # Create a more refined initial placement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for better packing
                x_offset = 0.025 + j * spacing_x
                if i % 2 == 1:
                    x_offset += spacing_x / 2
                    
                y_offset = 0.025 + i * spacing_y
                
                # Calculate appropriate initial radius based on spacing
                max_radius = min(spacing_x, spacing_y) / 2.0
                
                # Use slightly smaller radius to allow for optimization
                r = max_radius * 0.8
                
                circles.append([x_offset, y_offset, r])
            if len(circles) >= n:
                break
        
        # Fill remaining slots with carefully placed circles
        while len(circles) < n:
            # Place remaining circles near the center with small radii
            circles.append([0.5, 0.5, 0.02])
            
        return np.array(circles[:n])
    
    # Better constraint checking with early termination
    def check_containment(circles):
        """Check if all circles are contained in the unit square"""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True
    
    def check_overlap(circles):
        """Check if any circles overlap"""
        # Use spatial indexing for better performance
        if len(circles) < 2:
            return True
            
        # Simple pairwise check for small number of circles
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    return False
        return True
    
    def calculate_total_radius(circles):
        """Calculate sum of all radii"""
        return sum(circle[2] for circle in circles)
    
    # Improved evolutionary algorithm approach
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (set of circle parameters)"""
        # Convert flat array to circles
        circles = []
        for i in range(0, len(individual), 3):
            x, y, r = individual[i], individual[i+1], individual[i+2]
            circles.append([x, y, r])
        
        # Check constraints
        if not check_containment(circles):
            return -np.inf  # Invalid solution
            
        if not check_overlap(circles):
            return -np.inf  # Overlapping circles
            
        # Return negative of sum of radii (since we want to maximize)
        return -calculate_total_radius(circles)
    
    # Create evolutionary algorithm components
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.001, 0.999)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=96)  # 32*3
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.15)  # Reduced sigma for finer control
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population with better starting points
    def init_population_with_improvement(pop_size):
        population = []
        for _ in range(pop_size):
            # Start with a good initialization
            circles = initialize_better()
            # Add some noise to make it diverse
            for i in range(len(circles)):
                circles[i][0] += random.uniform(-0.01, 0.01)
                circles[i][1] += random.uniform(-0.01, 0.01)
                circles[i][2] *= random.uniform(0.95, 1.05)
            individual = []
            for x, y, r in circles:
                individual.extend([x, y, r])
            population.append(individual)
        return population
    
    # Run evolutionary algorithm
    try:
        # Initialize population with better starting points
        pop = init_population_with_improvement(30)
        
        # Run evolution with more generations
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        
        # Run with more generations to improve quality
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                          ngen=80, stats=stats, halloffame=hof, 
                                          verbose=False)
        
        # Get best individual
        best_individual = hof[0]
        circles = []
        for i in range(0, len(best_individual), 3):
            circles.append([best_individual[i], best_individual[i+1], best_individual[i+2]])
            
        # Refine with local optimization
        refined_circles = refine_solution(circles)
        return np.array(refined_circles)
        
    except Exception as e:
        # Fallback to local optimization if EA fails
        pass
    
    # Fallback to local optimization approach
    circles = initialize_better()
    
    # Define constraint functions with improved numerical stability
    def containment_constraints(x):
        """Ensure all circles are contained in unit square"""
        constraints = []
        for i in range(n):
            x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
            # r_i <= x_i <= 1-r_i
            constraints.append(x_i - r_i)  # x_i - r_i >= 0
            constraints.append(1 - x_i - r_i)  # 1 - x_i - r_i >= 0
            # r_i <= y_i <= 1-r_i
            constraints.append(y_i - r_i)  # y_i - r_i >= 0
            constraints.append(1 - y_i - r_i)  # 1 - y_i - r_i >= 0
        return np.array(constraints)
    
    def non_overlap_constraints(x):
        """Ensure no two circles overlap with numerical tolerance"""
        constraints = []
        # Use a more efficient approach for overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                
                # Distance between centers >= sum of radii
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                min_dist_sq = (r_i + r_j)**2
                
                # Add small tolerance to prevent numerical issues
                constraints.append(dist_sq - min_dist_sq - 1e-12)
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x):
        total_radius = 0
        for i in range(n):
            total_radius += x[3*i+2]  # Sum of radii
        return -total_radius
    
    # Flatten initial circles for optimization
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Set up bounds for optimization with tighter constraints
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Constraints
    cons = []
    # Add containment constraints
    def containment_func(x):
        return containment_constraints(x)
    cons.append({'type': 'ineq', 'fun': containment_func})
    
    # Add non-overlap constraints
    def overlap_func(x):
        return non_overlap_constraints(x)
    cons.append({'type': 'ineq', 'fun': overlap_func})
    
    # Optimize with multiple strategies
    try:
        # Strategy 1: SLSQP with tighter tolerances
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 3000, 'ftol': 1e-10, 'eps': 1e-8, 'disp': False})
        
        if result.success:
            # Extract optimized results
            optimized_circles = []
            for i in range(n):
                x_i = result.x[3*i]
                y_i = result.x[3*i+1]
                r_i = result.x[3*i+2]
                optimized_circles.append([x_i, y_i, r_i])
            return np.array(optimized_circles)
    except Exception as e:
        pass
    
    # Strategy 2: Try different optimization method if first fails
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False})
        
        if result.success:
            # Extract optimized results
            optimized_circles = []
            for i in range(n):
                x_i = result.x[3*i]
                y_i = result.x[3*i+1]
                r_i = result.x[3*i+2]
                optimized_circles.append([x_i, y_i, r_i])
            return np.array(optimized_circles)
    except Exception as e:
        pass
    
    # Strategy 3: If both fail, try trust-constr method which is often more robust
    try:
        result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=cons,
                         options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False})
        
        if result.success:
            # Extract optimized results
            optimized_circles = []
            for i in range(n):
                x_i = result.x[3*i]
                y_i = result.x[3*i+1]
                r_i = result.x[3*i+2]
                optimized_circles.append([x_i, y_i, r_i])
            return np.array(optimized_circles)
    except Exception as e:
        pass
    
    # If all optimization attempts fail, return the initial configuration
    return circles

def refine_solution(circles):
    """Apply local refinement to improve the solution"""
    # More sophisticated refinement approach
    refined = [list(circle) for circle in circles]
    
    # Apply a series of local improvements
    for iteration in range(30):  # More iterations for better refinement
        improved = False
        # Try to increase each radius where possible
        for i in range(len(refined)):
            x, y, r = refined[i]
            
            # Calculate maximum possible radius at this location
            max_radius = min(x, y, 1-x, 1-y)
            
            # Try to increase radius while maintaining constraints
            original_r = r
            step = 0.001
            
            # Binary search for maximum radius
            low = r
            high = max_radius
            best_r = r
            
            # Check if we can increase radius
            while high - low > 1e-8:
                test_r = (low + high) / 2
                valid = True
                
                # Check containment
                if test_r > x or test_r > y or test_r > (1-x) or test_r > (1-y):
                    valid = False
                
                # Check overlap with all others
                if valid:
                    for j in range(len(refined)):
                        if i != j:
                            x2, y2, r2 = refined[j]
                            dist_sq = (x - x2)**2 + (y - y2)**2
                            min_dist_sq = (test_r + r2)**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                
                if valid:
                    best_r = test_r
                    low = test_r
                else:
                    high = test_r
            
            if best_r > r + 1e-8:
                refined[i][2] = best_r
                improved = True
        
        # Try small position adjustments to free up space
        for i in range(len(refined)):
            x, y, r = refined[i]
            
            # Try to move circle slightly to make room for others
            best_x, best_y = x, y
            best_radius = r
            
            # Test small displacements
            for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                    new_x = x + dx
                    new_y = y + dy
                    
                    # Check if new position is valid
                    if new_x - r >= 0 and new_x + r <= 1 and new_y - r >= 0 and new_y + r <= 1:
                        # Check overlap with others
                        valid = True
                        for j in range(len(refined)):
                            if i != j:
                                x2, y2, r2 = refined[j]
                                dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                                min_dist_sq = (r + r2)**2
                                if dist_sq < min_dist_sq:
                                    valid = False
                                    break
                        
                        if valid:
                            # If valid, try to increase radius
                            max_r = min(new_x, new_y, 1-new_x, 1-new_y)
                            if max_r > r:
                                refined[i][0] = new_x
                                refined[i][1] = new_y
                                refined[i][2] = max_r
                                improved = True
                                break
            if improved:
                break
                
        if not improved:
            break
    
    return refined


# EVOLVE-BLOCK-END
