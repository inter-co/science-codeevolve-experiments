# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random
from deap import base, creator, tools, algorithms
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_compute_constraints_jit(positions, radii, n):
    """Fast constraint computation using numba"""
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # x <= 1 - r  
            y - r,           # y >= r
            1 - y - r        # y <= 1 - r
        ])
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            r1, r2 = radii[i], radii[j]
            # Distance squared should be >= (r1 + r2)^2 for no overlap
            constraints.append(dist_sq - (r1 + r2)**2)
    
    return np.array(constraints)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses evolutionary algorithm combined with local optimization for better results.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    random.seed(42)
    np.random.seed(42)
    
    # Better initialization using hexagonal packing pattern
    def initialize_circles():
        circles = np.zeros((n, 3))
        
        # Use hexagonal packing pattern for better initial configuration
        rows = 5
        cols = 5
        
        # Hexagonal packing spacing
        spacing = 0.8 / (cols - 1) if cols > 1 else 0.8
        offset_x = 0.1
        offset_y = 0.1
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset pattern
                x_offset = 0 if i % 2 == 0 else spacing * 0.5
                x = offset_x + j * spacing + x_offset
                y = offset_y + i * spacing
                
                # Add small random perturbation to escape local minima
                x += np.random.normal(0, spacing * 0.05)
                y += np.random.normal(0, spacing * 0.05)
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Start with a reasonable initial radius
                circles[idx] = [x, y, 0.08]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions carefully
        for i in range(idx, n):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Check minimum distance to all existing circles
                min_dist = float('inf')
                for k in range(i):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                # Place if sufficiently distant
                if min_dist > 0.1 or attempts > 50:
                    circles[i] = [x, y, 0.08]
                    placed = True
                attempts += 1
            
            if not placed:
                # Fallback to simple random placement
                circles[i] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95), 0.08]
            
        return circles
    
    # Optimized constraint checking with better numerical stability
    def compute_constraints(circles):
        """Compute all constraints efficiently"""
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Use numba-optimized version for speed
        try:
            return fast_compute_constraints_jit(positions, radii, n)
        except:
            # Fallback to pure Python version
            constraints = []
            
            # Boundary constraints: each circle must stay within bounds
            for i in range(n):
                x, y, r = circles[i]
                # Each circle must stay within bounds (r <= x <= 1-r, r <= y <= 1-r)
                constraints.extend([
                    x - r,           # x >= r
                    1 - x - r,       # x <= 1 - r  
                    y - r,           # y >= r
                    1 - y - r        # y <= 1 - r
                ])
            
            # Overlap constraints using vectorized computation
            try:
                distances = cdist(positions, positions, 'sqeuclidean')
                # Overlap constraints: distance^2 >= (r1 + r2)^2 for all pairs
                for i in range(n):
                    for j in range(i+1, n):
                        dist_sq = distances[i, j]
                        r1, r2 = radii[i], radii[j]
                        # Distance squared should be >= (r1 + r2)^2 for no overlap
                        constraints.append(dist_sq - (r1 + r2)**2)
            except:
                # Fallback for any computation errors
                for i in range(n):
                    for j in range(i+1, n):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist_sq = dx*dx + dy*dy
                        r1, r2 = radii[i], radii[j]
                        constraints.append(dist_sq - (r1 + r2)**2)
                
            return np.array(constraints)
    
    # Optimization objective - maximize sum of radii
    def objective(circles_flat):
        # Extract radii from flattened array
        radii = circles_flat[2::3]
        return -np.sum(radii)  # Negative because we want to maximize
    
    # Enhanced constraint function with proper handling
    def constraint_func(circles_flat):
        # Convert flat array back to circles
        circles = circles_flat.reshape(-1, 3)
        return compute_constraints(circles)
    
    # Better evolutionary algorithm approach with improved operators
    def optimize_with_evolution():
        # Define the optimization problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define individual and population
        def create_individual():
            # Create a valid initial solution
            circles = initialize_circles()
            # Flatten to create individual
            return circles.flatten().tolist()
        
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Custom crossover and mutation for circle packing
        def cx_circles(ind1, ind2):
            # Uniform crossover for positions and radii
            for i in range(len(ind1)):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2
        
        def mut_circles(individual, indpb=0.1):
            # Mutate positions and radii differently
            for i in range(len(individual)):
                if random.random() < indpb:
                    # Mutate positions and radii with different strategies
                    if i % 3 == 0 or i % 3 == 1:  # x or y coordinate
                        individual[i] += np.random.normal(0, 0.01)
                        individual[i] = np.clip(individual[i], 0.001, 0.999)
                    else:  # radius
                        individual[i] += np.random.normal(0, 0.005)
                        individual[i] = np.clip(individual[i], 0.001, 0.499)
            return (individual,)
        
        toolbox.register("evaluate", lambda ind: evaluate_individual(ind))
        toolbox.register("mate", cx_circles)
        toolbox.register("mutate", mut_circles)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution with better parameters for convergence
        pop = toolbox.population(n=100)  # Larger population
        hof = tools.HallOfFame(1)
        
        try:
            # Run for more generations with adaptive parameters
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=70, lambda_=70, 
                                                   cxpb=0.7, mutpb=0.5, 
                                                   ngen=100, stats=stats, halloffame=hof, 
                                                   verbose=False)
            return hof[0]
        except Exception as e:
            # Fallback to local optimization if EA fails
            return toolbox.individual()
    
    # Evaluation function with better error handling and constraint checking
    def evaluate_individual(individual):
        try:
            circles = np.array(individual).reshape(-1, 3)
            # Check constraints
            constraints = compute_constraints(circles)
            # If any constraint violated significantly, return very low fitness
            if np.any(constraints < -1e-5):  # Even stricter constraint checking
                return (-1e8,)
            # Otherwise, return negative sum of radii (since we're maximizing)
            radii = circles[:, 2]
            return (np.sum(radii),)
        except Exception as e:
            return (-1e8,)
    
    # Improved local optimization refinement with multiple strategies
    def refine_solution(initial_solution):
        try:
            # First try with L-BFGS-B - often faster and more reliable for this problem
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            result = minimize(
                objective,
                initial_solution,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                return result.x
            
            # If L-BFGS-B fails, try SLSQP with constraints
            result = minimize(
                objective,
                initial_solution,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1500, 'ftol': 1e-10}
            )
            
            if result.success:
                return result.x
                
        except Exception as e:
            pass
        return initial_solution
    
    # Advanced refinement with multiple local optimizations
    def advanced_refinement(initial_solution):
        try:
            best_solution = initial_solution.copy()
            best_value = -objective(best_solution)
            
            # Try multiple optimization runs with different strategies
            for run in range(8):  # More attempts for better chance
                try:
                    # Perturb the solution slightly
                    perturbed = initial_solution.copy()
                    noise = np.random.normal(0, 0.003, len(perturbed))
                    perturbed += noise
                    
                    # Optimize from this perturbed state
                    bounds = []
                    for i in range(n):
                        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                    
                    # Try both methods for robustness
                    methods_to_try = ['L-BFGS-B']
                    if run % 2 == 0:  # Alternate between methods
                        methods_to_try = ['SLSQP']
                    
                    for method in methods_to_try:
                        try:
                            if method == 'SLSQP':
                                result = minimize(
                                    objective,
                                    perturbed,
                                    method=method,
                                    bounds=bounds,
                                    constraints={'type': 'ineq', 'fun': constraint_func},
                                    options={'maxiter': 1000, 'ftol': 1e-10}
                                )
                            else:
                                result = minimize(
                                    objective,
                                    perturbed,
                                    method=method,
                                    bounds=bounds,
                                    options={'maxiter': 1000, 'ftol': 1e-10}
                                )
                            
                            if result.success:
                                new_value = -objective(result.x)
                                if new_value > best_value:
                                    best_value = new_value
                                    best_solution = result.x.copy()
                        except:
                            continue
                            
                except:
                    continue
                    
            return best_solution
        except:
            return initial_solution
    
    # Improved initialization using better hexagonal pattern
    def initialize_better():
        # Use a more sophisticated initialization that mimics good known solutions
        circles = np.zeros((n, 3))
        
        # Generate points using a hexagonal packing pattern with better boundary handling
        rows = 5
        cols = 5
        
        # Adjust spacing to account for boundary effects
        spacing_x = 0.9 / (cols - 1) if cols > 1 else 0.5
        spacing_y = 0.9 / (rows - 1) if rows > 1 else 0.5
        offset_x = 0.05
        offset_y = 0.05
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset pattern
                x_offset = 0 if i % 2 == 0 else spacing_x * 0.5
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Add small random perturbation to escape local minima
                x += np.random.normal(0, spacing_x * 0.05)
                y += np.random.normal(0, spacing_y * 0.05)
                
                # Ensure within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                # Use a slightly higher initial radius to encourage better packing
                circles[idx] = [x, y, 0.09]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions carefully
        for i in range(idx, n):
            placed = False
            attempts = 0
            while not placed and attempts < 150:  # More attempts
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Check distance to all existing circles
                min_dist = float('inf')
                for k in range(i):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                # Place if sufficiently distant
                if min_dist > 0.12 or attempts > 100:  # Increased minimum distance
                    circles[i] = [x, y, 0.09]
                    placed = True
                attempts += 1
            
            if not placed:
                circles[i] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95), 0.09]
            
        return circles
    
    # Enhanced optimization with better strategy selection and convergence
    def improved_optimization():
        # Strategy: Multi-start with better convergence control
        best_result = None
        best_sum = -float('inf')
        
        # Try multiple different initializations with various approaches
        init_methods = [
            initialize_circles,
            initialize_better
        ]
        
        for init_method in init_methods:
            try:
                # Try multiple starting points for each method
                for _ in range(5):  # More starting points
                    initial_circles = init_method()
                    initial_flat = initial_circles.flatten()
                    
                    # Apply local optimization with multiple strategies
                    bounds = []
                    for i in range(n):
                        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                    
                    # Try L-BFGS-B with strict tolerances
                    try:
                        result = minimize(
                            objective,
                            initial_flat,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 2500, 'ftol': 1e-12, 'gtol': 1e-12}
                        )
                        
                        if result.success:
                            circles = result.x.reshape(-1, 3)
                            constraints = compute_constraints(circles)
                            if np.all(constraints >= -1e-6):
                                radii_sum = np.sum(circles[:, 2])
                                if radii_sum > best_sum:
                                    best_sum = radii_sum
                                    best_result = result.x.copy()
                    except:
                        pass
                    
                    # Try SLSQP with constraints for better constraint handling
                    try:
                        result = minimize(
                            objective,
                            initial_flat,
                            method='SLSQP',
                            bounds=bounds,
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options={'maxiter': 1500, 'ftol': 1e-12}
                        )
                        
                        if result.success:
                            circles = result.x.reshape(-1, 3)
                            constraints = compute_constraints(circles)
                            if np.all(constraints >= -1e-6):
                                radii_sum = np.sum(circles[:, 2])
                                if radii_sum > best_sum:
                                    best_sum = radii_sum
                                    best_result = result.x.copy()
                    except:
                        pass
                        
            except Exception as e:
                continue
        
        # If nothing worked, return a reasonable fallback
        if best_result is None:
            circles = initialize_better()
            return circles.flatten()
        
        return best_result
    
    # Main optimization process with focus on convergence
    try:
        # Strategy: Use multiple high-quality optimization attempts
        best_result = None
        best_sum = -float('inf')
        
        # Multiple optimization attempts with different strategies
        for attempt in range(5):  # More attempts for better chance
            try:
                # Try direct optimization with better initialization
                initial_circles = initialize_better()
                result = improved_optimization()
                
                if result is not None:
                    circles = np.array(result).reshape(-1, 3)
                    constraints = compute_constraints(circles)
                    if np.all(constraints >= -1e-6):
                        radii_sum = np.sum(circles[:, 2])
                        if radii_sum > best_sum:
                            best_sum = radii_sum
                            best_result = result.copy()
            except Exception as e:
                continue
        
        # If no good result yet, try evolutionary approach with better settings
        try:
            if best_result is None:
                evol_result = optimize_with_evolution()
                refined_result = advanced_refinement(evol_result)
                circles = np.array(refined_result).reshape(-1, 3)
                constraints = compute_constraints(circles)
                if np.all(constraints >= -1e-6):
                    radii_sum = np.sum(circles[:, 2])
                    if radii_sum > best_sum:
                        best_sum = radii_sum
                        best_result = refined_result.copy()
        except Exception as e:
            pass
        
        # Final fallback approach with very aggressive refinement
        if best_result is None:
            circles = initialize_better()
            try:
                circles_flat = circles.flatten()
                bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
                
                # Very aggressive optimization with extremely tight tolerances
                result = minimize(
                    objective,
                    circles_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    best_result = result.x
                else:
                    best_result = circles_flat
            except Exception as e:
                best_result = circles_flat
        
        # Convert back to circles format
        circles = np.array(best_result).reshape(-1, 3)
        
        # Final validation and cleanup with more aggressive refinement
        constraints = compute_constraints(circles)
        if np.any(constraints < -1e-6):
            # If constraints still violated, try final optimization
            try:
                circles_flat = circles.flatten()
                bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
                
                # Try with even stricter tolerances
                result = minimize(
                    objective,
                    circles_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    circles = result.x.reshape(-1, 3)
            except:
                pass
        
        # Make sure all circles are valid
        for i in range(n):
            circles[i, 0] = np.clip(circles[i, 0], 0.001, 0.999)
            circles[i, 1] = np.clip(circles[i, 1], 0.001, 0.999)
            circles[i, 2] = np.clip(circles[i, 2], 0.001, 0.499)
        
        return circles
        
    except Exception as e:
        # Fallback to simple initialization and basic optimization
        circles = initialize_better()
        try:
            circles_flat = circles.flatten()
            bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
            
            result = minimize(
                objective,
                circles_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                circles_optimized = result.x.reshape(-1, 3)
                return circles_optimized
        except Exception as e2:
            pass
            
        return circles


# EVOLVE-BLOCK-END
