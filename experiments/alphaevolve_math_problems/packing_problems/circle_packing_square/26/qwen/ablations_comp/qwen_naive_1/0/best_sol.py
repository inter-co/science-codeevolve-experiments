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
def compute_constraints_fast(positions, radii, n):
    """Fast constraint computation using numba"""
    constraints = []
    
    # Boundary constraints: each circle must stay within bounds
    for i in range(n):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        # Each circle must stay within bounds (r <= x <= 1-r, r <= y <= 1-r)
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
    Uses hybrid approach with improved initialization, enhanced optimization, and better 
    constraint handling to beat the benchmark.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    random.seed(42)
    np.random.seed(42)
    
    # Improved initialization with better spatial distribution
    def initialize_circles():
        circles = np.zeros((n, 3))
        
        # Use a more sophisticated initialization based on known good patterns
        # Start with a hexagonal lattice pattern that's known to work well
        # This uses a triangular/hexagonal packing approach
        
        # Create a more systematic grid with better spacing
        rows = 5
        cols = 5
        spacing_x = 0.8 / (cols - 1) if cols > 1 else 0.8
        spacing_y = 0.8 / (rows - 1) if rows > 1 else 0.8
        offset_x = 0.1
        offset_y = 0.1
        
        idx = 0
        # Create a hexagonal pattern with alternating rows
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for better packing
                x_offset = (i % 2) * spacing_x * 0.5
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Add more structured jitter to avoid perfect symmetry
                jitter_x = np.random.normal(0, spacing_x * 0.03)
                jitter_y = np.random.normal(0, spacing_y * 0.03)
                x = offset_x + j * spacing_x + x_offset + jitter_x
                y = offset_y + i * spacing_y + jitter_y
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Start with a more informed initial radius
                # Use a value that allows room for growth while respecting boundaries
                r = min(0.08, 0.5 * min(x, 1-x, y, 1-y))
                circles[idx] = [x, y, r]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions with more intelligent random placement
        for i in range(idx, n):
            # Try to place far enough from existing circles
            attempts = 0
            placed = False
            while not placed and attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Check distance to existing circles
                min_dist = float('inf')
                for k in range(i):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                # Require minimum distance to existing circles
                if min_dist < 0.05:
                    attempts += 1
                    continue
                    
                # Set radius based on available space
                r = min(0.08, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]
                placed = True
                
            if not placed:
                # Fallback to simple random placement with reasonable radius
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = min(0.08, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]
            
        return circles
    
    # Enhanced constraint checking
    def compute_constraints(circles):
        """Compute all constraints efficiently"""
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Use fast numba version for performance
        constraints = compute_constraints_fast(positions, radii, n)
        return constraints
    
    # Optimization objective - maximize sum of radii
    def objective(circles_flat):
        # Extract radii from flattened array
        radii = circles_flat[2::3]
        return -np.sum(radii)  # Negative because we want to maximize
    
    # Enhanced constraint function
    def constraint_func(circles_flat):
        # Convert flat array back to circles
        circles = circles_flat.reshape(-1, 3)
        return compute_constraints(circles)
    
    # Improved evolutionary algorithm with better parameters
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
        
        # Evaluation function with better error handling
        def eval_individual(individual):
            try:
                circles = np.array(individual).reshape(-1, 3)
                # Check constraints
                constraints = compute_constraints(circles)
                # If any constraint violated significantly, return very low fitness
                if np.any(constraints < -1e-4):  # Looser tolerance for early pruning
                    return (-1e6,)
                # Otherwise, return negative sum of radii (since we're maximizing)
                radii = circles[:, 2]
                return (np.sum(radii),)
            except Exception as e:
                return (-1e6,)
        
        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.3)  # Reduced crossover probability for more stability
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.005, indpb=0.3)  # Smaller mutation for fine-tuning
        toolbox.register("select", tools.selTournament, tournsize=3)  # Even smaller tournament size for more exploitation
        
        # Run evolution with more generations for better exploration
        pop = toolbox.population(n=150)  # Slightly reduced population for faster runs
        hof = tools.HallOfFame(1)
        
        try:
            # Run for more generations with better parameters
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.4, 
                                             ngen=150, stats=stats, halloffame=hof, 
                                             verbose=False)
            return hof[0]
        except Exception as e:
            # Fallback to local optimization if EA fails
            return toolbox.individual()
    
    # Better local optimization with multiple restarts
    def refine_solution(initial_solution):
        best_result = initial_solution
        best_value = -float('inf')
        
        # Try multiple local optimizations with different starting points
        for restart in range(15):  # More restarts for better chance of finding good solution
            try:
                # Slightly perturb the initial solution for restarts
                perturbed = initial_solution.copy()
                for i in range(len(perturbed)):
                    if i % 3 != 2:  # Don't perturb positions too much
                        perturbed[i] += np.random.normal(0, 0.015)
                    else:  # Perturb radii more for exploration
                        perturbed[i] += np.random.normal(0, 0.005)
                
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 3000, 'ftol': 1e-9}  # More iterations and tighter tolerance
                )
                
                if result.success:
                    # Check if this result is better
                    current_radii = result.x[2::3]
                    current_sum = np.sum(current_radii)
                    if current_sum > best_value:
                        best_value = current_sum
                        best_result = result.x
                        
            except:
                continue
        
        return best_result
    
    # Alternative optimization approach using a more targeted strategy
    def optimized_local_search():
        # Start with a good initialization
        circles = initialize_circles()
        
        # Try a more aggressive optimization approach
        circles_flat = circles.flatten()
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Try multiple optimization methods
        methods = ['SLSQP', 'L-BFGS-B']
        best_result = None
        best_sum = -float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    circles_flat,
                    method=method,
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 2000, 'ftol': 1e-8}
                )
                
                if result.success:
                    current_radii = result.x[2::3]
                    current_sum = np.sum(current_radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result.x
            except:
                continue
        
        if best_result is not None:
            return best_result.reshape(-1, 3)
        else:
            return circles
    
    # Main optimization process
    try:
        # Strategy 1: Evolutionary approach for global optimization
        start_time = time.time()
        evol_result = optimize_with_evolution()
        evol_time = time.time() - start_time
        
        # Strategy 2: Local optimization refinement with multiple restarts
        refined_result = refine_solution(evol_result)
        
        # Convert back to circles format
        circles = np.array(refined_result).reshape(-1, 3)
        
        # Final validation and cleanup
        constraints = compute_constraints(circles)
        if np.any(constraints < -1e-4):
            # If constraints still violated significantly, try a different approach
            # Reset to a better initial configuration
            circles = initialize_circles()
            try:
                circles_flat = circles.flatten()
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                    
                result = minimize(
                    objective,
                    circles_flat,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 3000, 'ftol': 1e-9}
                )
                
                if result.success:
                    circles = result.x.reshape(-1, 3)
            except:
                pass
            
        # As final fallback, try direct optimization
        if np.sum(circles[:, 2]) < 2.5:
            circles = optimized_local_search()
            
        return circles
        
    except Exception as e:
        # Fallback to simple initialization and basic optimization
        circles = initialize_circles()
        try:
            circles_flat = circles.flatten()
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
            result = minimize(
                objective,
                circles_flat,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 3000, 'ftol': 1e-9}
            )
            
            if result.success:
                circles_optimized = result.x.reshape(-1, 3)
                return circles_optimized
        except:
            pass
            
        return circles


# EVOLVE-BLOCK-END
