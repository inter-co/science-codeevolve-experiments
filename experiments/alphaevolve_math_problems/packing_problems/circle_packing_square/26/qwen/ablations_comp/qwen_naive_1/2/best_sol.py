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
    Uses a hybrid approach combining smart initialization with local optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    random.seed(42)
    np.random.seed(42)
    
    # Enhanced initialization using a more sophisticated approach inspired by known good packings
    def initialize_better_circles():
        circles = np.zeros((n, 3))
        
        # Use a pattern that's closer to optimal known solutions
        # Start with a hexagonal close packing pattern but adapted for 26 circles
        rows = 5
        cols = 5
        
        # Calculate spacing based on hexagonal packing efficiency
        spacing_x = 0.85 / cols
        spacing_y = 0.85 / rows
        offset_x = 0.075
        offset_y = 0.075
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal packing: alternate rows offset
                x_offset = 0 if i % 2 == 0 else spacing_x * 0.5
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Add small randomization to avoid perfect grid artifacts
                x += np.random.uniform(-spacing_x*0.05, spacing_x*0.05)
                y += np.random.uniform(-spacing_y*0.05, spacing_y*0.05)
                
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Start with a reasonable initial radius
                circles[idx] = [x, y, 0.09]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions with careful placement
        for i in range(idx, n):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Check distance to all existing circles
                min_dist = float('inf')
                for k in range(i):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                # Place if sufficiently distant or after many attempts
                if min_dist > 0.15 or attempts > 50:
                    circles[i] = [x, y, 0.09]
                    placed = True
                attempts += 1
            
            if not placed:
                # Fallback to simple random placement with proper bounds
                circles[i] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95), 0.09]
            
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
    
    # Enhanced local optimization approach with better convergence
    def optimize_with_local_search(initial_solution):
        # Use a combination of optimization methods with better parameter tuning
        try:
            # Start with SLSQP for better convergence
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Try multiple optimization approaches with different parameters
            methods = ['SLSQP', 'L-BFGS-B']
            best_result = initial_solution.copy()
            best_value = -objective(initial_solution)
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_solution,
                        method=method,
                        bounds=bounds,
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                    
                    if result.success:
                        new_value = -objective(result.x)
                        if new_value > best_value:
                            best_value = new_value
                            best_result = result.x.copy()
                except:
                    continue
            
            return best_result
            
        except Exception as e:
            return initial_solution
    
    # Advanced refinement with multiple local searches and better restart strategy
    def advanced_refinement(initial_solution):
        try:
            best_solution = initial_solution.copy()
            best_value = -objective(best_solution)
            
            # Multiple restarts with different random perturbations
            for restart in range(15):  # Increased number of restarts
                try:
                    # Perturb the solution slightly with more aggressive perturbation
                    perturbed = initial_solution.copy()
                    # Add significant perturbation for exploration
                    noise = np.random.normal(0, 0.015, len(perturbed))  # More aggressive noise
                    perturbed += noise
                    
                    # Clip values to valid ranges
                    for i in range(0, len(perturbed), 3):
                        perturbed[i] = np.clip(perturbed[i], 0.001, 0.999)      # x
                        perturbed[i+1] = np.clip(perturbed[i+1], 0.001, 0.999)  # y
                        perturbed[i+2] = np.clip(perturbed[i+2], 0.001, 0.499)  # r
                    
                    # Optimize from this perturbed state
                    bounds = []
                    for i in range(n):
                        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                    
                    result = minimize(
                        objective,
                        perturbed,
                        method='SLSQP',
                        bounds=bounds,
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerance
                    )
                    
                    if result.success:
                        new_value = -objective(result.x)
                        if new_value > best_value:
                            best_value = new_value
                            best_solution = result.x.copy()
                            
                except:
                    continue
                    
            return best_solution
        except:
            return initial_solution
    
    # Even better initialization using a more systematic approach with adaptive spacing
    def initialize_adaptive_pattern():
        circles = np.zeros((n, 3))
        
        # Generate points in a more strategic pattern
        # Try to create a configuration that allows for larger radii
        
        # Use a more sophisticated approach: start with a regular grid then adjust
        grid_size = 5  # 5x5 grid for 25 circles, then add one more
        spacing_x = 0.9 / grid_size
        spacing_y = 0.9 / grid_size
        offset_x = 0.05
        offset_y = 0.05
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                # Add randomness to positions but keep them well-distributed
                x = offset_x + j * spacing_x + np.random.uniform(-spacing_x*0.15, spacing_x*0.15)
                y = offset_y + i * spacing_y + np.random.uniform(-spacing_y*0.15, spacing_y*0.15)
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Start with larger radius to allow for better packing
                circles[idx] = [x, y, 0.10]
                idx += 1
                
                if idx >= n:
                    break
        
        # Place the 26th circle strategically
        if idx < n:
            # Find the center point that maximizes minimum distance to existing circles
            best_x, best_y = 0.5, 0.5
            best_min_dist = 0
            
            # Sample potential locations
            for _ in range(1000):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                min_dist = float('inf')
                for k in range(idx):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_x, best_y = x, y
            
            circles[idx] = [best_x, best_y, 0.10]
        
        return circles
    
    # Enhanced constraint checker with early termination and better error handling
    def enhanced_constraint_check(circles):
        """More robust constraint checking with better error handling"""
        try:
            # Extract positions and radii
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Use numba-optimized version first
            return fast_compute_constraints_jit(positions, radii, n)
        except Exception:
            # Fallback to manual computation
            constraints = []
            
            # Boundary constraints
            for i in range(n):
                x, y, r = circles[i]
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
                    constraints.append(dist_sq - (r1 + r2)**2)
            
            return np.array(constraints)
    
    # Evolutionary algorithm approach for global search
    def evolutionary_approach():
        # Create a simple evolutionary algorithm for better global search
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        # Define gene boundaries: [x, y, r] for each circle
        bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
        
        def create_individual():
            individual = []
            for i in range(n):
                for bound in bounds[i*3:(i+1)*3]:
                    individual.append(random.uniform(bound[0], bound[1]))
            return creator.Individual(individual)
        
        def evaluate(individual):
            # Convert to circles format
            circles = np.array(individual).reshape(-1, 3)
            constraints = enhanced_constraint_check(circles)
            if np.any(constraints < -1e-6):
                return (-1000,)  # Invalid solution penalty
            return (np.sum(circles[:, 2]),)  # Return sum of radii
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.7, mutpb=0.3, 
                ngen=50, stats=stats, halloffame=hof, verbose=False
            )
            return np.array(hof[0]).reshape(-1, 3)
        except:
            return None
    
    # Main optimization process with improved strategies
    try:
        # Strategy: Use evolutionary approach for global search, then local optimization
        best_result = None
        best_sum = -float('inf')
        
        # Try evolutionary approach first for global search
        try:
            evol_result = evolutionary_approach()
            if evol_result is not None:
                # Validate and optimize the evolutionary result
                constraints = enhanced_constraint_check(evol_result)
                if np.all(constraints >= -1e-6):
                    radii_sum = np.sum(evol_result[:, 2])
                    if radii_sum > best_sum:
                        best_sum = radii_sum
                        best_result = evol_result.flatten().copy()
        except:
            pass
        
        # If evolutionary didn't work, try the initialization strategies
        init_strategies = [
            initialize_better_circles,
            initialize_adaptive_pattern
        ]
        
        # Try different initialization strategies
        for i, init_func in enumerate(init_strategies):
            try:
                # Initialize with this strategy
                initial_circles = init_func()
                initial_flat = initial_circles.flatten()
                
                # Apply multiple rounds of local optimization
                refined_result = advanced_refinement(initial_flat)
                
                # Evaluate final result
                circles = np.array(refined_result).reshape(-1, 3)
                constraints = enhanced_constraint_check(circles)
                
                if np.all(constraints >= -1e-6):  # Acceptable constraints
                    radii_sum = np.sum(circles[:, 2])
                    if radii_sum > best_sum:
                        best_sum = radii_sum
                        best_result = refined_result.copy()
                        
            except Exception as e:
                continue
        
        # If no good result found, use fallback with more aggressive optimization
        if best_result is None:
            # Use simple initialization and single optimization with more aggressive settings
            circles = initialize_better_circles()
            circles_flat = circles.flatten()
            
            # Single optimization run with more iterations and tighter tolerances
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            try:
                result = minimize(
                    objective,
                    circles_flat,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 2500, 'ftol': 1e-10, 'gtol': 1e-10}  # Very tight tolerance
                )
                
                if result.success:
                    best_result = result.x
                else:
                    best_result = circles_flat
            except Exception as e:
                best_result = circles_flat
        
        # Convert back to circles format
        circles = np.array(best_result).reshape(-1, 3)
        
        # Final validation and cleanup
        constraints = enhanced_constraint_check(circles)
        if np.any(constraints < -1e-6):
            # If constraints still violated, use the initialization
            circles = initialize_better_circles()
        
        return circles
        
    except Exception as e:
        # Fallback to simple initialization and basic optimization
        circles = initialize_better_circles()
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
                options={'maxiter': 2500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                circles_optimized = result.x.reshape(-1, 3)
                return circles_optimized
        except Exception as e2:
            pass
            
        return circles


# EVOLVE-BLOCK-END
