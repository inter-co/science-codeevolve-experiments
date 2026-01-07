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
from sklearn.cluster import KMeans
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
    
    # Improved initialization with better spatial distribution using hexagonal packing approach
    def initialize_circles():
        circles = np.zeros((n, 3))
        
        # Use a more sophisticated initialization based on known good packings
        # Try a better hexagonal grid pattern
        rows = 5
        cols = 5
        spacing_x = 0.8 / (cols - 1) if cols > 1 else 0.8
        spacing_y = 0.8 / (rows - 1) if rows > 1 else 0.8
        
        # Apply hexagonal offset for better packing
        hex_offset = spacing_x * 0.5
        
        offset_x = 0.1
        offset_y = 0.1
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset for even rows
                x_offset = hex_offset if i % 2 == 1 else 0
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Add small random jitter to avoid perfect patterns
                jitter_x = np.random.normal(0, spacing_x * 0.03)
                jitter_y = np.random.normal(0, spacing_y * 0.03)
                x = max(0.05, min(0.95, x + jitter_x))
                y = max(0.05, min(0.95, y + jitter_y))
                
                # Start with a slightly larger initial radius to encourage growth
                circles[idx] = [x, y, 0.08]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions with smart placement
        for i in range(idx, n):
            # Try to place far enough from existing circles using a smarter approach
            attempts = 0
            placed = False
            while not placed and attempts < 300:  # More attempts for better placement
                # Try different placement strategies
                strategy = np.random.choice(['edge', 'corner', 'random'])
                
                if strategy == 'edge' and i < 12:  # Place some near edges
                    # Place near edges or corners for better spread
                    edge_choice = np.random.choice([0, 1, 2, 3])  # 0=left, 1=top, 2=right, 3=bottom
                    if edge_choice == 0:  # Left edge
                        x = np.random.uniform(0.05, 0.15)
                        y = np.random.uniform(0.1, 0.9)
                    elif edge_choice == 1:  # Top edge
                        x = np.random.uniform(0.1, 0.9)
                        y = np.random.uniform(0.85, 0.95)
                    elif edge_choice == 2:  # Right edge
                        x = np.random.uniform(0.85, 0.95)
                        y = np.random.uniform(0.1, 0.9)
                    else:  # Bottom edge
                        x = np.random.uniform(0.1, 0.9)
                        y = np.random.uniform(0.05, 0.15)
                elif strategy == 'corner' and i < 15:  # Place some near corners
                    corner_choice = np.random.choice([0, 1, 2, 3])  # 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
                    if corner_choice == 0:  # Top-left
                        x = np.random.uniform(0.05, 0.15)
                        y = np.random.uniform(0.85, 0.95)
                    elif corner_choice == 1:  # Top-right
                        x = np.random.uniform(0.85, 0.95)
                        y = np.random.uniform(0.85, 0.95)
                    elif corner_choice == 2:  # Bottom-right
                        x = np.random.uniform(0.85, 0.95)
                        y = np.random.uniform(0.05, 0.15)
                    else:  # Bottom-left
                        x = np.random.uniform(0.05, 0.15)
                        y = np.random.uniform(0.05, 0.15)
                else:  # Random placement
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                
                # Check distance to existing circles more efficiently
                min_dist = float('inf')
                for k in range(i):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                # Require minimum distance for later placements
                if i >= 8 and min_dist < 0.10:
                    attempts += 1
                    continue
                    
                circles[i] = [x, y, 0.08]
                placed = True
                
            if not placed:
                # Fallback to simple random placement with better bounds
                circles[i] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95), 0.08]
            
        return circles
    
    # Optimized constraint checking with early exit for performance
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
    
    # Improved evolutionary algorithm with better parameters and early stopping
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
        
        # Evaluation function with better error handling and caching
        def eval_individual(individual):
            try:
                circles = np.array(individual).reshape(-1, 3)
                # Check constraints
                constraints = compute_constraints(circles)
                # If any constraint violated significantly, return very low fitness
                if np.any(constraints < -1e-6):  # Tighter tolerance for better convergence
                    return (-1e6,)
                # Otherwise, return negative sum of radii (since we're maximizing)
                radii = circles[:, 2]
                return (np.sum(radii),)
            except Exception as e:
                return (-1e6,)
        
        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.4)  # Higher crossover probability
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.015, indpb=0.5)  # Increased mutation rate
        toolbox.register("select", tools.selTournament, tournsize=5)  # Smaller tournament size for more diversity
        
        # Run evolution with adaptive parameters and early stopping
        pop = toolbox.population(n=300)  # Larger population for better exploration
        hof = tools.HallOfFame(1)
        
        try:
            # Run for more generations with better parameters
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Early stopping criteria
            prev_best = -float('inf')
            no_improvement_count = 0
            
            for gen in range(200):  # More generations to allow better exploration
                pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.5, 
                                                 ngen=1, stats=stats, halloffame=hof, 
                                                 verbose=False)
                
                current_best = hof[0].fitness.values[0]
                if current_best > prev_best:
                    prev_best = current_best
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Early stopping if no improvement for 30 generations
                if no_improvement_count >= 30:
                    break
                    
            return hof[0]
        except Exception as e:
            # Fallback to local optimization if EA fails
            return toolbox.individual()
    
    # Better local optimization with multiple restarts and adaptive strategy
    def refine_solution(initial_solution):
        best_result = initial_solution
        best_value = -float('inf')
        
        # Try multiple local optimizations with different starting points
        for restart in range(30):  # More restarts for better chance of finding good solution
            try:
                # Slightly perturb the initial solution for restarts
                perturbed = initial_solution.copy()
                # Apply different perturbation strategies
                for i in range(len(perturbed)):
                    if i % 3 == 0:  # X coordinate
                        perturbed[i] += np.random.normal(0, 0.015)  # Slightly larger perturbation
                    elif i % 3 == 1:  # Y coordinate
                        perturbed[i] += np.random.normal(0, 0.015)  # Slightly larger perturbation
                    else:  # Radius
                        perturbed[i] += np.random.normal(0, 0.008)  # Slightly larger perturbation
                
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                # Try different optimization methods for robustness
                methods = ['trust-constr', 'SLSQP', 'L-BFGS-B']  # More diverse methods
                for method in methods:
                    try:
                        result = minimize(
                            objective,
                            perturbed,
                            method=method,
                            bounds=bounds,
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options={'maxiter': 3000, 'ftol': 1e-9, 'gtol': 1e-9}  # More iterations, stricter tolerances
                        )
                        
                        if result.success:
                            # Check if this result is better
                            current_radii = result.x[2::3]
                            current_sum = np.sum(current_radii)
                            if current_sum > best_value:
                                best_value = current_sum
                                best_result = result.x
                        break  # Break if successful
                    except:
                        continue
                        
            except:
                continue
        
        return best_result
    
    # Enhanced final validation with iterative improvement
    def validate_and_improve(circles):
        # First run a few iterations of local optimization
        circles_flat = circles.flatten()
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
        try:
            # Use multiple optimization methods with fallback
            methods = ['trust-constr', 'SLSQP', 'L-BFGS-B']
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        circles_flat,
                        method=method,
                        bounds=bounds,
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-10}  # Even more iterations, extremely strict tolerances
                    )
                    
                    if result.success:
                        circles = result.x.reshape(-1, 3)
                        break
                except:
                    continue
        except:
            pass
            
        # Final validation
        constraints = compute_constraints(circles)
        if np.any(constraints < -1e-6):
            # If constraints still violated significantly, apply a hard constraint correction
            # This is a simple but effective correction approach
            for i in range(n):
                # Correct boundary violations
                if circles[i, 0] < circles[i, 2]:
                    circles[i, 0] = circles[i, 2] + 0.001
                if circles[i, 0] > 1 - circles[i, 2]:
                    circles[i, 0] = 1 - circles[i, 2] - 0.001
                if circles[i, 1] < circles[i, 2]:
                    circles[i, 1] = circles[i, 2] + 0.001
                if circles[i, 1] > 1 - circles[i, 2]:
                    circles[i, 1] = 1 - circles[i, 2] - 0.001
                    
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
        
        # Strategy 3: Final improvement with trust-constr optimization
        circles = validate_and_improve(circles)
        
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
                method='trust-constr',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 3000, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if result.success:
                circles_optimized = result.x.reshape(-1, 3)
                return circles_optimized
        except:
            pass
            
        return circles


# EVOLVE-BLOCK-END
