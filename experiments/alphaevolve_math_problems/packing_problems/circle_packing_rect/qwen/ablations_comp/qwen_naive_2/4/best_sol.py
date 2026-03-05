# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import math
from itertools import combinations
import time
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
import random
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')
import cvxpy as cp

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses advanced optimization techniques to achieve better results than the benchmark.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    best_sum_radii = 0
    best_circles = None
    best_width = 1.0
    best_height = 1.0
    
    # Try different aspect ratios - optimized for circle packing
    # Focus on ratios that are likely to give better results for 21 circles
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.618, 1.8, 2.0, 2.2, 2.5, 3.0] 
    
    # Improved hexagonal packing approach with better grid calculation
    def create_hexagonal_initialization(width, height, n=21):
        """Create initial configuration using improved hexagonal packing"""
        # For 21 circles, try different patterns
        if n == 21:
            # Pattern: 5 rows with 4,5,4,4,4 circles respectively (total 21)
            rows = 5
            cols_per_row = [4, 5, 4, 4, 4]
        else:
            rows = 5
            cols_per_row = [4, 5, 4, 4, 4]
        
        # Calculate cell dimensions based on available space
        max_cols = max(cols_per_row)
        cell_width = width / max_cols
        cell_height = height / rows
        
        # Adjust spacing to better fit the rectangle
        cell_width = min(cell_width, width / max_cols)
        cell_height = min(cell_height, height / rows)
        
        # Use more precise packing
        if width / height > 2.0:  # Wide rectangle
            # Try a 4-row pattern instead
            rows = 4
            cols_per_row = [5, 5, 5, 6] if n == 21 else [4, 5, 4, 4]
        elif height / width > 2.0:  # Tall rectangle
            # Try a 6-row pattern
            rows = 6
            cols_per_row = [3, 4, 4, 4, 4, 4] if n == 21 else [4, 4, 4, 4, 4, 4]
        
        circles = []
        idx = 0
        
        for row in range(rows):
            cols = cols_per_row[row] if row < len(cols_per_row) else cols_per_row[-1]
            row_y = (row + 0.5) * cell_height
            
            # Offset every other row for hexagonal packing
            x_offset = (row % 2) * cell_width * 0.5
            
            for col in range(cols):
                if idx >= n:
                    break
                row_x = (col + 0.5) * cell_width + x_offset
                
                # Ensure we're within bounds with padding
                row_x = max(cell_width/2, min(width - cell_width/2, row_x))
                row_y = max(cell_height/2, min(height - cell_height/2, row_y))
                
                # Initial radius - start with smaller value to allow optimization to grow
                radius = min(cell_width, cell_height) * 0.12
                
                circles.append([row_x, row_y, radius])
                idx += 1
                
            if idx >= n:
                break
        
        # Fill remaining slots if needed with better random positions
        while len(circles) < n:
            x = np.random.uniform(0.05, width - 0.05)
            y = np.random.uniform(0.05, height - 0.05)
            # Use a more uniform distribution for radii
            max_radius = min(width, height) * 0.25
            radius = np.random.uniform(0.01, max_radius)
            circles.append([x, y, radius])
            
        return np.array(circles)
    
    # More efficient evaluation function with vectorized operations
    def evaluate_individual(individual):
        # Decode individual into circles: [x1, y1, r1, x2, y2, r2, ...]
        circles = np.array(individual).reshape(-1, 3)
        n = len(circles)
        
        # Extract parameters - use fixed rectangle for evaluation
        width = 1.0
        height = 1.0
        
        # Check if any circles are outside boundaries
        for i in range(n):
            x, y, r = circles[i]
            if x < r or x > width - r or y < r or y > height - r:
                return -1e12  # Severe penalty for invalid solutions
        
        # Compute total radii (negative because we want to maximize)
        total_radii = np.sum(circles[:, 2])
        
        # Penalty for overlaps using vectorized operations for efficiency
        penalty = 0
        if n > 1:
            # Vectorized distance computation
            coords = circles[:, :2]
            radii = circles[:, 2]
            
            # Calculate pairwise distances efficiently
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            # Create overlap matrix
            radius_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlaps = np.maximum(0, radius_sums - dists)
            
            # Apply penalty only for actual overlaps - more aggressive penalty
            # Use a more sophisticated penalty that penalizes more heavily when overlaps are larger
            overlap_penalty = np.sum(overlaps * (overlaps > 1e-8) * overlaps) * 100000
            penalty = overlap_penalty
        
        return total_radii - penalty
    
    # Enhanced GA implementation with better operators and parameters
    def run_enhanced_ga(width, height, ratio):
        try:
            # Create initial population with better starting points
            pop_size = 150  # Increase population size for better exploration
            population = []
            
            for _ in range(pop_size):
                # Start with hexagonal packing
                circles = create_hexagonal_initialization(width, height, 21)
                
                # Add more sophisticated random perturbations
                for i in range(len(circles)):
                    # Perturb position with larger variance initially
                    circles[i][0] += np.random.normal(0, width * 0.15)
                    circles[i][1] += np.random.normal(0, height * 0.15)
                    # Perturb radius with wider range but keep within reasonable bounds
                    circles[i][2] += np.random.normal(0, 0.06)
                    circles[i][2] = max(0.005, min(min(width, height) * 0.3, circles[i][2]))  # Keep positive and bounded
                
                # Flatten and add to population
                flat_circles = [val for circle in circles for val in circle]
                population.append(flat_circles)
            
            # Genetic Algorithm setup
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Better initialization that respects constraints
            def init_individual():
                circles = create_hexagonal_initialization(width, height, 21)
                # Perturb slightly with better distribution
                for i in range(len(circles)):
                    circles[i][0] += np.random.normal(0, width * 0.1)
                    circles[i][1] += np.random.normal(0, height * 0.1)
                    circles[i][2] += np.random.normal(0, 0.04)
                    circles[i][2] = max(0.005, min(min(width, height) * 0.3, circles[i][2]))
                return [val for circle in circles for val in circle]
            
            toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate_individual)
            
            # Use different crossover and mutation operators for better exploration
            toolbox.register("mate", tools.cxUniform, indpb=0.8)  # Increased crossover probability
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.04, indpb=0.4)  # Higher sigma for more exploration
            toolbox.register("select", tools.selTournament, tournsize=3)  # Even smaller tournament size for more diversity
            
            # Run GA with more generations and better parameters
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run with more generations to allow for better convergence
            population = toolbox.population(n=pop_size)
            result, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.9, mutpb=0.5, 
                ngen=250, stats=stats, halloffame=hof, verbose=False
            )
            
            if len(hof) > 0:
                best_individual = hof[0]
                circles = np.array(best_individual).reshape(-1, 3)
                current_sum = np.sum(circles[:, 2])
                return current_sum, circles
                
        except Exception as e:
            return 0, None
    
    # Improved local optimization with better constraints and more iterations
    def run_local_optimization(initial_circles, width, height):
        try:
            n = len(initial_circles)
            
            def objective(params):
                circles_flat = params.reshape(-1, 3)
                radii = circles_flat[:, 2]
                return -np.sum(radii)  # Negative because we want to maximize
            
            def constraint_func(params):
                circles_flat = params.reshape(-1, 3)
                constraints = []
                
                # Pairwise distance constraints (no overlaps) - more efficient version
                coords = circles_flat[:, :2]
                radii = circles_flat[:, 2]
                
                # Vectorized constraint calculation
                diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                
                # For each pair, ensure distance >= sum of radii
                for i in range(n):
                    for j in range(i+1, n):
                        dist = dists[i, j]
                        r1, r2 = radii[i], radii[j]
                        constraints.append(dist - (r1 + r2))  # Should be >= 0
                    
                # Boundary constraints
                for i in range(n):
                    x, y, r = circles_flat[i]
                    constraints.extend([
                        x - r,              # left boundary
                        width - x - r,      # right boundary
                        y - r,              # bottom boundary
                        height - y - r      # top boundary
                    ])
                
                return np.array(constraints)
            
            initial_params = initial_circles.flatten()
            cons = {'type': 'ineq', 'fun': constraint_func}
            bounds = [(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)] * n
            
            # Try multiple optimization methods with different settings
            methods = ['SLSQP', 'trust-constr']
            best_result = None
            best_value = -np.inf
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15},
                        tol=1e-15
                    )
                    
                    if result.success:
                        optimized_circles = result.x.reshape(-1, 3)
                        current_sum = np.sum(optimized_circles[:, 2])
                        if current_sum > best_value:
                            best_value = current_sum
                            best_result = optimized_circles
                except Exception:
                    continue
            
            return best_result if best_result is not None else initial_circles
            
        except Exception:
            return initial_circles
    
    # Enhanced convex optimization approach for better results
    def run_convex_optimization(width, height):
        """Use convex optimization for better results"""
        try:
            # Create initial configuration
            circles = create_hexagonal_initialization(width, height, 21)
            
            # Use cvxpy for convex optimization approach
            n = 21
            x = cp.Variable(n)
            y = cp.Variable(n)
            r = cp.Variable(n)
            
            # Objective: maximize sum of radii
            objective = cp.Maximize(cp.sum(r))
            
            # Constraints
            constraints = []
            
            # Circle boundaries
            for i in range(n):
                constraints.append(x[i] >= r[i])
                constraints.append(y[i] >= r[i])
                constraints.append(x[i] <= width - r[i])
                constraints.append(y[i] <= height - r[i])
            
            # No overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    # Distance between centers should be >= sum of radii
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    constraints.append(dist_sq >= (r[i] + r[j])**2)
            
            # Solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            
            if prob.status == cp.OPTIMAL:
                optimized_circles = np.zeros((n, 3))
                for i in range(n):
                    optimized_circles[i] = [x[i].value, y[i].value, r[i].value]
                return optimized_circles
            
        except Exception:
            pass
        return circles
    
    # Try different aspect ratios and optimization approaches
    for ratio in ratios:
        width = 2.0 / (1 + ratio)  # width + height = 2, and width/height = ratio
        height = 2.0 / (1 + 1/ratio)
        
        # Try enhanced GA approach
        ga_sum, ga_circles = run_enhanced_ga(width, height, ratio)
        
        if ga_sum > best_sum_radii:
            best_sum_radii = ga_sum
            best_circles = ga_circles.copy()
            best_width = width
            best_height = height
    
    # If GA didn't work well, try local optimization on best heuristic
    if best_circles is None or best_sum_radii < 2.0:
        # Use a better heuristic initialization with more refined patterns
        width = 1.0
        height = 1.0
        circles = create_hexagonal_initialization(width, height, 21)
        
        # Try multiple refinement attempts with different strategies
        for attempt in range(15):  # More attempts for better chance
            refined_circles = run_local_optimization(circles, width, height)
            current_sum = np.sum(refined_circles[:, 2])
            
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_circles = refined_circles.copy()
                best_width = width
                best_height = height
            
            # Slightly perturb for next attempt with smaller variance
            for i in range(len(circles)):
                circles[i][0] += np.random.normal(0, width * 0.03)
                circles[i][1] += np.random.normal(0, height * 0.03)
                circles[i][2] += np.random.normal(0, 0.02)
                circles[i][2] = max(0.005, min(min(width, height) * 0.3, circles[i][2]))
    
    # Try convex optimization approach as final refinement
    if best_circles is not None:
        try:
            convex_circles = run_convex_optimization(best_width, best_height)
            convex_sum = np.sum(convex_circles[:, 2])
            if convex_sum > best_sum_radii:
                best_sum_radii = convex_sum
                best_circles = convex_circles.copy()
        except Exception:
            pass
    
    # Final fallback to a more robust approach
    if best_circles is None:
        # Try a hybrid approach with better initialization
        best_sum_radii = 0
        best_circles = None
        
        # Try a few different configurations with more focus on promising ones
        configs = [
            (1.0, 1.0),  # Square
            (1.2, 1.8),  # Wide rectangle
            (1.8, 1.2),  # Tall rectangle  
            (1.5, 1.5),  # Nearly square
            (0.9, 1.1),  # Very close to square
            (1.1, 0.9),  # Very close to square (flipped)
            (1.3, 1.7),  # Another wide rectangle
            (1.7, 1.3),  # Another tall rectangle
            (2.0, 1.0),  # Very wide rectangle
            (1.0, 2.0),  # Very tall rectangle
        ]
        
        for width, height in configs:
            # Start with a better hexagonal packing
            circles = create_hexagonal_initialization(width, height, 21)
            
            # Refine with optimization
            refined_circles = run_local_optimization(circles, width, height)
            current_sum = np.sum(refined_circles[:, 2])
            
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_circles = refined_circles.copy()
                best_width = width
                best_height = height
    
    # If still no good solution, return the best heuristic
    if best_circles is None:
        # Use the best hexagonal packing approach
        width, height = 1.0, 1.0
        circles = create_hexagonal_initialization(width, height, 21)
        return circles
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
