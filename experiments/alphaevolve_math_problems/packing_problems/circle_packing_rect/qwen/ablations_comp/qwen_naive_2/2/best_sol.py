# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List
import random
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import additional optimization libraries for better performance
try:
    import nevergrad as ng
    HAS_NEVERGRAD = True
except ImportError:
    HAS_NEVERGRAD = False

try:
    from skopt import gp_minimize
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# Import evolutionary optimization library
try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses an evolutionary approach combined with local optimization for better results.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Start timing
    start_time = time.time()
    
    # Try different rectangle dimensions to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Test several width/height combinations - focus on ratios that work well for circle packing
    ratios = [0.5, 0.7, 0.8, 1.0, 1.25, 1.5, 1.618, 2.0, 2.5, 3.0]
    
    # Also test some extreme ratios to explore more possibilities
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Initialize with better heuristic - use a more sophisticated approach
        circles = initialize_smart_placement(width, height, 21)
        
        # Optimize using a more efficient approach
        optimized_circles = optimize_efficiently(circles, width, height, start_time)
        
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized_circles
            
        # Early stopping if we're running close to time limit
        if time.time() - start_time > 55:
            break
    
    # If no good solution found, fallback to simple initialization
    if best_result is None:
        best_result = initialize_smart_placement(1.0, 1.0, 21)
    
    return best_result


def initialize_smart_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a smart approach based on known packing patterns."""
    circles = np.zeros((n, 3))
    
    # Use a more sophisticated approach: try different packing strategies
    if n <= 16:
        # For small numbers, use grid-based packing
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Adjust if we have fewer circles than grid capacity
        actual_n = min(n, rows * cols)
        
        # Calculate spacing
        cell_width = width / cols if cols > 0 else width
        cell_height = height / rows if rows > 0 else height
        
        # Place circles in a grid pattern with slight offset for better packing
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= actual_n:
                    break
                # Offset every other row for better packing
                x_offset = (j + 0.5) * cell_width
                if i % 2 == 1:
                    x_offset += cell_width / 2
                    
                y_offset = (i + 0.5) * cell_height
                
                # Ensure positions are within bounds with margin
                x = max(0.01, min(width - 0.01, x_offset))
                y = max(0.01, min(height - 0.01, y_offset))
                
                # Estimate initial radius based on available space
                max_radius = min(cell_width, cell_height) / 3
                
                # Make radius depend on how far from the boundary
                # Circles in center should have larger radii
                center_x = width / 2
                center_y = height / 2
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt((width/2)**2 + (height/2)**2)
                
                # Radius decreases with distance from center (but not too much)
                radius_factor = 0.8 + 0.2 * (1 - dist_from_center / max_dist)
                radius = max_radius * radius_factor
                
                # Ensure minimum radius
                radius = max(0.01, radius)
                circles[idx] = [x, y, radius]
                idx += 1
    else:
        # For larger numbers, use a combination approach
        # First place circles in a hexagonal pattern for most of them
        hex_rows = 4
        hex_cols = 6
        
        # Adjust if we have fewer circles than grid capacity
        hex_count = min(n, hex_rows * hex_cols)
        
        # Calculate spacing
        cell_width = width / hex_cols if hex_cols > 0 else width
        cell_height = height / hex_rows if hex_rows > 0 else height
        
        # Place circles in a hexagonal pattern
        idx = 0
        for i in range(hex_rows):
            for j in range(hex_cols):
                if idx >= hex_count:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (j + 0.5) * cell_width
                if i % 2 == 1:
                    x_offset += cell_width / 2
                    
                y_offset = (i + 0.5) * cell_height
                
                # Ensure positions are within bounds with margin
                x = max(0.01, min(width - 0.01, x_offset))
                y = max(0.01, min(height - 0.01, y_offset))
                
                # Estimate initial radius
                max_radius = min(cell_width, cell_height) / 3
                
                # Make radius depend on how far from the boundary
                center_x = width / 2
                center_y = height / 2
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt((width/2)**2 + (height/2)**2)
                
                # Radius decreases with distance from center (but not too much)
                radius_factor = 0.8 + 0.2 * (1 - dist_from_center / max_dist)
                radius = max_radius * radius_factor
                
                # Ensure minimum radius
                radius = max(0.01, radius)
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Fill remaining circles with a radial approach
        if idx < n:
            center_x = width / 2
            center_y = height / 2
            # For remaining circles, use a more strategic radial placement
            max_radius = min(width, height) / 10
            
            # Place in a circular pattern with increasing radius
            for i in range(idx, n):
                # Distribute remaining circles in a circular pattern
                angle = 2 * np.pi * i / n  # Even distribution
                # Use a spiral pattern to avoid clustering
                distance = 0.3 * min(width, height) * (1 + 0.3 * (i - idx) / (n - idx))
                
                x = center_x + distance * np.cos(angle)
                y = center_y + distance * np.sin(angle)
                
                # Keep within bounds with margin
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Use smaller radius for remaining circles
                circles[i] = [x, y, max_radius * 0.8]
    
    return circles


def optimize_efficiently(initial_circles: np.ndarray, width: float, height: float, start_time: float) -> np.ndarray:
    """Use a more efficient optimization approach with better constraint handling."""
    n = len(initial_circles)
    
    # Try multiple strategies to find the best solution
    results = []
    
    # Strategy 1: Evolutionary algorithm approach for better exploration
    try:
        optimized = optimize_with_evolutionary(initial_circles, width, height)
        results.append(optimized)
    except Exception as e:
        pass
    
    # Strategy 2: Try global optimization with better settings
    try:
        optimized = optimize_with_global_methods(initial_circles, width, height)
        results.append(optimized)
    except Exception:
        pass
    
    # Strategy 3: Improved local optimization with better restarts
    try:
        optimized = optimize_with_improved_local_searches(initial_circles, width, height)
        results.append(optimized)
    except Exception:
        pass
    
    # Strategy 4: Direct optimization with tighter tolerances and better approach
    try:
        optimized = direct_optimization_better(initial_circles, width, height)
        results.append(optimized)
    except Exception:
        pass
    
    # Select best result
    if results:
        best_result = max(results, key=lambda x: np.sum(x[:, 2]))
        return best_result
    else:
        return initial_circles


def optimize_with_evolutionary(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Use evolutionary algorithm for better exploration of the solution space."""
    if not HAS_DEAP:
        return initial_circles
        
    n = len(initial_circles)
    
    # Define individual and population structures
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define bounds for each parameter (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # For each circle: [x, y, r] 
        bounds.extend([
            (1e-6, width - 1e-6),     # x coordinate
            (1e-6, height - 1e-6),    # y coordinate  
            (1e-6, min(width, height)/2)  # radius
        ])
    
    # Generate initial population
    def init_individual():
        ind = []
        for i in range(n):
            x = np.random.uniform(1e-6, width - 1e-6)
            y = np.random.uniform(1e-6, height - 1e-6)
            r = np.random.uniform(1e-6, min(width, height)/2)
            ind.extend([x, y, r])
        return creator.Individual(ind)
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        """Evaluate fitness of an individual (sum of radii)"""
        circles = np.array(individual).reshape(-1, 3)
        
        # Check boundary constraints
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints: x-r >= 0, width-x-r >= 0, y-r >= 0, height-y-r >= 0
        if np.any(x - r < 0) or np.any(width - x - r < 0) or \
           np.any(y - r < 0) or np.any(height - y - r < 0):
            return -np.inf  # Invalid solution
        
        # Check overlap constraints
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                # Penalty for overlap (negative value when overlapping)
                if dist < r1 + r2:
                    penalty += (r1 + r2 - dist) * 1000  # Large penalty for overlaps
        
        # Return sum of radii minus penalties
        return np.sum(circles[:, 2]) - penalty
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolutionary algorithm
    try:
        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                          ngen=30, stats=stats, halloffame=hof, 
                                          verbose=False)
        
        if len(hof) > 0:
            return np.array(hof[0]).reshape(-1, 3)
    except Exception:
        pass
    
    return initial_circles


def optimize_with_global_methods(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Try global optimization methods that might be more effective."""
    n = len(initial_circles)
    
    def objective(params):
        circles = params.reshape(-1, 3)
        # Minimize negative sum of radii (maximize sum of radii)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Vectorized constraint checking for bounds
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints: x-r >= 0, width-x-r >= 0, y-r >= 0, height-y-r >= 0
        bound_constraints = np.concatenate([
            x - r,                    # x - r >= 0
            width - x - r,            # width - x - r >= 0  
            y - r,                    # y - r >= 0
            height - y - r            # height - y - r >= 0
        ])
        
        # Overlap constraints - compute all pairwise distances efficiently
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create overlap constraints for all pairs
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                # We want dist >= r1 + r2, so dist - r1 - r2 >= 0
                overlap_constraints.append(dist - r1 - r2)
        
        return np.concatenate([bound_constraints, overlap_constraints])
    
    # Create bounds for optimization - more careful about bounds
    bounds = []
    for i in range(n):
        # x bounds: [r, width-r] - make sure r is small enough to fit in bounds
        x_min = 1e-6
        x_max = width - 1e-6
        y_min = 1e-6
        y_max = height - 1e-6
        r_max = min(width, height) / 2 - 1e-6
        
        bounds.extend([(x_min, x_max), (y_min, y_max), (1e-6, r_max)])
    
    # Define constraints for optimization
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Try different global optimization methods
    try:
        # Try differential evolution with better parameters
        result = differential_evolution(
            objective,
            bounds,
            constraints=cons,
            seed=42,
            maxiter=100,      # Reduced iterations for speed but still sufficient
            popsize=20,       # Smaller population for faster execution
            mutation=(0.5, 1),
            recombination=0.9,
            atol=1e-8,
            tol=1e-8,
            workers=1,
            strategy='best1bin'
        )
        
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Fallback to simpler optimization
    return initial_circles


def optimize_with_improved_local_searches(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Run improved local searches with better strategies."""
    n = len(initial_circles)
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Vectorized boundary constraints
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints
        bound_constraints = np.concatenate([
            x - r,                    # x - r >= 0
            width - x - r,            # width - x - r >= 0  
            y - r,                    # y - r >= 0
            height - y - r            # height - y - r >= 0
        ])
        
        # Overlap constraints using distance matrix
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create overlap constraints for all pairs
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                overlap_constraints.append(dist - r1 - r2)
        
        return np.concatenate([bound_constraints, overlap_constraints])
    
    # Create bounds
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
    
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Try multiple local search strategies with different starting points
    best_result = initial_circles.copy()
    best_value = np.sum(initial_circles[:, 2])
    
    # Strategy 1: Multiple restarts with perturbed initial points
    for restart in range(5):  # More restarts for better chance of finding good solution
        # Add random perturbations to initial solution
        perturbed = initial_circles.copy()
        for i in range(n):
            # Perturb position with different magnitude
            perturbed[i, 0] += (np.random.random() - 0.5) * 0.05 * width
            perturbed[i, 1] += (np.random.random() - 0.5) * 0.05 * height
            # Keep within bounds
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0.01, width - 0.01)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0.01, height - 0.01)
        
        try:
            result = minimize(
                objective,
                perturbed.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerances
            )
            
            if result.success:
                current_value = -result.fun
                if current_value > best_value:
                    best_value = current_value
                    best_result = result.x.reshape(-1, 3)
        except Exception:
            continue
    
    return best_result


def direct_optimization_better(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Better direct optimization approach with smarter constraint handling."""
    n = len(initial_circles)
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Vectorized boundary constraints
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary constraints
        bound_constraints = np.concatenate([
            x - r,                    # x - r >= 0
            width - x - r,            # width - x - r >= 0  
            y - r,                    # y - r >= 0
            height - y - r            # height - y - r >= 0
        ])
        
        # Overlap constraints using distance matrix
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create overlap constraints for all pairs
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r1 = circles[i, 2]
                r2 = circles[j, 2]
                overlap_constraints.append(dist - r1 - r2)
        
        return np.concatenate([bound_constraints, overlap_constraints])
    
    # Create bounds
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
    
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Run optimization with better parameters
    try:
        result = minimize(
            objective,
            initial_circles.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerances
        )
        
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
