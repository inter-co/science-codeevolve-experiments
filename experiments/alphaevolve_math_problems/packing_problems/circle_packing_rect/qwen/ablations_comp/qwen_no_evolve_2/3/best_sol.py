# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
import random
from typing import Tuple
import time
from itertools import combinations
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
from optuna import create_study, Trial
from optuna.samplers import TPESampler
import numba
from numba import jit

@jit(nopython=True)
def check_overlap_numba(circle1, circle2):
    """Fast overlap checking using numba"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx*dx + dy*dy
    return dist_sq < (r1 + r2)**2

@jit(nopython=True)
def check_bounds_numba(circle, width, height):
    """Fast boundary checking using numba"""
    x, y, r = circle
    return (x >= r and x <= width - r and y >= r and y <= height - r)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization, evolutionary algorithm, and local optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Try more aspect ratios for better chance of finding optimal
    ratios = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    # For performance, run a few trials with different parameters
    for ratio in ratios:
        # Determine width and height based on the ratio
        if ratio <= 1:
            width = 1.0
            height = 2.0 - width  # so width + height = 2
        else:
            height = 1.0
            width = 2.0 - height  # so width + height = 2
            
        # Multi-scale approach: start with better initialization
        circles = initialize_better(width, height, 21)
        
        # Apply evolutionary algorithm with enhanced parameters
        circles = evolutionary_optimization(circles, width, height, generations=200, pop_size=100)
        
        # Refine with local optimization
        circles = refine_circles(circles, width, height)
        
        # Calculate sum of radii
        total_radius = np.sum(circles[:, 2])
        
        if total_radius > best_sum:
            best_sum = total_radius
            best_result = circles.copy()
    
    return best_result if best_result is not None else generate_default_solution(1.0, 1.0, 21)

def initialize_better(width: float, height: float, n: int) -> np.ndarray:
    """Better initialization using hexagonal packing idea and improved clustering"""
    circles = np.zeros((n, 3))
    
    # Try to create a more dense initial configuration
    # Use a hexagonal lattice pattern for better density
    rows = 5
    cols = 5
    
    # Adjust spacing to account for hexagonal arrangement
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    # Hexagonal packing spacing factors
    hex_spacing_x = spacing_x * 0.866  # sqrt(3)/2
    hex_spacing_y = spacing_y * 0.75   # 3/4
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            # Offset every other row for hexagonal packing
            offset = (i % 2) * (hex_spacing_x / 2)
            x = spacing_x * (j + 1) + offset
            y = spacing_y * (i + 1)
            
            # Add more substantial randomization for diversity
            x += np.random.uniform(-hex_spacing_x/4, hex_spacing_x/4)
            y += np.random.uniform(-hex_spacing_y/4, hex_spacing_y/4)
            
            # Ensure within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Initial radius - based on available space and proximity to others
            # Calculate minimum distance to edges
            min_edge_dist = min(x, width - x, y, height - y)
            # For better initialization, also consider distance to nearby circles
            max_radius = min_edge_dist * 0.35  # Slightly smaller to allow room for optimization
            
            # Add more randomness to initial radius
            radius = max_radius * (0.6 + np.random.uniform(0, 0.7))
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n:
            break
    
    # Improve by clustering to distribute circles more evenly
    if n > 10:
        points = circles[:, :2]
        # Use fewer clusters for better distribution
        n_clusters = min(5, n//3) if n > 12 else 3
        if n_clusters > 0:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(points)
                
                # Adjust radii based on cluster density
                for i in range(len(clusters)):
                    cluster_id = clusters[i]
                    # Get points in same cluster
                    cluster_points = points[clusters == cluster_id]
                    if len(cluster_points) > 1:
                        # Compute average distance to other points in cluster
                        distances = [np.linalg.norm(points[i] - p) for p in cluster_points if not np.allclose(points[i], p)]
                        if distances:
                            avg_dist = np.mean(distances)
                            if avg_dist > 0.01:
                                # Reduce radius to allow for better packing
                                circles[i, 2] = min(circles[i, 2], avg_dist * 0.2)
            except:
                pass  # Fall back to original if clustering fails
    
    return circles

def evolutionary_optimization(circles: np.ndarray, width: float, height: float, generations: int = 200, pop_size: int = 100) -> np.ndarray:
    """Use evolutionary algorithm for global optimization with improved parameters"""
    # Define the optimization problem
    def evaluate(individual):
        # Convert individual back to circles array
        circles_array = np.array(individual).reshape(-1, 3)
        
        # Check constraints and calculate fitness
        if not check_all_constraints_fast(circles_array, width, height):
            return (-1000000,)  # Penalty for invalid solutions
        
        # Fitness is the sum of radii (we want to maximize)
        return (np.sum(circles_array[:, 2]),)
    
    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.001, width - 0.001)
    toolbox.register("attr_float_y", np.random.uniform, 0.001, height - 0.001)
    toolbox.register("attr_radius", np.random.uniform, 0.001, min(0.5, width/4, height/4))
    
    # Individual is [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_float_y, toolbox.attr_radius), n=21)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.6)  # Increased crossover probability
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.04, indpb=0.3)  # Increased mutation
    toolbox.register("select", tools.selTournament, tournsize=5)  # Larger tournament size
    
    # Create population with higher diversity and more generations
    pop = toolbox.population(n=pop_size)
    
    # Run evolution with more generations and better termination criteria
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.85, mutpb=0.35, 
                                          ngen=generations, stats=stats, halloffame=hof, verbose=False)
    except Exception as e:
        # Fallback to simpler approach if evolutionary algorithm fails
        return circles
    
    # Return the best individual found
    best_individual = hof[0]
    return np.array(best_individual).reshape(-1, 3)

def check_all_constraints_fast(circles: np.ndarray, width: float, height: float) -> bool:
    """Fast constraint checking using numba JIT compilation"""
    n = len(circles)
    
    # Check boundary constraints efficiently
    for i in range(n):
        if not check_bounds_numba(circles[i], width, height):
            return False
    
    # Check overlap constraints more efficiently using vectorized operations
    # Use early termination for performance
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap_numba(circles[i], circles[j]):
                return False
    
    return True

def optimize_circle_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circle positions using constrained optimization with better settings"""
    n = len(circles)
    
    # Flatten parameters: [x0, y0, r0, x1, y1, r1, ...]
    initial_params = circles.flatten()
    
    def objective(params):
        # Reconstruct circles
        reconstructed = params.reshape(-1, 3)
        # Objective: maximize sum of radii (minimize negative sum)
        return -np.sum(reconstructed[:, 2])
    
    def constraint_func(params):
        # Reconstruct circles
        reconstructed = params.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = reconstructed[i]
            # Ensure circles don't exceed boundaries
            constraints.extend([
                x - r,  # left boundary
                width - x - r,  # right boundary
                y - r,  # bottom boundary
                height - y - r  # top boundary
            ])
        
        # Non-overlap constraints - use more efficient approach
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = reconstructed[i]
                x2, y2, r2 = reconstructed[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Constraint: dist^2 >= (r1 + r2)^2
                constraints.append(dist_sq - (r1 + r2)**2)
        
        return np.array(constraints)
    
    # Create constraints dictionary
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Optimize with more iterations and better settings
    try:
        result = minimize(objective, initial_params, method='SLSQP', constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-9, 'eps': 1e-9})
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception as e:
        pass
    
    # Return original if optimization fails
    return circles

def refine_circles(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Refine circle configuration using multi-stage optimization"""
    refined = circles.copy()
    
    # Stage 1: Global optimization using constrained optimization with more iterations
    refined = optimize_circle_positions(refined, width, height)
    
    # Stage 2: Local refinement with boundary-aware adjustments - more aggressive
    for _ in range(50):  # More refinement passes
        # Adjust positions to avoid overlaps and respect boundaries
        for i in range(len(refined)):
            x, y, r = refined[i]
            
            # Keep within bounds
            x = max(r, min(width - r, x))
            y = max(r, min(height - r, y))
            
            # Adjust radius to maximize it while respecting constraints
            max_radius = min(x, width - x, y, height - y)
            
            # Check overlaps with other circles more efficiently
            new_radius = max_radius
            for j in range(len(refined)):
                if i != j:
                    x2, y2, r2 = refined[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist > 0:
                        # Maximum radius without overlapping this circle
                        max_radius_for_this = dist - r2
                        new_radius = min(new_radius, max_radius_for_this)
            
            # Ensure positive radius
            new_radius = max(0.001, new_radius)
            refined[i] = [x, y, new_radius]
    
    # Stage 3: Final validation and adjustment with more aggressive overlap resolution
    refined = validate_and_correct(refined, width, height)
    
    return refined

def validate_and_correct(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Ensure all constraints are satisfied with better overlap resolution"""
    corrected_circles = circles.copy()
    
    # First, handle boundary violations
    for i in range(len(corrected_circles)):
        x, y, r = corrected_circles[i]
        # Correct positions that violate boundaries
        corrected_circles[i, 0] = max(r, min(width - r, x))
        corrected_circles[i, 1] = max(r, min(height - r, y))
    
    # Then resolve overlaps through iterative correction - more iterations
    max_iterations = 500
    for _ in range(max_iterations):
        # Find all overlaps
        overlaps = []
        for i in range(len(corrected_circles)):
            for j in range(i+1, len(corrected_circles)):
                x1, y1, r1 = corrected_circles[i]
                x2, y2, r2 = corrected_circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < (r1 + r2):
                    overlaps.append((i, j, distance, r1 + r2))
        
        if not overlaps:
            break
            
        # Resolve the most severe overlap first
        overlaps.sort(key=lambda x: x[3] - x[2])  # Sort by overlap amount
        i, j, dist, sum_radii = overlaps[-1]
        
        # Push circles apart along the line connecting centers
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0.001:  # Avoid division by zero
            push_amount = (sum_radii - distance) / 2
            dx_norm = dx / distance
            dy_norm = dy / distance
            
            # Move both circles away from each other with more aggressive pushing
            corrected_circles[i, 0] -= dx_norm * push_amount * 2.0
            corrected_circles[i, 1] -= dy_norm * push_amount * 2.0
            corrected_circles[j, 0] += dx_norm * push_amount * 2.0
            corrected_circles[j, 1] += dy_norm * push_amount * 2.0
            
            # Keep within bounds
            corrected_circles[i, 0] = max(corrected_circles[i, 2], 
                                        min(width - corrected_circles[i, 2], 
                                            corrected_circles[i, 0]))
            corrected_circles[i, 1] = max(corrected_circles[i, 2], 
                                        min(height - corrected_circles[i, 2], 
                                            corrected_circles[i, 1]))
            corrected_circles[j, 0] = max(corrected_circles[j, 2], 
                                        min(width - corrected_circles[j, 2], 
                                            corrected_circles[j, 0]))
            corrected_circles[j, 1] = max(corrected_circles[j, 2], 
                                        min(height - corrected_circles[j, 2], 
                                            corrected_circles[j, 1]))
    
    return corrected_circles

def generate_default_solution(width: float, height: float, n: int) -> np.ndarray:
    """Fallback solution if optimization fails"""
    circles = np.zeros((n, 3))
    
    # Better grid approach with adaptive spacing
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = width / (grid_size + 1)
    spacing_y = height / (grid_size + 1)
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            x = spacing_x * (i + 1)
            y = spacing_y * (j + 1)
            # Use larger initial radius for default solution
            radius = min(spacing_x, spacing_y) * 0.35
            
            # Ensure it's within bounds
            x = max(radius, min(width - radius, x))
            y = max(radius, min(height - radius, y))
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n:
            break
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
