# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
from numba import jit
import math
from deap import base, creator, tools, algorithms
import random

@jit(nopython=True)
def compute_distances_numba(positions):
    """Compute pairwise distances efficiently using numba"""
    n = positions.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a combination of mathematical insight, evolutionary algorithms, and advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Mathematical approach based on known circle packing patterns
    def initialize_mathematical_config():
        """Initialize using mathematical insights from circle packing theory"""
        # Start with a hexagonal packing pattern for maximum density
        positions = []
        radii = []
        
        # Create a systematic hexagonal grid with 6 rows and 6 columns
        rows, cols = 6, 6
        padding = 0.02  # Reduced padding for better utilization
        
        # Generate points in a hexagonal lattice pattern
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = (i % 2) * 0.5
                x = padding + (j + offset) * (1 - 2*padding) / (cols - 1)
                y = padding + i * (1 - 2*padding) / (rows - 1)
                
                # Ensure within bounds and add small random jitter
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Add small jitter for better distribution
                    x += np.random.normal(0, 0.002)
                    y += np.random.normal(0, 0.002)
                    x = np.clip(x, padding, 1-padding)
                    y = np.clip(y, padding, 1-padding)
                    positions.append([x, y])
            if len(positions) >= n:
                break
        
        # Fill remaining positions strategically
        if len(positions) < n:
            # Place additional circles near edges with some randomness
            edge_positions = []
            # Corner positions
            corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
            for corner in corners:
                if len(edge_positions) < n - len(positions):
                    edge_positions.append(corner)
            
            # Side positions - more systematic approach
            sides = []
            for _ in range(n - len(positions) - len(edge_positions)):
                side = np.random.randint(0, 4)
                if side == 0:  # top edge
                    sides.append((np.random.uniform(0.1, 0.9), 0.95))
                elif side == 1:  # bottom edge
                    sides.append((np.random.uniform(0.1, 0.9), 0.05))
                elif side == 2:  # left edge
                    sides.append((0.05, np.random.uniform(0.1, 0.9)))
                else:  # right edge
                    sides.append((0.95, np.random.uniform(0.1, 0.9)))
            
            positions.extend(edge_positions[:n-len(positions)])
            positions.extend(sides[:n-len(positions)-len(edge_positions)])
        
        positions = np.array(positions[:n])
        
        # Estimate initial radii more carefully using mathematical bounds
        tree = cKDTree(positions)
        radii = []
        for i in range(n):
            # Find nearest neighbors
            distances, indices = tree.query(positions[i], k=min(8, n), p=2)
            # Take the minimum distance to nearest neighbor divided by 2
            if len(distances) > 1:
                min_dist = np.min(distances[1:])  # exclude self-distance
                # Conservative estimate for better convergence
                radius = min(0.15, min_dist / 2.0 * 0.95)
                radii.append(max(0.01, radius))  # Ensure minimum radius
            else:
                radii.append(0.08)
        
        return positions, np.array(radii)
    
    # Enhanced constraint functions with better numerical handling
    def containment_constraints(circles_flat):
        """Ensure all circles are fully contained within unit square"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # Vectorized containment constraints
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        r_coords = radii
        
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # Adding a small epsilon to avoid numerical issues
        epsilon = 1e-8
        constraints = np.concatenate([
            x_coords - r_coords + epsilon,           # x - r >= 0
            1 - x_coords - r_coords + epsilon,       # 1 - x - r >= 0
            y_coords - r_coords + epsilon,           # y - r >= 0
            1 - y_coords - r_coords + epsilon        # 1 - y - r >= 0
        ])
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # More efficient non-overlap constraint computation
        distances = cdist(positions, positions, 'euclidean')
        radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Non-overlap constraints: distance >= (r_i + r_j) - epsilon for numerical stability
        # So we want: distances - radii_matrix >= -epsilon
        epsilon = 1e-8
        constraints = distances - radii_matrix + epsilon
        
        # Only keep upper triangle (avoid duplicates) and diagonal zeros
        mask = np.triu(np.ones_like(constraints), k=1).astype(bool)
        return constraints[mask]
    
    # Objective function (negative because we minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat.reshape(-1, 3)[:, 2])
    
    # Gradient of objective function
    def grad_objective(circles_flat):
        grad = np.zeros_like(circles_flat)
        grad[2::3] = -1.0  # gradient w.r.t. radii
        return grad
    
    # Custom constraint handling with improved numerical stability
    def safe_constraint_evaluator(circles_flat):
        """Evaluate constraints with better numerical handling"""
        # Check containment first
        containment = containment_constraints(circles_flat)
        
        # Then check non-overlap
        overlap = non_overlap_constraints(circles_flat)
        
        # Combine constraints (all must be >= 0)
        return np.concatenate([containment, overlap])
    
    # Enhanced evolutionary algorithm approach for better global search
    def enhanced_evolutionary_approach():
        """Use enhanced evolutionary algorithm to find better initial solutions"""
        # Create a more sophisticated evolutionary algorithm for circle packing
        toolbox = base.Toolbox()
        
        # Define individual representation: [x1, y1, r1, x2, y2, r2, ...]
        IND_SIZE = n * 3
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        def create_individual():
            # Create a valid individual with proper constraints
            positions, radii = initialize_mathematical_config()
            individual = []
            for i in range(n):
                individual.extend([positions[i, 0], positions[i, 1], radii[i]])
            return creator.Individual(individual)
        
        def evaluate(individual):
            # Convert individual to array
            circles_array = np.array(individual).reshape(-1, 3)
            
            # Check if all circles are valid
            for i in range(n):
                x, y, r = circles_array[i]
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    return (-1e8,)  # Invalid solution
            
            # Check overlaps with better tolerance
            positions = circles_array[:, :2]
            radii = circles_array[:, 2]
            distances = cdist(positions, positions, 'euclidean')
            radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Check if any circles are too close (overlap)
            # Use a small tolerance for numerical reasons
            tolerance = 1e-6
            overlap_matrix = distances < (radii_matrix - tolerance)
            overlap_count = np.sum(overlap_matrix) - n  # Subtract diagonal (self-overlaps)
            
            if overlap_count > 0:  # Serious overlap detected
                return (-1e8,)
            
            # Return negative sum of radii (since we're maximizing)
            return (-np.sum(radii),)
        
        def mutate(individual):
            # Mutate one element at a time with adaptive mutation rates
            for i in range(len(individual)):
                if random.random() < 0.15:  # Increased mutation rate
                    if i % 3 == 0:  # x coordinate
                        individual[i] += np.random.normal(0, 0.015)  # Smaller perturbation
                        individual[i] = np.clip(individual[i], 0.001, 0.999)
                    elif i % 3 == 1:  # y coordinate
                        individual[i] += np.random.normal(0, 0.015)  # Smaller perturbation
                        individual[i] = np.clip(individual[i], 0.001, 0.999)
                    else:  # radius
                        individual[i] += np.random.normal(0, 0.008)  # Smaller perturbation
                        individual[i] = np.clip(individual[i], 0.001, 0.499)
            return individual,
        
        def crossover(ind1, ind2):
            # Uniform crossover with better probability
            for i in range(len(ind1)):
                if random.random() < 0.6:  # Higher crossover probability
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=5)  # Larger tournament size
        
        # Run evolution with more generations
        population = toolbox.population(n=100)  # Increased population size
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.6, mutpb=0.3, 
                ngen=50, stats=stats, halloffame=hof, verbose=False
            )
            if hof:
                return np.array(hof[0]).reshape(-1, 3)
        except:
            pass
        
        return None
    
    # Improved optimization approach with better parameter tuning
    def improved_optimization(initial_circles):
        """Run improved optimization with multiple strategies"""
        # Set up bounds for variables (x, y, r for each circle)
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': lambda x: safe_constraint_evaluator(x)}
        ]
        
        # Use multiple optimization strategies for better results
        best_result = None
        best_sum = -np.inf
        
        # Strategy 1: SLSQP with multiple restarts and better initializations
        try:
            for restart in range(30):  # More restarts for better exploration
                np.random.seed(42 + restart)
                
                # Create better initial perturbations
                perturbed = initial_circles.copy()
                # Apply different perturbation patterns for each restart
                for i in range(n):
                    # Add moderate perturbations to positions
                    perturbed[i*3] += np.random.normal(0, 0.015)  # x
                    perturbed[i*3 + 1] += np.random.normal(0, 0.015)  # y
                    # Smaller perturbations to radii
                    perturbed[i*3 + 2] += np.random.normal(0, 0.005)  # r
                
                # Ensure bounds are respected
                for i in range(n):
                    perturbed[i*3] = np.clip(perturbed[i*3], 0.001, 0.999)
                    perturbed[i*3 + 1] = np.clip(perturbed[i*3 + 1], 0.001, 0.999)
                    perturbed[i*3 + 2] = np.clip(perturbed[i*3 + 2], 0.001, 0.499)
                
                # Try with different tolerance settings
                result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'eps': 1e-8},
                    callback=lambda x: None
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
        except Exception as e:
            pass
        
        # Strategy 2: Try Trust-Constr with even higher precision
        if best_result is None:
            try:
                result = minimize(
                    objective,
                    initial_circles,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 5000, 'gtol': 1e-12, 'xtol': 1e-12, 'disp': False},
                    callback=lambda x: None
                )
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception as e:
                pass
        
        # Strategy 3: Try L-BFGS-B with better initial values and tighter tolerances
        if best_result is None:
            try:
                result = minimize(
                    objective,
                    initial_circles,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12},
                    callback=lambda x: None
                )
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception as e:
                pass
        
        return best_result
    
    # Initial configuration using mathematical approach
    positions, radii = initialize_mathematical_config()
    initial_circles = np.column_stack([positions, radii]).flatten()
    
    # Try enhanced evolutionary approach first for better initial solution
    try:
        evolved_solution = enhanced_evolutionary_approach()
        if evolved_solution is not None:
            initial_circles = evolved_solution.flatten()
    except:
        pass
    
    # Run improved optimization
    best_result = improved_optimization(initial_circles)
    
    # Final fallback to initial configuration if all optimizations fail
    if best_result is not None:
        final_circles = best_result.x.reshape(-1, 3)
    else:
        final_circles = initial_circles.reshape(-1, 3)
    
    # Final validation and cleanup with stricter bounds
    validated_circles = []
    for i in range(n):
        x = max(0.001, min(0.999, final_circles[i, 0]))
        y = max(0.001, min(0.999, final_circles[i, 1]))
        r = max(0.001, min(0.499, final_circles[i, 2]))
        validated_circles.append([x, y, r])
    
    return np.array(validated_circles)


# EVOLVE-BLOCK-END
