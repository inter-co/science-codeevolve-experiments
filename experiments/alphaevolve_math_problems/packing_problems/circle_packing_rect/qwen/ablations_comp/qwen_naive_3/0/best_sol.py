# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import Voronoi
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial
import time
from sklearn.cluster import KMeans
from scipy.spatial import distance
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses a hybrid approach combining geometric initialization, advanced optimization, and evolutionary algorithms.
    """
    n = 21
    # Rectangle perimeter = 4 means width + height = 2
    # Try different aspect ratios to find the best one for packing
    best_aspect_ratio = 1.0
    best_radius_sum = 0
    best_config = None
    
    # Test several aspect ratios
    test_ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]
    
    for ratio in test_ratios:
        width = 2 / (1 + 1/ratio)
        height = 2 - width
        
        # Better initialization using hexagonal packing approach
        positions = []
        radii = []
        
        # Try different grid patterns for better packing
        # Use a more systematic approach - start with a hexagonal pattern
        rows = 5
        cols = 5
        
        # Calculate spacing to fit in the rectangle
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Calculate optimal radius based on available space
        max_radius = min(width, height) / 6  # Conservative estimate
        
        # Create hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + 1) * spacing_x + offset * spacing_x/2
                y = (i + 1) * spacing_y
                
                if x <= width and y <= height:
                    positions.append([x, y])
                    radii.append(max_radius)
                    
        if len(positions) >= n:
            # Check if configuration is valid and compute sum
            valid_positions = np.array(positions[:n])
            valid_radii = np.array(radii[:n])
            
            # Check constraints manually
            valid = True
            for i in range(n):
                # Boundary check
                if (valid_positions[i][0] - valid_radii[i] < 0 or 
                    valid_positions[i][0] + valid_radii[i] > width or
                    valid_positions[i][1] - valid_radii[i] < 0 or 
                    valid_positions[i][1] + valid_radii[i] > height):
                    valid = False
                    break
                    
            if valid:
                # Check overlaps
                distances = cdist(valid_positions, valid_positions)
                for i in range(n):
                    for j in range(i+1, n):
                        min_distance = valid_radii[i] + valid_radii[j]
                        actual_distance = distances[i, j]
                        if actual_distance < min_distance:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    radius_sum = np.sum(valid_radii)
                    if radius_sum > best_radius_sum:
                        best_radius_sum = radius_sum
                        best_aspect_ratio = ratio
                        best_config = (valid_positions, valid_radii)
    
    # If no good configuration found, fall back to a better initialization
    if best_config is None:
        # Use a more systematic approach with better clustering
        width = 2 / (1 + 1/best_aspect_ratio)
        height = 2 - width
        
        # Generate points using k-means clustering to get well-distributed points
        # First generate many candidate points
        candidate_points = []
        for i in range(2000):  # More candidates for better distribution
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            candidate_points.append([x, y])
        
        candidate_points = np.array(candidate_points)
        
        # Use k-means to find good cluster centers
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=20)  # More init attempts
        kmeans.fit(candidate_points)
        initial_centers = kmeans.cluster_centers_
        
        # Assign radii based on proximity to neighbors
        initial_radii = []
        distances = cdist(initial_centers, initial_centers)
        
        for i in range(n):
            # Find minimum distance to other centers
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    min_dist = min(min_dist, distances[i, j])
            
            # Set radius to half the minimum distance to nearest neighbor, but bounded
            radius = min(min_dist/2.0, min(width, height)/6.0)
            radius = max(0.001, radius)  # Ensure positive
            initial_radii.append(radius)
        
        best_config = (initial_centers, np.array(initial_radii))
    
    initial_positions, initial_radii = best_config
    width = 2 / (1 + 1/best_aspect_ratio)
    height = 2 - width
    
    # Flatten all parameters into one array: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
    
    def objective(params):
        # Extract positions and radii
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(radii)
    
    def constraint_placement(params):
        """Ensure all circles are within rectangle bounds"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Circle center must be at least radius away from edges
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Boundary constraints: center must be at least radius away from each edge
        left_constraint = x_coords - radii
        right_constraint = width - (x_coords + radii)
        bottom_constraint = y_coords - radii
        top_constraint = height - (y_coords + radii)
        
        return np.concatenate([left_constraint, right_constraint, bottom_constraint, top_constraint])
    
    def constraint_overlap(params):
        """Ensure no overlapping circles"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Compute pairwise distances efficiently
        distances = cdist(positions, positions)
        constraints = []
        
        # Only check pairs where first index < second index to avoid duplicates
        for i in range(n):
            for j in range(i+1, n):
                min_distance = radii[i] + radii[j]
                actual_distance = distances[i, j]
                constraints.append(actual_distance - min_distance)
        
        return np.array(constraints)
    
    def constraint_positive_radii(params):
        """Ensure all radii are positive"""
        radii = params[2*n:]
        return radii
    
    # Set up bounds
    bounds = []
    # Position bounds: [0, width] for x, [0, height] for y
    for i in range(n):
        bounds.append((0, width))   # x coordinate
        bounds.append((0, height))  # y coordinate
    
    # Radius bounds: [0.001, max possible radius] 
    max_radius = min(width, height) / 2
    for i in range(n):
        bounds.append((0.001, max_radius))
    
    # Constraint dictionaries
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_placement(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_positive_radii(x)}
    ]
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = -np.inf
    
    # First try with SLSQP - more robust than before
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-10, 'eps': 1e-8}
        )
        
        if result.success:
            final_radii = result.x[2*n:]
            current_sum = np.sum(final_radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result
    except Exception as e:
        pass
    
    # If first attempt failed or didn't improve much, try with different method
    if best_result is None or best_sum < 2.0:  # If we're not getting good results, try something else
        # Use L-BFGS-B which might handle this better
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success:
                final_radii = result.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            pass
    
    # If still no success or poor results, try evolutionary algorithm approach with better parameters
    if best_result is None or best_sum < 2.1:
        # Evolutionary algorithm approach for better global search
        def evaluate_individual(individual):
            # Convert individual to positions and radii
            positions = np.array(individual[:2*n]).reshape(-1, 2)
            radii = np.array(individual[2*n:])
            
            # Check constraints
            # Boundary constraints
            boundary_violations = 0
            for i in range(n):
                if positions[i][0] - radii[i] < 0 or positions[i][0] + radii[i] > width:
                    boundary_violations += 1
                if positions[i][1] - radii[i] < 0 or positions[i][1] + radii[i] > height:
                    boundary_violations += 1
            
            # Overlap constraints
            overlap_violations = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                    if dist < (radii[i] + radii[j]):
                        overlap_violations += 1
            
            # If constraints violated, penalize heavily
            if boundary_violations > 0 or overlap_violations > 0:
                penalty = 10000 * (boundary_violations + overlap_violations)
                return -(np.sum(radii) - penalty)
            
            return -np.sum(radii)  # Negative because we want to maximize
        
        # Create DEAP individuals
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.001, max_radius)
        toolbox.register("attr_pos_x", random.uniform, 0, width)
        toolbox.register("attr_pos_y", random.uniform, 0, height)
        
        # Individual is [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
        def create_individual():
            individual = []
            for _ in range(n):
                individual.extend([toolbox.attr_pos_x(), toolbox.attr_pos_y(), toolbox.attr_float()])
            return creator.Individual(individual)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=max_radius/20, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=5)
        
        # Create population and run evolution with more generations
        try:
            population = toolbox.population(n=150)  # Larger population
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run evolution for more generations
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.7, mutpb=0.3, 
                ngen=100, stats=stats, halloffame=hof, verbose=False
            )
            
            if len(hof) > 0:
                best_evolutionary = hof[0]
                # Convert back to our format
                best_positions = np.array(best_evolutionary[:2*n]).reshape(-1, 2)
                best_radii = np.array(best_evolutionary[2*n:])
                current_sum = np.sum(best_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    # Create a result object similar to scipy minimize
                    class Result:
                        def __init__(self, x, success=True):
                            self.x = x
                            self.success = success
                    best_result = Result(np.array(best_evolutionary))
        except Exception as e:
            pass
    
    # If still no success, fallback to our initial configuration
    if best_result is None:
        circles = np.column_stack([initial_positions, initial_radii])
        return circles
    
    # Extract final solution
    final_positions = best_result.x[:2*n].reshape(-1, 2)
    final_radii = best_result.x[2*n:]
    
    # Create output array
    circles = np.column_stack([final_positions, final_radii])
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
