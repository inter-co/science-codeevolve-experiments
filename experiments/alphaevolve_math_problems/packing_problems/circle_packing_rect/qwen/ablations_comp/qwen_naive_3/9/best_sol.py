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
    
    # Test several aspect ratios with more systematic approach
    test_ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]
    
    for ratio in test_ratios:
        width = 2 / (1 + 1/ratio)
        height = 2 - width
        
        # Improved initialization using more sophisticated packing approach
        positions = []
        radii = []
        
        # Use a more intelligent grid pattern with better spacing
        # Try to use approximately sqrt(21) = 4.59 grid size
        rows = 5
        cols = 5
        
        # Calculate spacing with better boundary handling
        spacing_x = width / (cols + 1) if cols > 0 else width / 2
        spacing_y = height / (rows + 1) if rows > 0 else height / 2
        
        # Calculate optimal radius based on available space
        max_radius = min(width, height) / 6  # Conservative estimate
        
        # Create a hexagonal-like pattern with better distribution
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for better packing
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + 1) * spacing_x + offset * spacing_x/2
                y = (i + 1) * spacing_y
                
                # Add small randomization to prevent perfect grid artifacts
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                
                # Make sure point is within bounds
                if 0 <= x <= width and 0 <= y <= height:
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
    
    # If no good configuration found, fall back to better initialization
    if best_config is None:
        # Use a more sophisticated approach with better spatial distribution
        width = 2 / (1 + 1/best_aspect_ratio)
        height = 2 - width
        
        # Generate points using better spatial sampling - golden spiral or better clustering
        # Use a combination of grid and random points with clustering to improve distribution
        grid_points = []
        grid_size = int(math.ceil(math.sqrt(n)))
        
        # Create a structured grid with some randomness
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_points) >= n:
                    break
                x = (j + 1) * width / (grid_size + 1) + np.random.uniform(-width/(4*(grid_size+1)), width/(4*(grid_size+1)))
                y = (i + 1) * height / (grid_size + 1) + np.random.uniform(-height/(4*(grid_size+1)), height/(4*(grid_size+1)))
                if 0 <= x <= width and 0 <= y <= height:
                    grid_points.append([x, y])
        
        # If we don't have enough points, fill with random ones
        while len(grid_points) < n:
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            grid_points.append([x, y])
        
        positions = np.array(grid_points[:n])
        
        # Assign radii based on proximity to neighbors with better calculation
        initial_radii = []
        distances = cdist(positions, positions)
        
        for i in range(n):
            # Find minimum distance to other centers (excluding self)
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    min_dist = min(min_dist, distances[i, j])
            
            # Set radius to a fraction of the minimum distance to nearest neighbor, with bounds
            # Use a more aggressive but safe approach
            radius = min(min_dist/3.0, min(width, height)/3.0)
            radius = max(0.001, radius)  # Ensure positive
            initial_radii.append(radius)
        
        best_config = (positions, np.array(initial_radii))
    
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
    
    # First try with SLSQP with better parameters
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-10, 'eps': 1e-7, 'iprint': 0}
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
    if best_result is None or best_sum < 2.2:  # If we're not getting good results, try something else
        # Use L-BFGS-B which might handle this better with more iterations
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-7}
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
    if best_result is None or best_sum < 2.3:
        # Evolutionary algorithm approach for better global search with more generations
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
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=max_radius/10, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population and run evolution with more generations and better parameters
        try:
            population = toolbox.population(n=200)  # Larger population
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run evolution for more generations with better parameters
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.8, mutpb=0.3, 
                ngen=150, stats=stats, halloffame=hof, verbose=False
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
    
    # If still no success, fallback to our initial configuration but with refinement
    if best_result is None:
        # Try a simple refinement step on the initial configuration
        try:
            # Simple gradient-based refinement on initial solution
            refined_params = initial_params.copy()
            
            # Try a few iterations of gradient descent on the radii only
            for iteration in range(30):  # More iterations
                # For simplicity, just try to increase radii where there's room
                positions = refined_params[:2*n].reshape(-1, 2)
                radii = refined_params[2*n:].copy()
                
                # Compute current overlaps and adjust radii accordingly
                distances = cdist(positions, positions)
                new_radii = radii.copy()
                
                for i in range(n):
                    # Find minimum distance to neighbors
                    min_dist = float('inf')
                    for j in range(n):
                        if i != j:
                            min_dist = min(min_dist, distances[i, j])
                    
                    # Try to increase radius up to the limit without overlap
                    max_allowed_radius = min(min_dist / 2.0, max_radius)
                    if max_allowed_radius > radii[i]:
                        new_radii[i] = max(radii[i], min(max_allowed_radius, max_radius))
                
                refined_params[2*n:] = new_radii
                if np.allclose(radii, new_radii, rtol=1e-6):
                    break  # No significant change
            
            final_positions = refined_params[:2*n].reshape(-1, 2)
            final_radii = refined_params[2*n:]
            circles = np.column_stack([final_positions, final_radii])
            return circles
        except Exception:
            # Final fallback to initial config
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
