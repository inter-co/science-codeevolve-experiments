# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.optimize import differential_evolution
import random
from sklearn.cluster import KMeans
import time
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses evolutionary algorithms with improved initialization and constraint handling.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up optimization parameters
    n = 21
    max_time = 55  # Leave some buffer for final processing
    start_time = time.time()
    
    # Try different rectangle dimensions to find optimal aspect ratio
    best_sum = 0
    best_circles = None
    
    # Focus on promising aspect ratios that have shown good results
    aspect_ratios = [(1.5, 0.5), (2.0, 0.5), (1.2, 0.8), (0.8, 1.2), (1.0, 1.0), (1.8, 0.7), (0.7, 1.8), (1.3, 0.7)]
    
    # Precompute some good aspect ratios from known circle packing results
    for width_ratio, height_ratio in aspect_ratios:
        if time.time() - start_time > max_time:
            break
            
        width = 2 * width_ratio / (width_ratio + height_ratio)
        height = 2 * height_ratio / (width_ratio + height_ratio)
        
        # Improved initialization using a more systematic approach
        circles = np.zeros((n, 3))
        
        # Create a more intelligent initial pattern based on hexagonal packing principles
        # Try a 4x6 grid pattern with proper spacing
        rows = 4
        cols = 6
        
        grid_width = width / cols if cols > 0 else width
        grid_height = height / rows if rows > 0 else height
        
        # Create a hexagonal pattern with offset rows
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5) * grid_width
                if row % 2 == 1:
                    x += grid_width * 0.5
                y = (row + 0.5) * grid_height
                
                # Apply boundary checks with margin
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Initial radius based on available space with better calculation
                # Make sure we don't put circles too close to edges
                r = min(x, width - x, y, height - y) * 0.35
                r = max(0.01, min(r, min(width, height) * 0.35))
                circles[count] = [x, y, r]
                count += 1
        
        # Fill remaining circles with better initial placement
        for i in range(count, n):
            # Place in a more strategic way rather than purely random
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            # Make sure radius is reasonable relative to position
            r = min(x, width - x, y, height - y) * 0.25
            r = max(0.01, min(r, min(width, height) * 0.3))
            circles[i] = [x, y, r]
        
        # Define constraint function for optimization
        def constraint_func(params):
            # params: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            # Distance constraint: no overlap
            distances = cdist(positions, positions)
            constraints = []
            
            # Non-overlap constraints (distance >= sum of radii)
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    # We want dist >= min_dist, so constraint is (dist - min_dist) >= 0
                    constraints.append(dist - min_dist)
            
            # Boundary constraints - ensure circles are within rectangle
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                constraints.append(x - r)  # left boundary
                constraints.append(width - x - r)  # right boundary
                constraints.append(y - r)  # bottom boundary
                constraints.append(height - y - r)  # top boundary
                
            return np.array(constraints)
        
        # Objective function to maximize (negative because minimize)
        def objective(params):
            radii = params.reshape(-1, 3)[:, 2]
            return -np.sum(radii)
        
        # More efficient constraint check with early termination
        def check_constraints(params):
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            # Check boundary constraints first
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                    return False
            
            # Check overlap constraints efficiently using spatial indexing
            distances = cdist(positions, positions)
            # Only check pairs where i < j to avoid double counting
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        return False
                        
            return True
        
        # Flatten initial parameters
        initial_params = circles.flatten()
        
        # Try evolutionary approach with better parameters and more generations
        try:
            # Create DEAP toolbox for our optimization problem
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Define bounds for each parameter (x, y, r for each circle)
            bounds = []
            for i in range(n):
                # x bounds
                bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
            
            # Create individual with bounds
            def create_individual():
                individual = []
                for bound in bounds:
                    individual.append(random.uniform(bound[0], bound[1]))
                return creator.Individual(individual)
            
            toolbox.register("individual", create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                # Check if individual is valid
                if not check_constraints(np.array(individual)):
                    return (-1e10,)  # Invalid solution penalty
                
                # Calculate fitness (sum of radii)
                radii = np.array(individual).reshape(-1, 3)[:, 2]
                return (np.sum(radii),)
            
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxUniform, indpb=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.015, indpb=0.15)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Run evolution with more generations for better exploration
            population = toolbox.population(n=70)  # Increased population size
            hof = tools.HallOfFame(1)
            
            # Run evolution with more generations and better statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run evolution with more generations and better parameters
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.8, mutpb=0.3, 
                ngen=150, stats=stats, halloffame=hof, verbose=False
            )
            
            if hof:
                best_individual = hof[0]
                if check_constraints(np.array(best_individual)):
                    optimized_circles = np.array(best_individual).reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_circles = optimized_circles.copy()
            
        except Exception as e:
            # Fall back to other optimization approaches if needed
            pass
        
        # If evolutionary didn't work well, try gradient-based optimization
        try:
            bounds = []
            for i in range(n):
                # x bounds
                bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
            
            # Try a more aggressive local optimization with multiple restarts
            best_local_sum = 0
            best_local_circles = None
            
            for restart in range(8):  # More restarts for better chance
                # Slightly perturb the initial solution
                if restart == 0:
                    perturbed = initial_params.copy()
                else:
                    perturbed = initial_params + np.random.normal(0, 0.01, len(initial_params))
                    
                local_result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                    options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-6}
                )
                
                if local_result.success:
                    optimized_circles = local_result.x.reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_local_sum:
                        best_local_sum = current_sum
                        best_local_circles = optimized_circles.copy()
            
            if best_local_circles is not None and best_local_sum > best_sum:
                best_sum = best_local_sum
                best_circles = best_local_circles.copy()
                    
        except Exception as e:
            pass
    
    # If no optimization worked well, try a more robust approach with better initialization
    if best_circles is None or best_sum < 1.0:
        # Try a more refined approach using a different aspect ratio
        width, height = 1.5, 0.5  # This has shown promise in many cases
        
        # Create a much better initial configuration using a more intelligent pattern
        circles = np.zeros((n, 3))
        
        # Try to place circles in a pattern inspired by known optimal packings
        # For 21 circles, we can try a pattern that's more compact and balanced
        positions = []
        
        # Create a grid-like but not perfectly regular pattern
        rows = 5
        cols = 5
        
        grid_width = width / cols if cols > 0 else width
        grid_height = height / rows if rows > 0 else height
        
        # Generate positions with some hexagonal offset
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= n:
                    break
                x = (col + 0.5) * grid_width
                if row % 2 == 1:
                    x += grid_width * 0.25  # Offset every other row
                y = (row + 0.5) * grid_height
                # Apply boundary constraints
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                positions.append([x, y])
        
        # Fill remaining with random but constrained positions
        while len(positions) < n:
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            positions.append([x, y])
        
        # Initialize with reasonable radii
        for i in range(min(n, len(positions))):
            x, y = positions[i]
            # Radius based on available space - more generous than before
            r = min(x, width - x, y, height - y) * 0.35
            r = max(0.01, min(r, min(width, height) * 0.35))
            circles[i] = [x, y, r]
        
        # Fill remaining circles with small random placements
        for i in range(len(positions), n):
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            r = random.uniform(0.01, min(width, height) * 0.2)
            circles[i] = [x, y, r]
        
        # Optimize with local search
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
        
        def objective(params):
            radii = params.reshape(-1, 3)[:, 2]
            return -np.sum(radii)
        
        def constraint_func(params):
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            distances = cdist(positions, positions)
            constraints = []
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    constraints.append(dist - min_dist)
            
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                constraints.append(x - r)
                constraints.append(width - x - r)
                constraints.append(y - r)
                constraints.append(height - y - r)
                
            return np.array(constraints)
        
        # Use a more aggressive local optimization with multiple restarts
        try:
            best_local_sum = 0
            best_local_circles = None
            
            for _ in range(10):  # Even more restarts for better chance
                # Slightly perturb the initial solution
                perturbed = circles.flatten() + np.random.normal(0, 0.01, len(circles.flatten()))
                local_result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if local_result.success:
                    optimized_circles = local_result.x.reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    if current_sum > best_local_sum:
                        best_local_sum = current_sum
                        best_local_circles = optimized_circles.copy()
            
            if best_local_circles is not None and best_local_sum > best_sum:
                best_sum = best_local_sum
                best_circles = best_local_circles.copy()
                
        except Exception as e:
            pass
    
    # Final fallback to a highly optimized structured approach
    if best_circles is None:
        # Try to find a better aspect ratio through a simple search
        best_aspect_ratio = (1.5, 0.5)
        best_aspect_sum = 0
        
        # Try a few key aspect ratios - focus on what's proven to work
        test_ratios = [(1.5, 0.5), (2.0, 0.5), (1.2, 0.8), (0.8, 1.2), (1.0, 1.0), (1.8, 0.7)]
        
        for w_ratio, h_ratio in test_ratios:
            if time.time() - start_time > max_time:
                break
                
            width = 2 * w_ratio / (w_ratio + h_ratio)
            height = 2 * h_ratio / (w_ratio + h_ratio)
            
            # Create a highly optimized pattern with better initial configuration
            circles = np.zeros((21, 3))
            
            # Create a pattern inspired by known efficient packings with better spacing
            # Use a 4x6 grid with some offsetting
            rows = 4
            cols = 6
            
            grid_width = width / cols
            grid_height = height / rows
            
            count = 0
            for row in range(rows):
                for col in range(cols):
                    if count >= 21:
                        break
                    x = (col + 0.5) * grid_width
                    if row % 2 == 1:
                        x += grid_width * 0.5
                    y = (row + 0.5) * grid_height
                    # Ensure we're within bounds with safety margin
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    # Use a more carefully calculated radius
                    r = min(grid_width, grid_height) * 0.32
                    r = max(0.01, min(r, min(width, height) * 0.32))
                    circles[count] = [x, y, r]
                    count += 1
            
            # Try to optimize this with a more aggressive local search
            bounds = []
            for i in range(21):
                bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
            
            def objective(params):
                radii = params.reshape(-1, 3)[:, 2]
                return -np.sum(radii)
            
            def constraint_func(params):
                positions = params.reshape(-1, 3)[:, :2]
                radii = params.reshape(-1, 3)[:, 2]
                
                distances = cdist(positions, positions)
                constraints = []
                
                for i in range(21):
                    for j in range(i+1, 21):
                        dist = distances[i, j]
                        min_dist = radii[i] + radii[j]
                        constraints.append(dist - min_dist)
                
                for i in range(21):
                    x, y, r = positions[i][0], positions[i][1], radii[i]
                    constraints.append(x - r)
                    constraints.append(width - x - r)
                    constraints.append(y - r)
                    constraints.append(height - y - r)
                    
                return np.array(constraints)
            
            try:
                # Try multiple restarts with better convergence and more iterations
                best_restart_sum = 0
                best_restart_circles = None
                
                for restart in range(8):  # More restarts
                    # Start with slightly perturbed version
                    if restart == 0:
                        start_params = circles.flatten()
                    else:
                        start_params = circles.flatten() + np.random.normal(0, 0.01, len(circles.flatten()))
                        
                    local_result = minimize(
                        objective,
                        start_params,
                        method='SLSQP',
                        bounds=bounds,
                        constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                        options={'maxiter': 500, 'ftol': 1e-7}
                    )
                    
                    if local_result.success:
                        optimized_circles = local_result.x.reshape(-1, 3)
                        current_sum = np.sum(optimized_circles[:, 2])
                        if current_sum > best_restart_sum:
                            best_restart_sum = current_sum
                            best_restart_circles = optimized_circles.copy()
                
                if best_restart_circles is not None and best_restart_sum > best_aspect_sum:
                    best_aspect_sum = best_restart_sum
                    best_circles = best_restart_circles.copy()
                    best_aspect_ratio = (w_ratio, h_ratio)
            except:
                continue
        
        # If still no success, return the basic structured solution with improved parameters
        if best_circles is None:
            # Create the final fallback solution with a very robust approach
            width, height = 1.5, 0.5
            circles = np.zeros((21, 3))
            
            # Use a clean grid approach with even better initialization
            rows = 4
            cols = 6
            grid_width = width / cols
            grid_height = height / rows
            
            count = 0
            for row in range(rows):
                for col in range(cols):
                    if count >= 21:
                        break
                    x = (col + 0.5) * grid_width
                    if row % 2 == 1:
                        x += grid_width * 0.25  # Slight offset
                    y = (row + 0.5) * grid_height
                    # Add slight randomness to avoid perfect patterns
                    x += (random.random() - 0.5) * grid_width * 0.15
                    y += (random.random() - 0.5) * grid_height * 0.15
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    r = min(grid_width, grid_height) * 0.33
                    circles[count] = [x, y, r]
                    count += 1
            
            return circles
    
    return best_circles if best_circles is not None else np.zeros((21, 3))


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
