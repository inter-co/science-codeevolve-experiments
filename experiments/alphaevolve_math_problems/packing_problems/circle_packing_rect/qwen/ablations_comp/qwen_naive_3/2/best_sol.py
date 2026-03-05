# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary and optimization approach to beat the benchmark.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal
    width, height = 1.0, 1.0
    
    # Number of circles
    n = 21
    
    # Use a more sophisticated initialization approach
    def generate_better_initialization():
        # Try different rectangle dimensions to find optimal aspect ratio
        best_config = None
        best_sum = 0
        
        # Test different width/height ratios
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        for ratio in ratios:
            w = 2.0 / (1 + ratio)  # width + height = 2
            h = 2.0 / (1 + 1/ratio)
            
            # Try hexagonal packing with this dimension
            try:
                # Calculate area per circle
                total_area = w * h
                circle_area = total_area / n * 0.8  # Leave some margin
                avg_radius = np.sqrt(circle_area / np.pi)
                
                # Hexagonal packing parameters
                spacing = 2 * avg_radius
                hex_radius = spacing * np.sqrt(3) / 2
                
                # Generate hexagonal grid
                rows = int(np.ceil(h / hex_radius)) + 1
                cols = int(np.ceil(w / spacing)) + 1
                
                centers = []
                for i in range(rows):
                    for j in range(cols):
                        x = 0.1 + j * spacing + (i % 2) * spacing / 2
                        y = 0.1 + i * hex_radius
                        if x <= w - 0.1 and y <= h - 0.1:
                            centers.append([x, y])
                
                # Take first n centers, or pad if needed
                if len(centers) >= n:
                    selected_centers = np.array(centers[:n])
                else:
                    # Add random points for remaining circles
                    selected_centers = np.array(centers)
                    remaining = n - len(centers)
                    for _ in range(remaining):
                        x = random.uniform(0.1, w - 0.1)
                        y = random.uniform(0.1, h - 0.1)
                        selected_centers = np.vstack([selected_centers, [x, y]])
                
                # Test this configuration
                test_radii = np.full(n, avg_radius * 0.8)  # Start with reasonable radii
                test_sum = np.sum(test_radii)
                
                if test_sum > best_sum:
                    best_sum = test_sum
                    best_config = (selected_centers, w, h)
            except:
                continue
        
        if best_config is not None:
            return best_config[0], best_config[1], best_config[2]
        else:
            # Fallback to default
            centers = []
            for i in range(n):
                x = random.uniform(0.1, 1.9)
                y = random.uniform(0.1, 1.9)
                centers.append([x, y])
            return np.array(centers), 1.0, 1.0
    
    # Generate initial configuration
    initial_centers, width, height = generate_better_initialization()
    
    # Set initial radii based on available space
    initial_radii = np.full(n, 0.05)
    
    # Combine into one array for optimization
    initial_params = np.column_stack([initial_centers, initial_radii])
    
    # Define constraint checking functions
    def check_containment(x, y, r, w, h):
        """Check if circle fits within bounds"""
        return (x - r >= 0 and y - r >= 0 and x + r <= w and y + r <= h)
    
    def check_non_overlap(x1, y1, r1, x2, y2, r2):
        """Check if two circles don't overlap"""
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        return dist_sq >= (r1 + r2)**2
    
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (negative sum of radii)"""
        # Reshape individual to (n, 3) format
        circles = individual.reshape((n, 3))
        
        # Extract parameters
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Check constraints
        valid = True
        penalty = 0
        
        # Containment constraints
        for i in range(n):
            if not check_containment(xs[i], ys[i], rs[i], width, height):
                valid = False
                penalty += 1000  # Large penalty
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                if not check_non_overlap(xs[i], ys[i], rs[i], xs[j], ys[j], rs[j]):
                    valid = False
                    penalty += 1000  # Large penalty
        
        # If invalid, return large penalty
        if not valid:
            return penalty + 1000000
        
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(rs)
    
    # Evolutionary algorithm approach
    def run_evolutionary_optimization():
        # Create DEAP types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Define bounds for each parameter
        bounds = []
        for i in range(n):
            # x bounds
            bounds.extend([(0.01, width - 0.01)])
            # y bounds  
            bounds.extend([(0.01, height - 0.01)])
            # r bounds
            bounds.extend([(0.001, min(width, height)/2 - 0.01)])
        
        # Initialize individuals
        def init_individual(icls, bounds):
            ind = []
            for lower, upper in bounds:
                ind.append(random.uniform(lower, upper))
            return icls(ind)
        
        toolbox.register("individual", init_individual, creator.Individual, bounds)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register evaluation function
        toolbox.register("evaluate", evaluate_individual)
        
        # Register operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        # Run with limited generations to save time
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        
        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.7, mutpb=0.2, 
                ngen=20, stats=stats, halloffame=hof, verbose=False
            )
            return hof[0]
        except:
            # Fallback to simple optimization if evolution fails
            return None
    
    # Try evolutionary approach first
    try:
        evolved_individual = run_evolutionary_optimization()
        if evolved_individual is not None:
            circles = evolved_individual.reshape((n, 3))
            # Fine-tune with local optimization
            circles = fine_tune_with_local_optimization(circles, width, height)
            return circles
    except:
        pass
    
    # Fallback to improved optimization approach
    # Define constraints and bounds properly for scipy optimization
    def get_constraints_and_bounds():
        # Bounds for x, y, r
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.01, width - 0.01),   # x bounds
                (0.01, height - 0.01),  # y bounds
                (0.001, min(width, height)/2 - 0.01)  # r bounds
            ])
        
        # Constraint functions
        constraints = []
        
        # Circle containment constraints
        def containment_constraint(params):
            # params is flattened array of [x1, y1, r1, x2, y2, r2, ...]
            results = []
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Ensure circle is within bounds
                results.append(x - r)  # Should be >= 0
                results.append(y - r)  # Should be >= 0
                results.append(width - x - r)  # Should be >= 0
                results.append(height - y - r)  # Should be >= 0
            return np.array(results)
        
        # Non-overlapping constraints - more efficient implementation
        def non_overlap_constraint(params):
            # Check pairwise distances efficiently
            results = []
            for i in range(n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                for j in range(i+1, n):
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    # Distance between centers
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    # Should be >= sum of radii (for no overlap)
                    results.append(dist - (r1 + r2))
            return np.array(results)
        
        constraints.append({'type': 'ineq', 'fun': containment_constraint})
        constraints.append({'type': 'ineq', 'fun': non_overlap_constraint})
        
        return bounds, constraints
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        # Extract radii
        radii = params[2::3]
        # Return negative sum of radii (we want to maximize)
        return -np.sum(radii)
    
    # Optimization
    bounds, constraints = get_constraints_and_bounds()
    
    # Run optimization with multiple attempts for better results
    best_result = None
    best_sum = 0
    
    # Try multiple optimization runs with different starting points
    for attempt in range(10):  # Increased attempts
        # Create slightly different starting points
        perturbed_params = initial_params.copy()
        if attempt > 0:
            # Add small random perturbations
            for i in range(n):
                perturbed_params[i, 0] += random.uniform(-0.1, 0.1)
                perturbed_params[i, 1] += random.uniform(-0.1, 0.1)
                # Keep within bounds
                perturbed_params[i, 0] = np.clip(perturbed_params[i, 0], 0.01, width - 0.01)
                perturbed_params[i, 1] = np.clip(perturbed_params[i, 1], 0.01, height - 0.01)
        
        try:
            # Try different optimization methods
            result = minimize(
                objective,
                perturbed_params.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-5}
            )
            
            if result.success:
                # Check if this result is better
                final_params = result.x
                test_radii = final_params[2::3]
                test_sum = np.sum(test_radii)
                if test_sum > best_sum:
                    best_sum = test_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If we found a good result, use it; otherwise fall back to hexagonal packing
    if best_result is not None:
        final_params = best_result.x
        circles = np.reshape(final_params, (n, 3))
    else:
        # Fallback to better hexagonal packing
        circles = np.zeros((n, 3))
        # Generate hexagonal pattern with optimized spacing
        spacing = 0.25  # Adjusted spacing
        hex_radius = spacing * np.sqrt(3) / 2
        row_count = int(np.ceil(np.sqrt(n)))
        col_count = int(np.ceil(n / row_count))
        
        idx = 0
        for i in range(row_count):
            for j in range(col_count):
                if idx >= n:
                    break
                x = 0.1 + j * spacing + (i % 2) * spacing / 2
                y = 0.1 + i * hex_radius
                # Make sure we stay within bounds
                if x <= width - 0.1 and y <= height - 0.1:
                    circles[idx] = [x, y, 0.05]
                    idx += 1
                else:
                    circles[idx] = [width/2, height/2, 0.05]
                    idx += 1
                if idx >= n:
                    break
    
    # Final refinement with more sophisticated approach
    circles = fine_tune_with_local_optimization(circles, width, height)
    
    return circles

def fine_tune_with_local_optimization(circles, width, height):
    """Apply additional local optimization to improve solution"""
    n = len(circles)
    
    # Try to increase radii while maintaining constraints
    max_iterations = 100  # Increased iterations
    for iteration in range(max_iterations):
        improved = False
        # Shuffle circle indices to avoid systematic bias
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            # Try to increase radius of circle i
            current_x, current_y, current_r = circles[i]
            
            # Calculate how much we can increase radius
            max_increase = current_r
            
            # Check containment constraints
            max_increase = min(max_increase, current_x - 0.01)
            max_increase = min(max_increase, current_y - 0.01)
            max_increase = min(max_increase, width - current_x - 0.01)
            max_increase = min(max_increase, height - current_y - 0.01)
            
            # Check non-overlap constraints with other circles
            for j in range(n):
                if i != j:
                    other_x, other_y, other_r = circles[j]
                    dist = np.sqrt((current_x - other_x)**2 + (current_y - other_y)**2)
                    max_increase = min(max_increase, dist - other_r - 0.001)
            
            # If we can increase radius, do so
            if max_increase > 0.001:
                # Increase radius by a small amount
                new_r = min(current_r + max_increase * 0.1, current_r * 1.1)
                # Check if this is valid
                valid = True
                for j in range(n):
                    if i != j:
                        other_x, other_y, other_r = circles[j]
                        dist = np.sqrt((current_x - other_x)**2 + (current_y - other_y)**2)
                        if dist < (new_r + other_r):
                            valid = False
                            break
                
                if valid:
                    circles[i, 2] = new_r
                    improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
