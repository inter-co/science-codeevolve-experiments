# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations
import time

# Import evolutionary optimization library
try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

# For better optimization, let's also import some additional libraries
try:
    import nevergrad as ng
    HAS_NEVERGRAD = True
except ImportError:
    HAS_NEVERGRAD = False

# More efficient constraint handling
def compute_distances(positions):
    """Compute pairwise distances between circle centers efficiently"""
    return cdist(positions, positions)

def check_feasibility_fast(positions, radii, width, height):
    """Fast feasibility check using vectorized operations"""
    # Check boundary constraints
    if np.any(positions[:, 0] - radii < 0) or \
       np.any(positions[:, 0] + radii > width) or \
       np.any(positions[:, 1] - radii < 0) or \
       np.any(positions[:, 1] + radii > height):
        return False
    
    # Check overlap constraints using vectorized operations
    if len(positions) > 1:
        dist_matrix = compute_distances(positions)
        # Create mask for upper triangle (avoid double counting)
        upper_triangle = np.triu(np.ones((len(positions), len(positions)), dtype=bool), k=1)
        # Check if any pair violates non-overlap constraint
        min_distances = radii[:, None] + radii[None, :]
        actual_distances = dist_matrix[upper_triangle]
        min_distances = min_distances[upper_triangle]
        
        if np.any(actual_distances < min_distances):
            return False
    
    return True

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses an improved evolutionary algorithm approach with better initialization and constraint handling.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    
    # Better initialization using hexagonal packing principles and adaptive strategies
    def initialize_better_layout():
        best_sum = 0
        best_circles = None
        best_width = 1.0
        best_height = 1.0
        
        # Try different rectangle dimensions to find optimal aspect ratio
        # Focus on more promising ratios around 1.0 (square-like) and 0.5 and 2.0
        ratios = np.concatenate([np.linspace(0.3, 1.0, 15), np.linspace(1.0, 3.0, 15)])
        
        for ratio in ratios:
            width = 1.0 * ratio
            height = 2.0 - width
            
            if width <= 0 or height <= 0:
                continue
                
            # Use hexagonal packing-inspired approach for better density
            circles = []
            
            # Try to place circles in a more sophisticated pattern
            # Start with a hexagonal lattice pattern
            if width > 0.1 and height > 0.1:
                # Hexagonal packing parameters
                min_radius = 0.01
                max_radius = min(width, height) * 0.3
                
                # Determine grid size based on desired density
                grid_size = int(np.ceil(np.sqrt(n)))
                if grid_size * grid_size < n:
                    grid_size += 1
                    
                # Calculate spacing
                spacing_x = width / (grid_size + 1)
                spacing_y = height / (grid_size + 1)
                
                # Adjust spacing for hexagonal pattern
                hex_spacing_x = spacing_x
                hex_spacing_y = spacing_y * np.sqrt(3) / 2
                
                # Place circles in staggered rows
                row_offset = 0
                placed_count = 0
                
                for row in range(grid_size):
                    if placed_count >= n:
                        break
                    for col in range(grid_size):
                        if placed_count >= n:
                            break
                            
                        # Stagger every other row
                        x_offset = row_offset * hex_spacing_x / 2
                        x = (col + 0.5) * hex_spacing_x + x_offset
                        y = (row + 0.5) * hex_spacing_y
                        
                        # Keep within bounds
                        x = np.clip(x, 0, width)
                        y = np.clip(y, 0, height)
                        
                        # Calculate max possible radius
                        max_radius_at_pos = min(x, width - x, y, height - y)
                        
                        # Use radius that balances center proximity and boundary constraints
                        # Circles near center should get larger radii
                        center_dist = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                        center_factor = 1.0 - 0.8 * (center_dist / (np.sqrt((width/2)**2 + (height/2)**2)))
                        center_factor = max(0.2, min(1.0, center_factor))
                        
                        # Different strategies for different positions
                        if placed_count < 5:  # First few very large
                            radius = max_radius_at_pos * 0.4 * center_factor
                        elif placed_count < 12:  # Middle ones
                            radius = max_radius_at_pos * 0.3 * center_factor
                        else:  # Last ones smaller
                            radius = max_radius_at_pos * 0.15 * center_factor
                        
                        radius = max(radius, min_radius)
                        
                        # Ensure valid placement
                        if x >= radius and x <= width - radius and \
                           y >= radius and y <= height - radius and \
                           radius > 0:
                            circles.append([x, y, radius])
                            placed_count += 1
                    
                    # Alternate row offset
                    row_offset = 1 - row_offset
            
            # If we don't have enough circles, fill with random placements
            if len(circles) < n:
                for i in range(len(circles), n):
                    x = random.uniform(0.05, width - 0.05)
                    y = random.uniform(0.05, height - 0.05)
                    max_radius = min(x, width - x, y, height - y)
                    radius = max_radius * 0.2  # Larger initial radius
                    radius = max(radius, 0.005)
                    circles.append([x, y, radius])
            
            current_sum = sum(circle[2] for circle in circles[:n])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles[:n]
                best_width = width
                best_height = height
        
        # If we still haven't found a good configuration, use a more systematic approach
        if best_circles is None or best_sum < 1.0:
            circles = []
            width, height = 1.0, 1.0  # Square for simplicity
            
            # Try a different strategy: pack in a spiral pattern from center outward
            center_x, center_y = width/2, height/2
            max_radius = min(width, height) * 0.2
            
            # Spiral arrangement
            angle = 0
            radius = 0.05
            step_angle = 0.5
            step_radius = 0.1
            
            while len(circles) < n and radius < min(width, height) / 2:
                # Calculate position along spiral
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Keep within bounds
                x = np.clip(x, 0, width)
                y = np.clip(y, 0, height)
                
                # Calculate max radius
                max_radius_at_pos = min(x, width - x, y, height - y)
                circle_radius = max_radius_at_pos * 0.25
                
                if x >= circle_radius and x <= width - circle_radius and \
                   y >= circle_radius and y <= height - circle_radius and \
                   circle_radius > 0:
                    circles.append([x, y, circle_radius])
                
                # Move to next point on spiral
                angle += step_angle
                radius += step_radius
                
                # Occasionally increase step size to cover area faster
                if len(circles) % 7 == 0:
                    step_radius *= 1.1
            
            # Fill any remaining spots with random placement
            while len(circles) < n:
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                max_radius = min(x, width - x, y, height - y)
                radius = max_radius * 0.2
                radius = max(radius, 0.005)
                circles.append([x, y, radius])
            
            best_circles = circles[:n]
            best_width = width
            best_height = height
            
        return np.array(best_circles), best_width, best_height
    
    # Initialize with better configuration
    circles, rect_width, rect_height = initialize_better_layout()
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(params):
        # Reshape params into positions and radii
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Calculate negative sum of radii (we want to maximize sum, so minimize negative)
        return -np.sum(radii)
    
    # Vectorized constraint functions for better performance
    def non_overlap_constraint(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Compute distance matrix efficiently
        dist_matrix = compute_distances(positions)
        
        # Get all constraint violations using vectorized operations
        # Create upper triangular matrix of all pairs
        upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
        actual_distances = dist_matrix[upper_triangle]
        required_distances = (radii[:, None] + radii[None, :])[upper_triangle]
        
        # Violations are negative values (distance < required distance)
        violations = actual_distances - required_distances
        
        return violations
    
    # Boundary constraints for circles to stay within rectangle
    def boundary_constraint(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Vectorized boundary constraints
        left_bound = positions[:, 0] - radii
        right_bound = rect_width - positions[:, 0] - radii
        bottom_bound = positions[:, 1] - radii
        top_bound = rect_height - positions[:, 1] - radii
        
        # Return all boundary violations (should be positive for feasibility)
        return np.concatenate([left_bound, right_bound, bottom_bound, top_bound])
    
    # Combined constraints - all must be >= 0 for feasibility
    def combined_constraints(params):
        # Non-overlap constraints (positive means satisfied)
        overlap_violations = non_overlap_constraint(params)
        # Boundary constraints (positive means satisfied)  
        boundary_violations = boundary_constraint(params)
        # Combine constraints (positive means satisfied)
        return np.concatenate([overlap_violations, boundary_violations])
    
    # Enhanced constraint checking with vectorized operations
    def check_feasibility(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        return check_feasibility_fast(positions, radii, rect_width, rect_height)
    
    # Improved optimization using hybrid approach with better exploration
    # Initial parameter vector: [x1, y1, x2, y2, ..., xn, yn, r1, r2, ..., rn]
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # Positions
        circles[:, 2]              # Radii
    ])
    
    # Set bounds for positions and radii
    # Positions: [0, width] for x and y coordinates
    # Radii: [1e-6, min(width, height)/2] to prevent degenerate cases
    bounds = [(0, rect_width) for _ in range(2*n)] + [(1e-6, min(rect_width, rect_height)/2) for _ in range(n)]
    
    # Define constraints - all must be >= 0
    constraints = {
        'type': 'ineq',
        'fun': combined_constraints
    }
    
    # Use a more robust optimization approach with better strategies
    try:
        # Strategy 1: Try a hybrid approach with better exploration
        best_result = None
        best_value = float('-inf')
        
        # Strategy 1a: Use Nevergrad for better optimization if available
        if HAS_NEVERGRAD:
            try:
                # Create optimizer with higher budget and better settings
                optimizer = ng.optimizers.NGOpt(
                    dimension=len(initial_params),
                    budget=2000,  # Increased budget
                    num_workers=1
                )
                
                # Evaluate initial solution
                initial_value = objective(initial_params)
                optimizer.tell(initial_params, initial_value)
                
                # Run optimization with more iterations
                for _ in range(2000):
                    candidate = optimizer.ask()
                    try:
                        value = objective(candidate)
                        optimizer.tell(candidate, value)
                    except:
                        continue
                
                # Get best solution
                best_candidate = optimizer.provide_recommendation()
                current_value = -objective(best_candidate)
                
                if current_value > best_value:
                    best_value = current_value
                    best_result = type('obj', (object,), {'x': best_candidate, 'success': True})()
                    
            except Exception as e:
                pass
        
        # Strategy 1b: Use differential evolution with better settings
        if best_result is None:
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    constraints=constraints,
                    seed=42,
                    maxiter=1000,      # Increased iterations
                    popsize=100,       # Larger population size
                    mutation=(0.5, 1.0),
                    recombination=0.9,
                    atol=1e-12,
                    rtol=1e-12,
                    disp=False
                )
                
                if de_result.success:
                    current_value = -de_result.fun
                    if current_value > best_value:
                        best_value = current_value
                        best_result = de_result
            except:
                pass
        
        # Strategy 2: Multi-start local optimization with better diversity
        if best_result is None:
            # Multi-start local optimization with better diversity
            for restart in range(20):  # More restarts for better exploration
                # Perturb the initial solution with more substantial changes
                perturbed_params = initial_params.copy()
                # Add more significant noise for exploration
                noise_scale = 0.2  # Increased noise for more exploration
                perturbed_params += np.random.normal(0, noise_scale, len(initial_params))
                
                # Clip to bounds
                for i, (bound, param) in enumerate(zip(bounds, perturbed_params)):
                    perturbed_params[i] = np.clip(param, bound[0], bound[1])
                
                try:
                    local_result = minimize(
                        objective,
                        perturbed_params,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12},  # Even tighter tolerances
                        tol=1e-12
                    )
                    
                    if local_result.success:
                        current_value = -local_result.fun
                        if current_value > best_value:
                            best_value = current_value
                            best_result = local_result
                except:
                    continue
        
        # Strategy 3: Try a simple genetic algorithm approach for better exploration
        if best_result is None and HAS_DEAP:
            try:
                # Simple genetic algorithm approach with better parameters
                toolbox = base.Toolbox()
                
                # Create individual (chromosome) representation
                def create_individual():
                    individual = []
                    for i in range(n):
                        # Position coordinates
                        individual.extend([random.uniform(0, rect_width), random.uniform(0, rect_height)])
                        # Radius
                        individual.append(random.uniform(1e-6, min(rect_width, rect_height)/2))
                    return individual
                
                # Fitness function (minimize negative sum of radii)
                def eval_fitness(individual):
                    positions = np.array(individual[:-n]).reshape(-1, 2)
                    radii = np.array(individual[-n:])
                    
                    # Check feasibility
                    if not check_feasibility_fast(positions, radii, rect_width, rect_height):
                        return (float('inf'),)  # Invalid solution
                    
                    return (-np.sum(radii),)  # Minimize negative sum (maximize sum)
                
                # Register functions
                toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("evaluate", eval_fitness)
                toolbox.register("mate", tools.cxTwoPoint)
                toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.15)  # Increased mutation
                toolbox.register("select", tools.selTournament, tournsize=5)  # Larger tournament size
                
                # Create population with more individuals
                population = toolbox.population(n=100)
                
                # Evaluate initial population
                fitnesses = list(map(toolbox.evaluate, population))
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit
                
                # Evolution loop with more generations
                for generation in range(100):
                    # Select the next generation individuals
                    offspring = toolbox.select(population, len(population))
                    offspring = list(map(toolbox.clone, offspring))
                    
                    # Apply crossover and mutation on the offspring
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < 0.7:  # Higher crossover rate
                            toolbox.mate(child1, child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    
                    for mutant in offspring:
                        if random.random() < 0.3:  # Higher mutation rate
                            toolbox.mutate(mutant)
                            del mutant.fitness.values
                    
                    # Evaluate the individuals with an invalid fitness
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = list(map(toolbox.evaluate, invalid_ind))
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    
                    # Replace the old population with the new one
                    population[:] = offspring
                
                # Find best individual
                best_ind = tools.selBest(population, 1)[0]
                current_value = -best_ind.fitness.values[0]
                
                if current_value > best_value:
                    best_value = current_value
                    # Convert back to the format expected by our code
                    best_result = type('obj', (object,), {'x': best_ind, 'success': True})()
                    
            except Exception as e:
                pass
        
        # Extract results if we have a successful optimization
        if best_result is not None:
            final_positions = best_result.x[:-n].reshape(-1, 2)
            final_radii = best_result.x[-n:]
            
            # Update circles array with optimized values
            circles[:, 0] = final_positions[:, 0]
            circles[:, 1] = final_positions[:, 1]
            circles[:, 2] = final_radii
            
            # Ensure all radii are positive and reasonable
            circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
            # Make sure radii don't exceed reasonable limits
            max_radius_allowed = min(rect_width, rect_height) / 2
            circles[:, 2] = np.minimum(circles[:, 2], max_radius_allowed)
            
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
