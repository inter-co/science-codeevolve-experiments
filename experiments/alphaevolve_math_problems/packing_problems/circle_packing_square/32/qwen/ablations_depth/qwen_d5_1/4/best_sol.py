# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
from deap import base, creator, tools, algorithms
import math

# Global constants for the problem
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

def generate_optimized_hexagonal_lattice() -> np.ndarray:
    """Generate initial configuration using optimized hexagonal lattice pattern"""
    # For 32 circles, we'll create a more efficient hexagonal packing
    # We'll arrange in approximately 5 rows of 6-7 columns
    
    # Calculate optimal grid dimensions
    rows = 5
    cols_per_row = [6, 7, 6, 7, 6]  # Alternating pattern
    
    positions = []
    
    # Hexagonal packing parameters
    sqrt3 = np.sqrt(3)
    # Start with a spacing that allows good packing
    spacing_x = 0.15
    spacing_y = spacing_x * sqrt3 / 2
    
    for i in range(rows):
        row_cols = cols_per_row[i]
        offset = (i % 2) * spacing_x / 2  # Offset every other row
        
        for j in range(row_cols):
            if len(positions) >= N_CIRCLES:
                break
            x = offset + j * spacing_x
            y = i * spacing_y
            positions.append([x, y])
        
        if len(positions) >= N_CIRCLES:
            break
    
    # Trim to exactly N_CIRCLES
    positions = positions[:N_CIRCLES]
    
    # Initialize with calculated radii that fit well in the space
    circles = np.array(positions)
    # Initial radii based on spacing
    radii = np.full(N_CIRCLES, spacing_x * 0.3)
    circles = np.column_stack([circles, radii])
    
    # Adjust for boundary constraints
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        max_r = min(x, y, 1-x, 1-y)
        circles[i, 2] = min(r, max_r * 0.9)  # Slightly smaller to ensure feasibility
    
    return circles

def calculate_radius_bounds(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate minimum and maximum possible radii for each circle"""
    n = len(circles)
    min_radius = np.zeros(n)
    max_radius = np.zeros(n)
    
    for i in range(n):
        x, y, _ = circles[i]
        # Maximum radius without going outside the square
        max_radius[i] = min(x, y, 1-x, 1-y)
        
        # Minimum radius is determined by non-overlap constraints
        min_radius[i] = 0.001  # Small positive value
    
    return min_radius, max_radius

def compute_objective(circles: np.ndarray) -> float:
    """Compute the objective function (negative sum of radii for minimization)"""
    return -np.sum(circles[:, 2])

def check_constraints(circles: np.ndarray) -> bool:
    """Check if all constraints are satisfied"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > y or r > (1-x) or r > (1-y):
            return False
    
    # Check non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < (r1 + r2):
                return False
    
    return True

def evaluate_individual(individual: np.ndarray) -> float:
    """Evaluate fitness of individual (negative sum of radii)"""
    # Convert flat individual to circles array
    circles = individual.reshape((N_CIRCLES, 3))
    
    # Check constraints
    if not check_constraints(circles):
        # Return very poor fitness for infeasible solutions
        return -np.inf
    
    # Return negative sum of radii (since we're minimizing in DEAP)
    return -np.sum(circles[:, 2])

def mutate_individual(individual: np.ndarray, indpb: float = 0.1) -> np.ndarray:
    """Mutate individual with parameter probability"""
    mutated = individual.copy()
    n_params = len(mutated)
    
    # Mutate positions and radii
    for i in range(n_params):
        if random.random() < indpb:
            # Determine what type of parameter we're mutating
            param_idx = i % 3  # 0=x, 1=y, 2=r
            
            if param_idx == 0 or param_idx == 1:  # x or y coordinate
                # Add small random perturbation
                mutated[i] += random.uniform(-0.02, 0.02)
                # Keep within bounds
                mutated[i] = max(0.001, min(0.999, mutated[i]))
            else:  # radius
                # Add small random change to radius
                mutated[i] += random.uniform(-0.01, 0.01)
                # Keep within bounds
                mutated[i] = max(0.001, min(0.499, mutated[i]))
    
    return mutated

def optimize_with_evolutionary(initial_circles: np.ndarray, generations: int = 50) -> np.ndarray:
    """Use evolutionary algorithm to optimize circle positions and radii"""
    
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Initialize individuals
    def init_individual():
        return initial_circles.flatten().copy()
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation and mutation functions
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mutate", mutate_individual, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create population
    pop = toolbox.population(n=50)
    
    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                         ngen=generations, stats=stats, 
                                         halloffame=hof, verbose=False)
    except Exception:
        # Fallback to simple optimization if evolution fails
        return initial_circles
    
    # Return the best individual found
    if hof:
        best_individual = hof[0]
        return best_individual.reshape((N_CIRCLES, 3))
    
    return initial_circles

def optimize_with_gradient_refinement(initial_circles: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Refine using gradient-based optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = initial_circles.flatten()
    
    def objective_flat(params):
        # Reshape back to circles array
        circles = params.reshape((n, 3))
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_func(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Containment constraints (all must be >= 0)
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,      # x >= r
                y - r,      # y >= r
                1 - x - r,  # 1-x >= r
                1 - y - r   # 1-y >= r
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - (r1 + r2))  # distance >= r1 + r2
        
        return np.array(constraints)
    
    # Create bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    try:
        result = minimize(
            objective_flat,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            # Verify constraints are satisfied
            if check_constraints(optimized_circles):
                return optimized_circles
    except Exception:
        pass
    
    # Return initial if optimization fails
    return initial_circles

def perturb_configuration(circles: np.ndarray, perturbation: float = 0.01) -> np.ndarray:
    """Create a slightly perturbed version of the configuration"""
    perturbed = circles.copy()
    for i in range(len(perturbed)):
        # Slightly perturb position
        perturbed[i, 0] += random.uniform(-perturbation, perturbation)
        perturbed[i, 1] += random.uniform(-perturbation, perturbation)
        # Ensure within bounds
        perturbed[i, 0] = max(0.001, min(0.999, perturbed[i, 0]))
        perturbed[i, 1] = max(0.001, min(0.999, perturbed[i, 1]))
        
        # Keep radius reasonable
        perturbed[i, 2] = max(0.001, min(0.499, perturbed[i, 2]))
    return perturbed

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm and gradient-based refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    best_circles = None
    best_sum = 0
    
    # Try multiple starting configurations with different strategies
    for attempt in range(10):
        # Generate initial configuration with different strategies
        if attempt == 0:
            # First attempt: optimized hexagonal lattice
            circles = generate_optimized_hexagonal_lattice()
        elif attempt < 5:
            # Random initialization with better spread
            circles = np.zeros((N_CIRCLES, 3))
            for i in range(N_CIRCLES):
                circles[i, 0] = random.uniform(0.05, 0.95)
                circles[i, 1] = random.uniform(0.05, 0.95)
                circles[i, 2] = random.uniform(0.01, 0.15)
        else:
            # Perturb previous best if available
            if best_circles is not None:
                circles = perturb_configuration(best_circles, 0.05)
            else:
                circles = generate_optimized_hexagonal_lattice()
        
        # Apply evolutionary algorithm for global search
        evolved_circles = optimize_with_evolutionary(circles, generations=30)
        
        # Refine with gradient-based optimization
        refined_circles = optimize_with_gradient_refinement(evolved_circles, max_iter=300)
        
        # Calculate sum of radii
        sum_radii = np.sum(refined_circles[:, 2])
        
        # Update best solution
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = refined_circles.copy()
    
    # Final refinement with more iterations
    if best_circles is not None:
        final_circles = optimize_with_gradient_refinement(best_circles, max_iter=1000)
        sum_radii = np.sum(final_circles[:, 2])
        
        # If still better, update
        if sum_radii > best_sum:
            best_circles = final_circles
    
    # Ensure we have a valid solution even if optimization failed
    if best_circles is None:
        best_circles = generate_optimized_hexagonal_lattice()
    
    return best_circles


# EVOLVE-BLOCK-END
