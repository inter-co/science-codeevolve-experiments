# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.optimize import minimize, Bounds, NonlinearConstraint # Added Bounds and NonlinearConstraint
from scipy.spatial.distance import pdist # For vectorized distance calculations
from numba import jit # Added numba for JIT compilation

# --- Global Constants and DEAP Setup (defined outside functions for single initialization) ---
# Number of circles to pack
N_CIRCLES = 26
# Bounds for x and y coordinates within the unit square [0,1]
BOUND_LOW_XY = 0.0
BOUND_UP_XY = 1.0
# Bounds for radii: minimum to avoid numerical issues, maximum based on square dimensions
EPSILON = 1e-7 # Small numerical constant for floating-point comparisons (e.g., minimum radius)
BOUND_LOW_R = EPSILON # Use EPSILON as the effective lower bound for radius
BOUND_UP_R = 0.5     # A single circle can have a max radius of 0.5 if centered at (0.5, 0.5)

# Penalty coefficients for the fitness function in the Evolutionary Algorithm.
# Tuned values, generally lower than initial target, but still high enough to guide effectively.
# Squared penalties are used for both types of violations for stronger gradients.
PENALTY_COEFF_OVERLAP = 500.0       # Adjusted from 10000.0, for sqrt-free penalty magnitude
PENALTY_COEFF_CONTAINMENT = 1000.0   # Adjusted from 10000.0
PENALTY_COEFF_NEGATIVE_RADIUS = 5000.0 # Added explicit penalty for negative radii

# Evolutionary Algorithm parameters (tuned for a balance between exploration and time limits)
POP_SIZE = 600   # Increased population size for more exploration (from 400)
NGEN = 1500      # Increased generations for deeper search (from 600)
CXPB = 0.8       # Crossover probability, slightly increased (from 0.7)
MUTPB = 0.2      # Mutation probability, slightly decreased (from 0.3)
HALL_OF_FAME_SIZE = 1 # Stores the single best individual found across all generations

# Parameters for eaMuPlusLambda algorithm
MU = POP_SIZE      # Number of individuals to select for the next generation (changed from POP_SIZE // 2)
LAMBDA_ = POP_SIZE # Number of offspring to produce at each generation

# Random seed for reproducibility of the stochastic evolutionary process
RANDOM_SEED = 42

# Initialize DEAP types:
try: # Use try-except to avoid re-creation errors in environments like notebooks
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Define a fitness that should be maximized
    # Individual is an np.ndarray for better performance with Numba and numpy operations
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
except AttributeError:
    pass # Classes already created

# Initialize DEAP toolbox for registering genetic operators
toolbox = base.Toolbox()

# --- Helper Functions for Evolutionary Algorithm and Local Refinement ---

@jit(nopython=True, fastmath=True) # Apply Numba JIT for performance critical evaluation function
def evaluate_packing(individual_1d_array: np.ndarray) -> tuple[float]:
    """
    Evaluates an individual's fitness. Fitness is defined as the sum of radii
    minus penalties for constraint violations (overlap and containment).
    Maximizing this value encourages larger circles that fit within the square and do not overlap.
    This function is JIT compiled for maximum performance.
    """
    # Reshape the flat individual array into a (N_CIRCLES, 3) numpy array for easier processing
    circles = individual_1d_array.reshape((N_CIRCLES, 3))
    
    sum_radii = 0.0
    containment_penalty = 0.0
    overlap_penalty = 0.0
    negative_radius_penalty = 0.0

    # Process each circle for radius sum and containment violations
    for i in range(N_CIRCLES):
        xi, yi, ri = circles[i, 0], circles[i, 1], circles[i, 2]
        
        sum_radii += ri
        
        # Radius positivity penalty
        if ri < BOUND_LOW_R:
            negative_radius_penalty += (BOUND_LOW_R - ri)**2 # Squared penalty for being below min radius
        
        # Containment penalties (squared violations)
        containment_penalty += max(0.0, ri - xi)**2           # Circle extends left of x=r
        containment_penalty += max(0.0, xi + ri - BOUND_UP_XY)**2 # Circle extends right of x=1
        containment_penalty += max(0.0, ri - yi)**2           # Circle extends below y=r
        containment_penalty += max(0.0, yi + ri - BOUND_UP_XY)**2 # Circle extends above y=1

    # Calculate non-overlap penalty: sum of squared overlap distances for any pair of circles
    # This loop is optimized by Numba JIT.
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES): # Iterate over unique pairs
            xi, yi, ri = circles[i, 0], circles[i, 1], circles[i, 2]
            xj, yj, rj = circles[j, 0], circles[j, 1], circles[j, 2]

            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist_required = ri + rj
            
            # Check for overlap: if actual distance squared is less than required distance squared
            if dist_sq < min_dist_required**2 and min_dist_required > 0: # Added min_dist_required > 0 check to prevent division by zero or NaN if r_i, r_j are near 0
                # Sqrt-free penalty: (R+r)^2 - d^2 (from Inspiration 2)
                overlap_penalty += (min_dist_required**2 - dist_sq)

    # The final fitness value: sum of radii penalized by constraint violations
    fitness = sum_radii - \
              PENALTY_COEFF_OVERLAP * overlap_penalty - \
              PENALTY_COEFF_CONTAINMENT * containment_penalty - \
              PENALTY_COEFF_NEGATIVE_RADIUS * negative_radius_penalty
              
    return fitness, # DEAP expects a tuple for fitness values


def generate_circle_params_list():
    """
    Generates random x, y, r attributes for a single circle as a list.
    Initial radii are kept relatively small to provide more space for initial arrangements,
    and x, y are ensured to be within [r, 1-r] for initial containment.
    """
    initial_r_range = (BOUND_LOW_R, BOUND_UP_R / 3.0) # Start with radii up to 1/3 of max possible
    r = random.uniform(*initial_r_range)
    x = random.uniform(r, BOUND_UP_XY - r)
    y = random.uniform(r, BOUND_UP_XY - r)
    return [x, y, r]

def initIndividual_ndarray(ind_class, param_gen_func, n_circles):
    """
    Custom initializer for an individual as an np.ndarray.
    It calls `param_gen_func` n_circles times and flattens the resulting (x,y,r) lists
    into a single numpy array as required by the `creator.Individual` (np.ndarray).
    """
    all_params = []
    for _ in range(n_circles):
        all_params.extend(param_gen_func())
    return ind_class(np.array(all_params, dtype=np.float64))

# Register the custom individual and population initialization methods
toolbox.register("params_gen", generate_circle_params_list) # Renamed attr_circle to params_gen
toolbox.register("individual", initIndividual_ndarray, creator.Individual, toolbox.params_gen, N_CIRCLES) # Uses new initIndividual_ndarray
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", evaluate_packing)

# Define gene-specific bounds for genetic operators (mutation and crossover)
# The individual is a flat array: [x1, y1, r1, x2, y2, r2, ...]
# So, the bounds must be repeated for each (x,y,r) triplet.
gene_low = ([BOUND_LOW_XY, BOUND_LOW_XY, BOUND_LOW_R] * N_CIRCLES)
gene_up = ([BOUND_UP_XY, BOUND_UP_XY, BOUND_UP_R] * N_CIRCLES)

# Register genetic operators with defined bounds
# cxSimulatedBinaryBounded: Crossover for real-valued genes within specified bounds
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=gene_low, up=gene_up, eta=20.0)
# mutPolynomialBounded: Mutation for real-valued genes within specified bounds.
# indpb is set to mutate, on average, one component (x, y, or r) per individual (inspired by Inspiration 2).
toolbox.register("mutate", tools.mutPolynomialBounded, low=gene_low, up=gene_up, indpb=1.0 / (3 * N_CIRCLES), eta=20.0)
# selTournament: Tournament selection operator
toolbox.register("select", tools.selTournament, tournsize=5) # Increased tournsize for stronger selection pressure (from 3)


def refine_solution_scipy(individual_params: np.ndarray) -> np.ndarray: # Added type hint
    """
    Refines the best solution found by the Evolutionary Algorithm using SciPy's
    local optimization (minimize). This step aims to "polish" the solution by
    strictly enforcing constraints and incrementally improving the sum of radii.
    """
    # Objective function for scipy.optimize: Minimize the negative sum of radii
    def objective(params):
        circles_arr = params.reshape((N_CIRCLES, 3))
        return -np.sum(circles_arr[:, 2])

    # Vectorized constraint function (inspired by Inspiration 2) for performance.
    def constraints_func(params):
        circles_arr = params.reshape((N_CIRCLES, 3))
        xs, ys, rs = circles_arr[:, 0], circles_arr[:, 1], circles_arr[:, 2]

        # 1. Radius positivity constraint (vectorized)
        radius_constraints = rs - EPSILON

        # 2. Containment constraints (vectorized)
        containment_constraints = np.concatenate([
            xs - rs,                      # x_i - r_i >= 0
            BOUND_UP_XY - xs - rs,        # 1 - x_i - r_i >= 0
            ys - rs,                      # y_i - r_i >= 0
            BOUND_UP_XY - ys - rs         # 1 - y_i - r_i >= 0
        ])

        # 3. Non-overlap constraints (vectorized using pdist from Inspiration 2, sqrt-free)
        coords = circles_arr[:, :2]
        if coords.shape[0] > 1:
            # Pairwise distances squared between centers
            dists_sq = pdist(coords, metric='sqeuclidean') # Use squared Euclidean distance
            # Pairwise sums of radii for all unique pairs, then squared
            radii_matrix = np.add.outer(rs, rs)
            indices = np.triu_indices(N_CIRCLES, k=1)
            radii_sums_sq = radii_matrix[indices]**2 # (r_i + r_j)^2
            
            overlap_constraints = dists_sq - radii_sums_sq # d_ij^2 - (r_i + r_j)^2 >= 0
        else:
            overlap_constraints = np.array([])

        return np.concatenate([radius_constraints, containment_constraints, overlap_constraints])

    # Define bounds for each variable (x, y, r) for scipy.optimize.minimize
    bounds_list = []
    for _ in range(N_CIRCLES):
        bounds_list.extend([(BOUND_LOW_XY, BOUND_UP_XY), (BOUND_LOW_XY, BOUND_UP_XY), (BOUND_LOW_R, BOUND_UP_R)])
    bnds = Bounds(*zip(*bounds_list)) # Use Bounds object for clarity and efficiency

    # Use NonlinearConstraint for a single, comprehensive constraint function
    # Lower bound is 0 (g(x) >= 0), upper bound is infinity
    nlc = NonlinearConstraint(constraints_func, 0, np.inf, jac='2-point')

    # The initial guess for the local optimizer is the best individual from the EA
    initial_guess = np.array(individual_params)

    # Perform local optimization using SLSQP (Sequential Least SQuares Programming)
    res = minimize(objective, initial_guess, method='SLSQP', bounds=bnds, constraints=[nlc],
                   options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}) # Increased maxiter and ftol for thorough refinement (from 3000 and 1e-9 to 2000 and 1e-9)

    if res.success:
        # Verify constraints are truly met after local optimization
        final_violations = constraints_func(res.x)
        # Allow a small tolerance for floating point errors
        if np.all(final_violations >= -EPSILON * 10):
            return res.x.reshape((N_CIRCLES, 3))
    
    # Fallback to the pre-optimization solution if local search fails or produces an invalid result
    # print(f"WARNING: SciPy optimization failed or produced invalid result: {res.message}. Returning EA's best solution.") # Uncomment for debugging
    return initial_guess.reshape((N_CIRCLES, 3))


def circle_packing26() -> np.ndarray:
    """
    Optimally places 26 non-overlapping circles within a unit square,
    maximizing the sum of their radii.
    This function utilizes a hybrid approach:
    1. An Evolutionary Algorithm (DEAP) to explore the vast search space and find
       promising general configurations.
    2. A local optimization step (SciPy's minimize) to refine the best solution
       from the EA, ensuring strict constraint adherence and pushing for local optimality.

    Returns:
        circles: np.ndarray of shape (26, 3), where each row is [x, y, r]
                 for a circle's center coordinates and radius.
    """
    # Set up random seeds at the start of the main function for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize the population for the Evolutionary Algorithm
    pop = toolbox.population(n=POP_SIZE) # Initial population size is POP_SIZE (which is MU)

    # Create a HallOfFame object to keep track of the best individual found during evolution.
    # `similar=np.array_equal` is crucial when Individual is an np.ndarray (from Inspiration 2).
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE, similar=np.array_equal)

    # Initial evaluation of the population
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    # --- Custom Evolutionary Loop with Adaptive Mutation (inspired by Inspiration 2) ---
    for gen in range(1, NGEN + 1):
        # Adaptive mutation: increase eta over generations to shift from exploration to exploitation.
        # Starting at 20.0 and increasing to 100.0 (20 + 80)
        current_eta = 20.0 + 80.0 * (gen / NGEN)**2
        # Re-register mutation with updated eta. This is necessary because eta is a parameter
        # to the mutPolynomialBounded function, not a mutable attribute of the operator itself.
        toolbox.register("mutate", tools.mutPolynomialBounded, low=gene_low, up=gene_up, indpb=1.0 / (3 * N_CIRCLES), eta=current_eta)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals to avoid modifying them directly
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # Invalidate the fitness of children as they have changed
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Invalidate the fitness of mutants
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring (MuPlusLambda style, where mu=lambda=POP_SIZE)
        pop[:] = offspring
        hof.update(pop) # Update the Hall of Fame with the new population

    # Retrieve the best individual (flattened numpy array of parameters) from the HallOfFame
    best_ea_individual = hof[0]

    # Apply the local refinement step using SciPy's optimizer
    # This aims to precisely adjust the circle parameters for maximum sum_radii
    # while strictly satisfying all geometric constraints.
    refined_circles = refine_solution_scipy(best_ea_individual)

    # Final clamp to ensure radii are non-negative, though the constraints should ideally handle this.
    refined_circles[:, 2] = np.maximum(EPSILON, refined_circles[:, 2])

    return refined_circles


# EVOLVE-BLOCK-END
