# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, minimize # Added minimize for local refinement
# Removed qmc as it's being replaced by a perturbed grid initialization

# Define the objective function at the module level for good practice and compatibility.
def _calculate_negative_min_max_ratio(coords_flat: np.ndarray) -> float:
    """
    Calculates the negative of the min/max distance ratio.
    The optimizer will minimize this value, effectively maximizing the ratio.
    """
    n = 16  # Fixed number of points for this problem
    d = 2   # Fixed dimensions for this problem
    
    # Reshape the 1D array of coordinates into a (n, d) array of points
    points = coords_flat.reshape((n, d))

    # Calculate all pairwise Euclidean distances.
    distances = pdist(points)

    if distances.size == 0:
        return np.inf

    dmin = np.min(distances)
    dmax = np.max(distances)
    
    # Penalize collapsed or overlapping points heavily.
    # Using np.isclose is more robust for floating-point comparisons to zero.
    if np.isclose(dmin, 0.0):
        return np.inf

    return -dmin / dmax

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    This implementation uses a two-stage optimization strategy:
    1.  Global Search: `dual_annealing` with `no_local_search=True` and Halton initialization
        for broad exploration.
    2.  Local Refinement: `minimize` with `L-BFGS-B` to polish the global solution for precision.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    n = 16  # Number of points
    d = 2   # Dimensions
    num_coordinates = n * d # Total number of variables for the optimizer (16 * 2 = 32)

    # Define the bounds for each coordinate. All x and y values must be within [0, 1].
    bounds = [(0, 1)] * num_coordinates

    # Set the random seed for reproducibility.
    np.random.seed(42)

    # Generate an initial guess using a perturbed grid (inspired by the best-performing program).
    # A 4x4 grid is an excellent heuristic for N=16. Adding a small perturbation
    # breaks the symmetry and provides a superior starting point for optimization.
    side_length = int(np.sqrt(n))
    grid_coords = np.linspace(0.5 / side_length, 1 - 0.5 / side_length, side_length)
    grid_points = np.array([[x, y] for x in grid_coords for y in grid_coords])
    
    # Add a small random perturbation to each point.
    perturbation_strength = 0.05 
    perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, (n, d))
    initial_guess_2d = np.clip(grid_points + perturbation, 0, 1)
    
    # Flatten the 2D array into a 1D array for the optimizer.
    x0_perturbed_grid = initial_guess_2d.flatten()

    # Step 1: Global Search with dual_annealing
    # `no_local_search=True` makes dual_annealing a pure global explorer.
    # The iteration count is increased to fully utilize the time budget, and additional
    # parameters from inspirations are added to fine-tune the search.
    result_global = dual_annealing(
        func=_calculate_negative_min_max_ratio,
        bounds=bounds,
        x0=x0_perturbed_grid, # Use the superior perturbed grid for initial guess
        seed=42,              # Ensures reproducibility of the optimization run
        maxiter=50000,        # Further increased iterations to use the full time budget
        maxfun=2_000_000,     # Add a high function evaluation limit as a safeguard
        initial_temp=6000.0,  # Set a higher initial temp for broader initial exploration
        no_local_search=True, # Disable internal local search for faster global exploration
    )

    # The result from dual_annealing is a good candidate from the global search.
    global_solution = result_global.x

    # Step 2: Local Refinement (Polishing)
    # Use a gradient-based local optimizer (L-BFGS-B) to refine the solution
    # found by the global search, achieving higher precision.
    result_local = minimize(
        fun=_calculate_negative_min_max_ratio,
        x0=global_solution,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # Tuned parameters for precise convergence
    )
    
    # Extract the final polished 1D coordinate array.
    optimal_flat_points = result_local.x

    # Reshape the optimal flattened coordinates back into a (16, 2) array.
    optimal_points = optimal_flat_points.reshape((n, d))

    # Clip the final points to ensure they are strictly within the [0, 1] bounds.
    optimal_points = np.clip(optimal_points, 0, 1)

    return optimal_points


# EVOLVE-BLOCK-END
