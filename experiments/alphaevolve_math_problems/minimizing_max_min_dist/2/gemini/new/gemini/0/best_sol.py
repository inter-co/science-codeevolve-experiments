# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize # Added minimize for local refinement
from scipy.spatial.distance import pdist
from scipy.stats import qmc # For Sobol sequence initialization


def _calculate_negative_min_max_ratio(flat_points: np.ndarray) -> float:
    """
    Objective function for global optimization, designed to be minimized.

    Takes a 1D NumPy array of point coordinates, reshapes it, calculates the
    pairwise distances, and returns the negative of the dmin/dmax ratio.
    Penalizes invalid configurations with a large positive value.

    Args:
        flat_points: A 1D array representing [x0, y0, x1, y1, ...].

    Returns:
        The negative of the dmin/dmax ratio, or a large penalty value (np.inf).
    """
    n_points = 16 # Fixed number of points for this problem
    n_dims = 2
    points = flat_points.reshape((n_points, n_dims))

    # Handle edge case: if there are fewer than 2 points, distances cannot be calculated.
    if points.shape[0] < 2:
        return np.inf

    # Calculate all unique pairwise Euclidean distances efficiently
    distances = pdist(points, 'euclidean')

    # If no distances could be calculated (e.g., all points identical from start, N<2)
    if distances.size == 0:
        return np.inf

    dmin = np.min(distances)
    dmax = np.max(distances)

    # Heavily penalize arrangements where points are effectively overlapping.
    # Using np.isclose is more robust for floating-point comparisons.
    if np.isclose(dmin, 0.0):
        return np.inf
    
    # We want to maximize the ratio, so we minimize its negative.
    ratio = dmin / dmax
    return -ratio


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Optimizes the placement of 16 points in 2 dimensions to maximize the ratio
    of minimum distance to maximum distance between all point pairs.
    The points are constrained within the unit square [0,1]x[0,1].

    This implementation uses a two-stage optimization strategy:
    1. Global Search: `dual_annealing` for broad exploration.
    2. Local Refinement: `minimize` (L-BFGS-B) for precise convergence.
    It leverages insights from inspiration programs for better performance,
    including an informed initial guess.

    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of
                the optimized 16 points within the unit square [0,1]x[0,1].
    """

    n_points = 16  # Number of points
    n_dims = 2   # Dimensions (2D)
    num_coordinates = n_points * n_dims # Total number of coordinates (16 * 2 = 32)

    # Set random seed for reproducibility.
    np.random.seed(42)

    # Generate an initial guess using a perturbed grid. For N=16, a 4x4 grid is a strong heuristic.
    # Adding a small random perturbation helps break symmetry and provides a superior starting point.
    side_length = int(np.sqrt(n_points)) # Should be 4 for N=16
    grid_coords = np.linspace(0.5 / side_length, 1 - 0.5 / side_length, side_length)
    grid_points = np.array([[x, y] for x in grid_coords for y in grid_coords])
    
    # Add a small random perturbation to each point for exploration.
    perturbation_strength = 0.05 
    perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, (n_points, n_dims))
    initial_guess_2d = np.clip(grid_points + perturbation, 0, 1)
    
    # Flatten the 2D array into a 1D array for the optimizer.
    initial_flat_points = initial_guess_2d.flatten()

    # Define the bounds for each coordinate. All x and y coordinates must be within [0, 1].
    bounds = [(0.0, 1.0)] * num_coordinates

    # Step 1: Global Search with dual_annealing
    # `no_local_search=True` makes dual_annealing focus purely on global exploration.
    # `maxiter` is set to allow for thorough global search within the computational budget.
    result_global = dual_annealing(
        func=_calculate_negative_min_max_ratio,
        bounds=bounds,
        x0=initial_flat_points, # Use the perturbed grid for initial guess
        seed=42,                # Ensures reproducibility of the optimization run
        maxiter=60000,          # Further increased iterations for even more thorough global search
        maxfun=2_500_000,       # Increased function evaluation limit as a safeguard
        initial_temp=6000.0,    # Retain high initial temperature for broad initial exploration
        no_local_search=True,   # Retain: Disable internal local search for faster global exploration
    )

    # The result from dual_annealing provides a good candidate solution from the global search.
    global_solution = result_global.x

    # Step 2: Local Refinement (Polishing)
    # Use a gradient-based local optimizer (L-BFGS-B) to refine the solution
    # found by the global search, achieving higher precision.
    result_local = minimize(
        fun=_calculate_negative_min_max_ratio,
        x0=global_solution,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # Tuned options for local convergence (from Inspiration 2)
    )
    
    # Extract the final polished 1D coordinate array.
    optimized_flat_points = result_local.x

    # Reshape the 1D array back into a (n, d) array of points.
    optimized_points = optimized_flat_points.reshape((n_points, n_dims))

    # Clip the final coordinates to ensure they are strictly within the [0, 1] bounds,
    # correcting for any minor floating-point overruns by the optimizer.
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points


# EVOLVE-BLOCK-END
