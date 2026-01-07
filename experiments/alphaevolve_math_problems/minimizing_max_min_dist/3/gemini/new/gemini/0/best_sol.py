# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize # Added minimize for local refinement
from numba import jit # Numba is essential for performance, already present


# Helper function to calculate condensed distance matrix, JIT compiled
@jit(nopython=True)
def _calculate_distances_condensed(points: np.ndarray) -> np.ndarray:
    """
    Computes the condensed distance matrix for a set of 3D points.
    This implementation is Numba-compatible and equivalent to
    scipy.spatial.distance.pdist(points, 'euclidean') for this specific use case,
    providing superior performance.
    """
    n_points = points.shape[0]
    num_distances = n_points * (n_points - 1) // 2
    distances = np.empty(num_distances, dtype=points.dtype)
    
    k = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d_sq = 0.0
            for dim in range(points.shape[1]):
                d_sq += (points[i, dim] - points[j, dim])**2
            distances[k] = np.sqrt(d_sq)
            k += 1
    return distances

# Objective function for optimization, JIT compiled and robustified
@jit(nopython=True)
def _objective_function(points_flat: np.ndarray) -> float:
    """
    Objective function for the optimization.
    Takes a flattened array of 3D point coordinates and returns the negative
    of the min/max distance ratio. This function is designed to be minimized.
    It is Numba-compiled for performance and includes robust penalty handling
    for degenerate configurations, as seen in the best inspiration programs.
    """
    n_points = 14
    n_dimensions = 3
    points = points_flat.reshape((n_points, n_dimensions))

    distances = _calculate_distances_condensed(points)

    # If no distances could be computed (e.g., n_points < 2),
    # or if all points are identical leading to dmax=0, return a high penalty.
    if distances.size == 0: # This should not happen with N=14
        return 1e9 # Use a large finite number instead of np.inf for broader optimizer compatibility

    dmin = np.min(distances)
    dmax = np.max(distances)

    # Penalize configurations where points are too close (dmin effectively zero)
    # or all points are effectively coincident (dmax effectively zero).
    # Both lead to a dmin/dmax ratio of 0 (or ill-defined), which is the worst for maximization.
    if dmin < 1e-9: 
        return 1e9
    
    # If dmax is very small, it implies all points are very close to each other.
    # This also leads to a poor ratio, so penalize.
    if dmax < 1e-9: # This case is mostly covered by dmin < 1e-9, but kept for explicit robustness
        return 1e9
    
    # The objective is to maximize dmin/dmax, so we return its negative for minimization.
    return -dmin / dmax


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Generates an optimal arrangement of exactly 14 points in 3D space,
    maximizing the ratio of minimum distance to maximum distance between all point pairs.

    This implementation adopts a two-phase optimization strategy inspired by high-performing
    inspiration programs:
    1. An extensive global search using `dual_annealing` with a Numba-JIT compiled
       objective function for speed.
    2. A local refinement using `scipy.optimize.minimize` (L-BFGS-B method) starting
       from the best point found by the global search, to polish the solution.
    Points are constrained to the unit cube [0,1]^3.

    Returns:
        np.ndarray: A (14, 3) NumPy array of the optimized point coordinates.
    """
    n_points = 14
    n_dimensions = 3
    num_coords = n_points * n_dimensions # Total number of coordinates: 14 * 3 = 42

    # Define the bounds for each coordinate: [0, 1] for x, y, z of each point.
    bounds = [(0, 1)] * num_coords

    # --- Phase 1: Global optimization with dual_annealing ---
    # Parameters are aggressively tuned to fully utilize the time budget,
    # leveraging the speed of the Numba-compiled objective function.
    # Increased maxfun and maxiter to use more of the available 360s budget.
    result_da = dual_annealing(
        _objective_function,
        bounds,
        seed=42,                # For reproducibility
        maxiter=67000,          # Increased for more annealing steps (scaled for new maxfun)
        visit=2,                # Number of function calls per temperature step
        maxfun=int(28_000_000), # Scaled up to use ~350s of the 360s budget
        no_local_search=True    # Disable internal local search; we use a dedicated one next
    )

    # --- Phase 2: Local refinement using L-BFGS-B ---
    # This step starts from the best point found by dual_annealing to fine-tune
    # the solution to a higher precision.
    result_local = minimize(
        _objective_function,
        result_da.x, # Start local search from the global optimum found by dual_annealing
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # Aggressive options for precision
    )

    # --- Final Selection ---
    # Select the best result from the two optimization phases.
    # Since the objective function returns the negative of the ratio,
    # a smaller (more negative) 'fun' value indicates a better ratio.
    if result_local.fun < result_da.fun:
        optimal_flat_coords = result_local.x
    else:
        optimal_flat_coords = result_da.x
    
    # Reshape the flattened optimized coordinates back into a (14, 3) array.
    optimized_points = optimal_flat_coords.reshape((n_points, n_dimensions))

    return optimized_points


# EVOLVE-BLOCK-END
