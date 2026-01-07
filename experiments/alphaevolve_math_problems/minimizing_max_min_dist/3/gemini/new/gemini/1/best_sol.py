# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing
from numba import jit
import time

# Helper function to calculate condensed distance matrix, JIT compiled
@jit(nopython=True)
def _calculate_distances_condensed(points: np.ndarray) -> np.ndarray:
    """
    Computes the condensed distance matrix for a set of 3D points.
    This implementation is Numba-compatible and equivalent to
    scipy.spatial.distance.pdist(points, 'euclidean') for this specific use case.
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

# Objective function for optimization, JIT compiled
@jit(nopython=True)
def _objective_function(points_flat: np.ndarray) -> float:
    """
    Calculates the negative of the min/max distance ratio for a flattened array of points.
    This function is minimized by the optimizer. It includes robust penalties for
    degenerate configurations, inspired by high-performing examples.
    """
    n_points = 14
    points = points_flat.reshape((n_points, 3))

    distances_condensed = _calculate_distances_condensed(points)
    
    # This case should not be hit with n_points=14, but is good practice for robustness.
    if distances_condensed.size == 0:
        return 1e9 # Use a large number instead of inf for better optimizer compatibility

    dmin = np.min(distances_condensed)
    dmax = np.max(distances_condensed)

    # Penalize configurations where points are nearly coincident (dmin is near zero).
    # This is a critical check to avoid degenerate solutions.
    if dmin < 1e-9:
        return 1e9

    # This case is mostly covered by the dmin check, but is an explicit guard
    # against all points collapsing to a single location.
    if dmax < 1e-9:
        return 1e9
    
    # We want to maximize dmin/dmax, so we minimize its negative
    return - (dmin / dmax)


from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Generates an optimal arrangement of exactly 14 points in 3D space by
    maximizing the min/max distance ratio. This implementation uses a two-phase
    strategy inspired by top-performing solutions:
    1. A comprehensive global search with `dual_annealing` using a Numba-JIT
       compiled objective function to find a promising region.
    2. A high-precision local refinement with `minimize('L-BFGS-B')` to polish
       the solution found in the global phase.
    The points are constrained to a unit cube [0,1]^3.

    Returns:
        points: np.ndarray of shape (14,3) containing the optimized coordinates.
    """

    n = 14
    d = 3
    
    # Define bounds for coordinates (unit cube [0,1]^3)
    bounds = [(0, 1) for _ in range(n * d)]

    # --- Phase 1: Global Search with dual_annealing ---
    # Parameters are aggressively tuned to fully utilize the time budget,
    # leveraging the speed of the Numba-compiled objective function.
    result_da = dual_annealing(
        func=_objective_function,
        bounds=bounds,
        maxiter=55000,          # Increased for more annealing steps
        maxfun=int(23_000_000), # Scaled up to use ~345s of the 360s budget
        seed=42,                # Ensures reproducibility
        no_local_search=True    # Disable internal local search; we use a better one next
    )

    # --- Phase 2: Local Refinement with L-BFGS-B ---
    # Start from the best point found by the global search to fine-tune.
    result_local = minimize(
        fun=_objective_function,
        x0=result_da.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8} # High-precision settings
    )

    # --- Final Selection ---
    # Compare the results and choose the configuration with the better (lower)
    # objective function value.
    if result_local.fun < result_da.fun:
        optimal_flat_coords = result_local.x
    else:
        optimal_flat_coords = result_da.x

    # Reshape the flattened array back to (n, d) for the final output.
    optimized_points = optimal_flat_coords.reshape((n, d))

    return optimized_points


# EVOLVE-BLOCK-END
