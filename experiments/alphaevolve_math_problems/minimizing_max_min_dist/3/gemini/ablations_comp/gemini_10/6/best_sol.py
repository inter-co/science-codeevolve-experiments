# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from numba import jit # Added Numba for performance

# --- Constants ---
N_POINTS = 14
DIMENSIONS = 3
RANDOM_SEED = 42

# --- JIT-COMPILED DISTANCE HELPER ---
@jit(nopython=True)
def _pairwise_distances_numba(points: np.ndarray) -> np.ndarray:
    """
    Calculates all pairwise Euclidean distances between points using Numba for speed.
    """
    n = points.shape[0]
    num_pairs = n * (n - 1) // 2
    distances = np.empty(num_pairs, dtype=points.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for dim in range(points.shape[1]):
                diff = points[i, dim] - points[j, dim]
                dist_sq += diff * diff
            distances[k] = np.sqrt(dist_sq)
            k += 1
    return distances

# --- Shared Helper ---
def _calculate_ratio(points: np.ndarray) -> float:
    if len(points) <= 1: return 0.0
    distances = _pairwise_distances_numba(points) # Using Numba-optimized distance calculation
    if distances.size == 0: return 0.0
    d_min, d_max = np.min(distances), np.max(distances)
    # Return 0.0 for degenerate cases (very small d_max or d_min).
    # The objective functions will convert this 0.0 to np.inf for minimization.
    if d_max < 1e-9 or d_min < 1e-9:
        return 0.0
    return d_min / d_max

# --- Spherical Optimization Functions (for Seeding Stage) ---
def _spherical_to_cartesian_symmetric(angles_flat_free: np.ndarray) -> np.ndarray:
    points_cartesian = np.zeros((N_POINTS, 3))
    points_cartesian[0, :] = [0, 0, 1]
    theta_1 = angles_flat_free[0]
    points_cartesian[1, 0] = np.sin(theta_1)
    points_cartesian[1, 2] = np.cos(theta_1)
    points_cartesian[1, 1] = 0.0 # Explicitly set y-coordinate to 0 for phi=0
    if N_POINTS > 2:
        remaining_angles = angles_flat_free[1:].reshape(N_POINTS - 2, 2)
        phi, theta = remaining_angles[:, 0], remaining_angles[:, 1]
        points_cartesian[2:, 0] = np.sin(theta) * np.cos(phi)
        points_cartesian[2:, 1] = np.sin(theta) * np.sin(phi)
        points_cartesian[2:, 2] = np.cos(theta)
    return points_cartesian

def _objective_ratio_spherical(angles_flat_free: np.ndarray) -> float:
    points = _spherical_to_cartesian_symmetric(angles_flat_free)
    ratio = _calculate_ratio(points)
    # Convert 0.0 ratio (degenerate case) to np.inf for minimization
    if ratio == 0.0:
        return np.inf
    return -ratio

def _find_spherical_seed() -> np.ndarray:
    np.random.seed(RANDOM_SEED)
    bounds = [(0, np.pi)] + [(0, 2 * np.pi), (0, np.pi)] * (N_POINTS - 2)
    seed_result = dual_annealing(func=_objective_ratio_spherical, bounds=bounds,
                                 maxiter=20000, maxfun=400000, seed=RANDOM_SEED) # Further increased iterations for better seed
    return _spherical_to_cartesian_symmetric(seed_result.x)

# --- Cartesian Optimization Functions (for Refinement Stages) ---
def _objective_ratio_cartesian(flat_points: np.ndarray) -> float:
    points = flat_points.reshape((N_POINTS, DIMENSIONS))
    ratio = _calculate_ratio(points)
    # Convert 0.0 ratio (degenerate case) to np.inf for minimization
    if ratio == 0.0:
        return np.inf
    return -ratio

def _objective_potential(flat_points: np.ndarray, power: int) -> float:
    points = flat_points.reshape((N_POINTS, DIMENSIONS))
    distances = _pairwise_distances_numba(points) # Using Numba-optimized distance calculation
    if np.any(distances < 1e-9): return np.inf
    return np.sum(1.0 / (distances ** power))

# --- Main Multi-Stage Optimization Function ---
def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in 3D using a multi-stage hybrid approach:
    1. Spherical Seeding: A fast `dual_annealing` run on a reduced-dimension spherical
       coordinate space generates a high-quality, rotationally-invariant starting point.
    2. Global Cartesian Search (DA): The main `dual_annealing` search explores the full
       Cartesian space, seeded by the result of stage 1, to find a promising basin.
    3. Local Potential Refinement (L-BFGS-B): A gradient-based local optimizer minimizes
       a smooth potential energy function to aggressively push points apart.
    4. Final Ratio Polish (SLSQP): A final local optimization directly on the dmin/dmax
       ratio to achieve maximum precision.
    """
    np.random.seed(RANDOM_SEED)

    # --- Seeding Stage: Generate a high-quality seed via spherical optimization ---
    x0_seed_points = _find_spherical_seed()
    x0_seed = x0_seed_points.flatten()

    bounds = [(-5.0, 5.0)] * (N_POINTS * DIMENSIONS)

    # --- Stage 1: Global Cartesian Search (Dual Annealing on Ratio) ---
    global_result = dual_annealing(
        func=_objective_ratio_cartesian, bounds=bounds, x0=x0_seed,
        maxiter=80000, maxfun=1600000, initial_temp=2e4, seed=RANDOM_SEED) # Slightly increased iterations, assuming Numba speedup

    # --- Stage 2: Local Potential Refinement (L-BFGS-B) ---
    potential_result = minimize(
        fun=_objective_potential, x0=global_result.x, args=(32,),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})

    # --- Stage 3: Final Ratio Polish (SLSQP) ---
    final_result = minimize(
        fun=_objective_ratio_cartesian, x0=potential_result.x,
        method='SLSQP', bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'eps': 1e-9})
    
    # --- Final Normalization to [0, 1]^3 Cube ---
    points_unscaled = final_result.x.reshape((N_POINTS, DIMENSIONS))
    min_coords = np.min(points_unscaled, axis=0)
    points_shifted = points_unscaled - min_coords
    max_extent = np.max(points_shifted)
    if max_extent < 1e-9: return np.random.rand(N_POINTS, DIMENSIONS)
    normalized_points = points_shifted / max_extent
    return np.clip(normalized_points, 0, 1)


# EVOLVE-BLOCK-END
