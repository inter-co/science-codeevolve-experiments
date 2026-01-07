# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, dual_annealing
from numba import jit
import random

# Setting random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Define the number of points and dimensions globally
N_POINTS = 14
N_DIMENSIONS = 3

@jit(nopython=True, fastmath=True)
def objective_spherical(flat_spherical_coords: np.ndarray) -> float:
    """
    Calculates -dmin/dmax from spherical coordinates. JIT-compiled for speed.
    This function is adapted from Inspiration Program 2 & 3. It performs fast
    spherical-to-cartesian conversion and distance calculations internally.
    It uses squared distances to avoid costly sqrt operations in the inner loop.
    """
    spherical_coords = flat_spherical_coords.reshape((N_POINTS, N_DIMENSIONS))
    
    # Fast spherical to cartesian conversion
    cartesian_points = np.empty((N_POINTS, N_DIMENSIONS))
    for i in range(N_POINTS):
        r, theta, phi = spherical_coords[i, 0], spherical_coords[i, 1], spherical_coords[i, 2]
        
        sin_phi = np.sin(phi)
        cartesian_points[i, 0] = r * sin_phi * np.cos(theta)
        cartesian_points[i, 1] = r * sin_phi * np.sin(theta)
        cartesian_points[i, 2] = r * np.cos(phi)

    # Fast pdist stats using squared distances
    d_min_sq = np.inf
    d_max_sq = -1.0
    for i in range(N_POINTS):
        for j in range(i + 1, N_POINTS):
            d_sq = 0.0
            for k in range(N_DIMENSIONS):
                tmp = cartesian_points[i, k] - cartesian_points[j, k]
                d_sq += tmp * tmp
            
            if d_sq < d_min_sq:
                d_min_sq = d_sq
            if d_sq > d_max_sq:
                d_max_sq = d_sq
    
    if d_max_sq < 1e-20:
        return 1.0 # High cost for collapsed points, avoids division by zero

    return -np.sqrt(d_min_sq / d_max_sq)


def fibonacci_sphere(samples: int) -> np.ndarray:
    """
    Generates points on a unit sphere using the Fibonacci lattice method.
    Adapted from Inspiration Program 2 & 3 for superior initial point distribution.
    """
    points = np.zeros((samples, N_DIMENSIONS))
    phi_angle = np.pi * (3. - np.sqrt(5.))  # Golden angle

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = [x, y, z]
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D to maximize the min/max distance ratio using a hybrid
    global-local optimization strategy in a spherical coordinate system. This method
    is heavily inspired by the successful techniques in Inspiration Programs 2 and 3.

    1.  **Spherical Coordinates**: Optimization is performed in (r, theta, phi) space
        within a unit ball, a more natural geometry for this dispersion problem.
    2.  **JIT-Compiled Objective**: A `numba` JIT-compiled objective function provides
        massive speed-up by accelerating the coordinate conversion and distance calculations.
    3.  **Fibonacci Initialization**: A highly uniform `fibonacci_sphere` distribution
        is used as the starting guess, providing a strong initial configuration.
    4.  **Hybrid Optimization**: A global `dual_annealing` search is followed by a
        high-precision local `SLSQP` refinement to find a high-quality optimum.
    """
    # --- Helper functions for coordinate conversion ---
    def cartesian_to_spherical(cartesian_coords: np.ndarray) -> np.ndarray:
        spherical = np.zeros_like(cartesian_coords)
        x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        # Ensure theta is in [0, 2*pi] for consistency with bounds
        theta[theta < 0] += 2 * np.pi
        
        # Clip to avoid domain errors with arccos due to floating point inaccuracies
        r_safe = np.where(r == 0, 1e-9, r)
        phi = np.arccos(np.clip(z / r_safe, -1.0, 1.0))

        spherical[:, 0] = r
        spherical[:, 1] = theta
        spherical[:, 2] = phi
        return spherical

    def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
        cartesian = np.zeros_like(spherical_coords)
        r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
        
        sin_phi = np.sin(phi)
        cartesian[:, 0] = r * sin_phi * np.cos(theta)
        cartesian[:, 1] = r * sin_phi * np.sin(theta)
        cartesian[:, 2] = r * np.cos(phi)
        return cartesian

    # --- Setup for Optimization in Spherical Coordinates ---
    # Bounds for each point: r in [0,1], theta in [0, 2*pi], phi in [0, pi]
    bounds = []
    for _ in range(N_POINTS):
        bounds.extend([(0.0, 1.0), (0.0, 2 * np.pi), (0.0, np.pi)])

    # Initialize points on a sphere of radius 0.5 within the unit ball
    initial_cartesian = fibonacci_sphere(samples=N_POINTS) * 0.5
    initial_guess_spherical = cartesian_to_spherical(initial_cartesian)
    initial_guess_flat = initial_guess_spherical.flatten()
    
    # --- Stage 1: Global Optimization using dual_annealing ---
    # Increased budget given the fast JIT'd objective function
    global_result = dual_annealing(
        objective_spherical, 
        bounds, 
        x0=initial_guess_flat,
        seed=SEED,
        maxiter=10000,
        maxfun=3_000_000, # More function evaluations
        minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'ftol': 1e-8}}
    )

    # --- Stage 2: Local Refinement using SLSQP ---
    # Refine the best result from the global search for higher precision
    local_result = minimize(
        objective_spherical, 
        global_result.x,
        method='SLSQP',
        bounds=bounds, 
        options={'disp': False, 'maxiter': 50000, 'ftol': 1e-13}
    )

    # --- Post-processing: Convert back and normalize to [0,1]^3 cube ---
    optimized_spherical = local_result.x.reshape((N_POINTS, N_DIMENSIONS))
    optimized_cartesian = spherical_to_cartesian(optimized_spherical)

    # Normalize the final configuration to fit within the [0,1]^3 unit cube,
    # preserving the optimal distance ratio.
    min_coords = optimized_cartesian.min(axis=0)
    points_shifted = optimized_cartesian - min_coords
    scale_factor = points_shifted.max()
    if scale_factor > 1e-9:
        normalized_points = points_shifted / scale_factor
    else:
        # Avoid division by zero if all points are collapsed
        normalized_points = points_shifted
        
    return normalized_points


# EVOLVE-BLOCK-END
