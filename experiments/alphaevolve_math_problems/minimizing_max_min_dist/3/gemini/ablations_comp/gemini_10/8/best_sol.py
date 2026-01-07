# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial.distance as dist
from scipy.optimize import minimize
import math


# --- Constants ---
N_POINTS = 14
DIMENSIONS = 3
RANDOM_SEED = 42 # Centralized seed for reproducibility


def calculate_ratio(points: np.ndarray) -> float:
    """
    Helper function to calculate dmin/dmax ratio.
    Incorporates robust handling of near-zero distances for dmin, inspired by Inspiration 1/2/3.
    """
    pairwise_distances = dist.pdist(points)
    if pairwise_distances.size == 0:
        return 0.0
    
    # Filter out extremely small distances to get a meaningful dmin, preventing numerical issues.
    # Using 1e-9 as seen in inspiration programs.
    non_zero_distances = pairwise_distances[pairwise_distances > 1e-9]
    
    if len(non_zero_distances) == 0: 
        # If all distances are effectively zero, it means points are collapsed, so ratio is 0.0.
        return 0.0

    dmin = np.min(non_zero_distances) # Use filtered distances for dmin
    dmax = np.max(pairwise_distances) # dmax should still consider all distances

    if dmax < 1e-9: # Safeguard against dmax being too small
        return 0.0
    return dmin / dmax


def objective_function(points_flat: np.ndarray, n: int, d: int) -> float:
    """Objective function for scipy.minimize: returns -dmin/dmax."""
    points = points_flat.reshape(n, d)
    return -calculate_ratio(points)


def generate_fibonacci_sphere_points(n_points: int, center: tuple, radius: float, seed: int) -> np.ndarray:
    """
    Generates points on a sphere using the Fibonacci spiral method, then scales, translates,
    and perturbs them. Uses a dedicated RNG for consistency and reproducibility.
    """
    rng = np.random.RandomState(seed) # Use a dedicated RNG for this function
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        r_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r_at_y
        z = np.sin(theta) * r_at_y
        points.append([x, y, z])

    points_np = np.array(points) * radius
    points_np += np.array(center)
    
    # Perturbation strength adjusted relative to the radius.
    perturbation_strength = 0.01 * radius
    points_np += rng.uniform(-perturbation_strength, perturbation_strength, points_np.shape)
    
    # Do NOT clip here. Let the optimization handle bounds in the main function.
    return points_np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D space to maximize the min/max distance ratio using a three-stage strategy,
    adopted from the inspiration programs for improved performance and ratio.

    1.  **Structured Initialization**: Points start on a Fibonacci sphere for a high-quality guess,
        now generated within a wider space for more flexibility.
    2.  **Global Search (Custom Simulated Annealing)**: A custom SA algorithm explores the solution space efficiently,
        now operating in a wider, unconstrained space using a dedicated random number generator.
    3.  **Local Refinement (L-BFGS-B)**: The best SA solution is polished to a precise local optimum.
        This also operates in the wider space.
    4.  **Final Normalization**: The optimized points are scaled and shifted to fit within the [0,1]^3 unit cube.

    This hybrid approach combines the speed of SA with the precision of gradient-based methods,
    and leverages the benefit of optimizing in a scale-invariant manner before final normalization.

    Returns:
        points: np.ndarray of shape (14,3) containing the optimized (x,y,z) coordinates.
    """
    n = N_POINTS
    d = DIMENSIONS
    seed = RANDOM_SEED
    
    # Use a dedicated RNG for the main optimization loop for better control and reproducibility
    rng = np.random.RandomState(seed)

    # --- Stage 0: Structured Initial Point Generation ---
    # Generate points around the origin with a certain radius, to be optimized in wider bounds.
    # Using center=(0,0,0) and radius=1.0 for a good spread within typical wider bounds.
    initial_points = generate_fibonacci_sphere_points(
        n_points=n, center=(0.0, 0.0, 0.0), radius=1.0, seed=seed
    )
    
    # Define wider bounds for optimization, allowing the optimizer more freedom.
    # The ratio is scale-invariant, so optimizing in a larger box and then normalizing
    # is often better, as seen in Inspiration 1 and 2.
    optimization_bounds_val = 2.0 # Example: points can range from -2.0 to 2.0 in each coord
    bounds = [(-optimization_bounds_val, optimization_bounds_val)] * (n * d)

    # --- Stage 1: Global Search with Custom Simulated Annealing ---
    current_points = initial_points.copy()
    current_ratio = calculate_ratio(current_points)
    best_points = current_points.copy()
    best_ratio = current_ratio

    max_sa_iterations = 2_000_000 # Significantly increased iterations to leverage time budget and wider search space
    initial_temperature = 0.5 # Increased for more aggressive exploration in the wider search space
    cooling_rate = 0.99995 # Keep as is, provides good balance
    perturbation_scale = 0.05 # Increased perturbation for more aggressive exploration in wider space, relative to optimization_bounds_val
    temperature = initial_temperature

    for _ in range(max_sa_iterations):
        candidate_points = current_points.copy()
        p_idx = rng.randint(n) # Use dedicated RNG
        perturbation = rng.uniform(-perturbation_scale, perturbation_scale, d) # Use dedicated RNG
        candidate_points[p_idx] += perturbation
        
        # Clip to the wider optimization bounds during SA
        candidate_points[p_idx] = np.clip(candidate_points[p_idx], -optimization_bounds_val, optimization_bounds_val)

        candidate_ratio = calculate_ratio(candidate_points)

        # Metropolis acceptance criterion
        if candidate_ratio > current_ratio or rng.rand() < math.exp((candidate_ratio - current_ratio) / temperature): # Use dedicated RNG
            current_points = candidate_points
            current_ratio = candidate_ratio
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = current_points.copy()

        temperature *= cooling_rate
        if temperature < 1e-6: # Prevent temperature from going to effectively zero
             temperature = 1e-6

    # --- Stage 2: Local Refinement with L-BFGS-B ---
    refined_result = minimize(
        fun=objective_function,
        x0=best_points.flatten(),
        args=(n, d),
        method='L-BFGS-B',
        bounds=bounds, # Use the wider bounds for L-BFGS-B
        options={
            'maxiter': 10000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8, 'eps': 1e-9
        }
    )
    
    # Compare the negative objective function value (-refined_result.fun) with the positive best_ratio.
    # The L-BFGS-B might find a slightly better local optimum than the SA's best_ratio.
    # Ensure -refined_result.fun is finite for comparison.
    refined_ratio = -refined_result.fun if np.isfinite(refined_result.fun) else 0.0

    if refined_ratio > best_ratio:
        final_x = refined_result.x
    else:
        final_x = best_points.flatten() # Fallback to SA's best if local refinement didn't improve

    optimized_points = final_x.reshape(n, d)
    
    # --- Stage 3: Final Normalization to [0,1]^3 ---
    # Shift points so the minimum coordinate is 0, then scale so the maximum extent is 1.
    min_coords = np.min(optimized_points, axis=0)
    optimized_points -= min_coords 
    max_coord_val = np.max(optimized_points) 
    if max_coord_val > 1e-9: # Safeguard against division by zero
        optimized_points /= max_coord_val 
    
    # Clip for floating point safety, ensuring strict adherence to the [0,1] range.
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points


# EVOLVE-BLOCK-END
