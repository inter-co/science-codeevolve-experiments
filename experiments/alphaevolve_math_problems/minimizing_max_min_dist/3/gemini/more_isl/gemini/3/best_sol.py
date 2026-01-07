# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist # Added cdist for initial guess generation
from scipy.optimize import minimize # For local refinement
from numba import njit # For Numba acceleration

# Numba-accelerated pairwise distance calculation
@njit
def fast_pdist_jit(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    num_dists = n * (n - 1) // 2
    if num_dists == 0: return np.empty(0, dtype=np.float64)
    
    dists = np.empty(num_dists, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = 0.0
            for dim in range(pts.shape[1]):
                diff = pts[i, dim] - pts[j, dim]
                d_sq += diff * diff
            dists[k] = np.sqrt(d_sq)
            k += 1
    return dists

# Numba-accelerated SA energy function (minimizes d_max / d_min)
@njit
def calculate_energy_jit(pts: np.ndarray) -> float:
    if pts.shape[0] < 2: return np.inf

    # Center the points for scale-invariant evaluation
    mean_pt = np.empty(pts.shape[1], dtype=np.float64)
    for i in range(pts.shape[1]):
        mean_pt[i] = np.mean(pts[:, i])
    centered_pts = pts - mean_pt
    
    distances = fast_pdist_jit(centered_pts)
    if distances.size == 0: return np.inf

    d_min = np.min(distances)
    if d_min < 1e-9: return np.inf # Infinitely bad state (coincident points)

    d_max = np.max(distances)
    if d_max < 1e-9: return np.inf # All points collapsed

    return d_max / d_min # We want to minimize this ratio

# Differentiable objective function using logsumexp for smooth min/max
# Inspired by Inspiration Program 1, adapted for unconstrained optimization.
# This function minimizes -(d_min/d_max), thus maximizing the ratio.
def _objective_dmin_dmax_ratio_smoothed(x_flat_coords: np.ndarray, n_points: int, n_dim: int, k_min: float, k_max: float) -> float:
    """
    Objective function to minimize the negative of the smoothed dmin/dmax ratio.
    This directly targets the problem's primary objective (maximizing dmin/dmax)
    using differentiable approximations of min and max (via logsumexp),
    suitable for gradient-based optimization.
    """
    points = x_flat_coords.reshape((n_points, n_dim))
    
    # The ratio is translation-invariant, so no explicit centering is needed here.
    # It is also scale-invariant, making it ideal for unconstrained optimization.
    distances = pdist(points)
    
    if distances.size == 0: return np.inf
    # Handle very small distances to prevent numerical instability.
    distances = np.maximum(distances, 1e-9)

    # Smooth min calculation: softmin(x) = - (1/k) * log(sum(exp(-k*x)))
    log_sum_exp_neg_k_min = np.logaddexp.reduce(-k_min * distances)
    d_min_smoothed = - (1.0 / k_min) * log_sum_exp_neg_k_min

    # Smooth max calculation: softmax(x) = (1/k) * log(sum(exp(k*x)))
    log_sum_exp_pos_k_max = np.logaddexp.reduce(k_max * distances)
    d_max_smoothed = (1.0 / k_max) * log_sum_exp_pos_k_max

    if d_max_smoothed < 1e-10:
        return np.inf # Penalize collapsed configurations

    # Objective: minimize -(d_min/d_max)
    return - (d_min_smoothed / d_max_smoothed)

def _run_force_directed_pre_opt(initial_points: np.ndarray, num_points: int, dimensions: int) -> np.ndarray:
    """
    Implements a fast physics-based pre-optimization using a strong 1/r^4 repulsive force.
    The cloud is dynamically centered and scaled to ensure stability.
    Adapted from Inspiration Program 1.
    """
    points = initial_points.copy()
    time_step = 0.02
    num_iterations = 5000 
    damping_factor = 0.999

    for iteration in range(num_iterations):
        current_time_step = time_step * (damping_factor ** iteration)
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        dists_clipped = np.clip(dists, 1e-7, None)

        force_magnitudes_per_dist_unit = 0.1 / (dists_clipped**3) # F ~ 1/d^2 (electrostatic-like), so multiply by 1/d^3 for direction vector
        
        repulsive_forces_matrix = diffs * force_magnitudes_per_dist_unit[:, :, np.newaxis]
        total_repulsive_forces = np.sum(repulsive_forces_matrix, axis=1)
        points += current_time_step * total_repulsive_forces
        
        # Dynamic centering and scaling
        points -= np.mean(points, axis=0)
        max_extent = np.max(np.abs(points))
        if max_extent > 1e-9:
            points /= (max_extent * 2) # Scale to prevent explosion
        points += 0.5 # Translate back to roughly [0,1] for next iteration's candidates if needed, for SA it's just a starting point
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D to maximize min/max distance ratio using an advanced multi-start hybrid approach.
    It combines:
    1. Two distinct high-quality geometric initializations (Cube-centric, Fibonacci sphere).
    2. A force-directed pre-optimization step for each initial configuration.
    3. Numba-accelerated Simulated Annealing for global search in unconstrained space.
    4. Unconstrained L-BFGS-B local refinement.
    The best result from the two independent runs is selected.

    Returns
        points: np.ndarray of shape (14,3) scaled so the maximum distance is 1.0.
    """
    n_points = 14
    n_dim = 3
    n_vars = n_points * n_dim

    np.random.seed(42)

    # --- Generate Initial Config 1: Cube + Faces ---
    vertices = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    face_centers = np.array([[0.5,0.5,0],[0.5,0.5,1],[0.5,0,0.5],[0.5,1,0.5],[0,0.5,0.5],[1,0.5,0.5]])
    initial_points_1 = np.vstack((vertices, face_centers))
    initial_points_1 = _run_force_directed_pre_opt(initial_points_1, n_points, n_dim) # Apply pre-opt

    # --- Generate Initial Config 2: Fibonacci Sphere ---
    phi = np.pi * (3. - np.sqrt(5.))
    sphere_points = np.zeros((n_points, n_dim))
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x, z = np.cos(theta) * radius, np.sin(theta) * radius
        sphere_points[i] = [x, y, z]
    # No normalization to [0,1] cube yet, let force-directed pre-opt handle general spread
    initial_points_2 = _run_force_directed_pre_opt(sphere_points, n_points, n_dim) # Apply pre-opt

    # --- Generate Initial Config 3: Random Start ---
    # This adds diversity to catch basins of attraction missed by structured starts.
    initial_points_3 = np.random.rand(n_points, n_dim)
    initial_points_3 = _run_force_directed_pre_opt(initial_points_3, n_points, n_dim) # Apply pre-opt

    # --- Multi-Start Optimization ---
    all_initial_points = [initial_points_1, initial_points_2, initial_points_3]
    best_overall_points = None
    best_overall_score = -np.inf # Maximize d_min/d_max, so initialize with negative infinity

    # SA parameters, budget split between three runs, tuned for Numba and unconstrained search
    temperature = 0.5 # Increased initial temperature for broader exploration
    perturbation_scale = 0.05 # Increased perturbation scale for larger jumps
    # Adjusted iterations for three runs to fit within the 60s time budget (5M per run, 15M total)
    iterations = 5000000
    # Adjusted cooling rate for the new total iterations, maintaining the same cooling schedule length.
    # log(new_rate) = (old_iterations / new_iterations) * log(old_rate)
    # log(new_rate) = (4500000 / 5000000) * log(0.99998444) = 0.9 * (-1.556e-5) = -1.4004e-5
    # new_rate = exp(-1.4004e-5) approx 0.999985996
    cooling_rate = 0.999985996

    for run_idx, initial_points in enumerate(all_initial_points):
        sa_rng = np.random.default_rng(seed=43 + run_idx) # Different seed for each run
        
        current_points = initial_points.copy()
        current_energy = calculate_energy_jit(current_points) # Use JIT-compiled energy
        best_points_sa = current_points.copy()
        best_energy_sa = current_energy
        temp = temperature # Reset temperature for each run

        for i in range(iterations):
            new_points = current_points.copy()
            point_idx = sa_rng.integers(n_points)
            new_points[point_idx] += sa_rng.normal(0, perturbation_scale, n_dim)
            # NO CLIPPING: SA operates in unconstrained Euclidean space.

            new_energy = calculate_energy_jit(new_points)

            # Metropolis-Hastings acceptance criterion (for minimization)
            if new_energy < current_energy:
                current_points, current_energy = new_points, new_energy
                if new_energy < best_energy_sa:
                    best_points_sa, best_energy_sa = new_points.copy(), new_energy
            elif temp > 1e-7 and sa_rng.random() < np.exp(-(new_energy - current_energy) / temp):
                current_points, current_energy = new_points, new_energy
            
            temp *= cooling_rate
            if temp < 1e-7: break # Early exit if temperature is too low
        
        # --- Local Refinement (L-BFGS-B) with Smoothed Objective ---
        # Use the differentiable logsumexp-based objective for precise local search.
        # This technique is adopted from Inspiration Program 1.
        k_param = 2000.0 # Sharpness parameter for smooth min/max approximation
        res = minimize(
            _objective_dmin_dmax_ratio_smoothed, # Minimize -(d_min_smooth / d_max_smooth)
            best_points_sa.flatten(),
            args=(n_points, n_dim, k_param, k_param),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        # The objective function minimizes -(d_min/d_max), so the resulting ratio is -res.fun
        current_ratio = -res.fun if res.success and res.fun < 0 else 0.0
        
        if current_ratio > best_overall_score:
            best_overall_score = current_ratio
            refined_points = res.x.reshape((n_points, n_dim))
            
            # --- Final Scaling for the current best ---
            # Center the final points
            centered_points = refined_points - np.mean(refined_points, axis=0)
            final_distances = pdist(centered_points)
            
            if len(final_distances) > 0 and np.max(final_distances) > 1e-9:
                scaling_factor = 1.0 / np.max(final_distances)
                best_overall_points = centered_points * scaling_factor
            else:
                best_overall_points = refined_points # Fallback if scaling fails

    return best_overall_points


# EVOLVE-BLOCK-END
