# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from numba import njit

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

# Objective function for scipy.minimize, minimizing d_max/d_min
def _objective_function_scipy(points_flat: np.ndarray, n_points: int, n_dim: int) -> float:
    """
    Objective function for scipy.optimize.minimize, minimizing d_max/d_min.
    Leverages the Numba-accelerated calculate_energy_jit for performance.
    """
    points = points_flat.reshape((n_points, n_dim))
    # calculate_energy_jit already handles centering and returns d_max / d_min
    return calculate_energy_jit(points)

def _run_force_directed_pre_opt(initial_points: np.ndarray, num_points: int, dimensions: int) -> np.ndarray:
    """
    Implements a fast physics-based pre-optimization using a strong repulsive force,
    with dynamic centering and scaling to a max distance of 1.0 at each step.
    Inspired by Inspiration Program 1's stable normalization approach.
    """
    points = initial_points.copy()
    time_step, num_iterations, damping_factor = 0.02, 5000, 0.999
    for iteration in range(num_iterations):
        current_time_step = time_step * (damping_factor ** iteration)
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        dists_clipped = np.clip(dists, 1e-7, None)
        force_magnitudes_per_dist_unit = 0.1 / (dists_clipped**5)
        repulsive_forces_matrix = diffs * force_magnitudes_per_dist_unit[:, :, np.newaxis]
        total_repulsive_forces = np.sum(repulsive_forces_matrix, axis=1)
        points += current_time_step * total_repulsive_forces
        
        # Center the points
        points -= np.mean(points, axis=0)

        # Scale to keep max distance at 1.0 (inspired by Insp 1)
        # This is more robust than scaling by max extent and avoids the arbitrary '+0.5' shift.
        current_dists = fast_pdist_jit(points)
        if current_dists.size > 0:
            current_dmax = np.max(current_dists)
            if current_dmax > 1e-9:
                points /= current_dmax
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D to maximize min/max distance ratio using a funneling hybrid approach.
    1. Generates multiple diverse initial configurations (Cube, Sphere, Greedy Farthest-Point).
    2. Runs a fast physics-based pre-optimization on all candidates.
    3. Selects the most promising candidate based on the energy function.
    4. Concentrates the full computational budget on a single, deep Simulated Annealing run.
    5. Refines the result with L-BFGS-B.
    """
    n_points, n_dim = 14, 3
    np.random.seed(42)

    # Local import for greedy initializer dependency
    from scipy.spatial.distance import cdist

    # Helper function for greedy farthest-point initialization (from Inspiration 1)
    def _generate_greedy_farthest_points(n: int, d: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        points = np.zeros((n, d))
        points[0] = rng.uniform(-1, 1, d)
        num_candidates = 1000
        for i in range(1, n):
            candidates = rng.uniform(-1, 1, (num_candidates, d))
            dist_matrix = cdist(candidates, points[:i])
            min_dists = np.min(dist_matrix, axis=1)
            best_candidate_idx = np.argmax(min_dists)
            points[i] = candidates[best_candidate_idx]
        return points

    # --- Generate Initial Configurations ---
    initial_guesses = []
    # 1. Cube + Faces
    vertices = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    face_centers = np.array([[0.5,0.5,0],[0.5,0.5,1],[0.5,0,0.5],[0.5,1,0.5],[0,0.5,0.5],[1,0.5,0.5]])
    initial_guesses.append(np.vstack((vertices, face_centers)))
    # 2. Fibonacci Sphere
    phi = np.pi * (3. - np.sqrt(5.))
    sphere_points = np.zeros((n_points, n_dim))
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x, z = np.cos(theta) * radius, np.sin(theta) * radius
        sphere_points[i] = [x, y, z]
    initial_guesses.append(sphere_points)
    # 3. Greedy Farthest-Point (inspired by Inspiration 1)
    greedy_points = _generate_greedy_farthest_points(n_points, n_dim, seed=43)
    initial_guesses.append(greedy_points)

    # --- Pre-optimization and Selection Funnel ---
    pre_optimized_points = [
        _run_force_directed_pre_opt(p, n_points, n_dim) for p in initial_guesses
    ]
    energies = np.array([calculate_energy_jit(p) for p in pre_optimized_points])
    best_start_idx = np.argmin(energies)
    initial_points = pre_optimized_points[best_start_idx]

    # --- Main Optimization on the Best Candidate ---
    # SA parameters: budget is now concentrated on one run.
    temperature, perturbation_scale = 0.1, 0.03
    iterations = 13000000 # Use the full budget (2 * 6.5M)
    # Cooling rate adjusted for doubled iterations: old_rate^(old_iter/new_iter)
    cooling_rate = 0.999989225 ** 0.5 # approx 0.999994612

    sa_rng = np.random.default_rng(seed=43) # Use a single seed for the single run
    current_points = initial_points.copy()
    current_energy = calculate_energy_jit(current_points)
    best_points_sa, best_energy_sa = current_points.copy(), current_energy
    temp = temperature

    for i in range(iterations):
        new_points = current_points.copy()
        point_idx = sa_rng.integers(n_points)
        new_points[point_idx] += sa_rng.normal(0, perturbation_scale, n_dim)
        new_energy = calculate_energy_jit(new_points)

        if new_energy < current_energy:
            current_points, current_energy = new_points, new_energy
            if new_energy < best_energy_sa:
                best_points_sa, best_energy_sa = new_points.copy(), new_energy
        elif temp > 1e-7 and sa_rng.random() < np.exp(-(new_energy - current_energy) / temp):
            current_points, current_energy = new_points, new_energy
        
        temp *= cooling_rate
        if temp < 1e-7: break
    
    # --- Local Refinement (L-BFGS-B) ---
    res = minimize(
        _objective_function_scipy, best_points_sa.flatten(), args=(n_points, n_dim),
        method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9}
    )
    
    # --- Finalization ---
    if res.success and res.fun > 1e-9:
        refined_points = res.x.reshape((n_points, n_dim))
    else: # Fallback if L-BFGS-B fails
        refined_points = best_points_sa

    centered_points = refined_points - np.mean(refined_points, axis=0)
    final_distances = pdist(centered_points)
    if final_distances.size > 0 and np.max(final_distances) > 1e-9:
        scaling_factor = 1.0 / np.max(final_distances)
        best_overall_points = centered_points * scaling_factor
    else:
        best_overall_points = centered_points # Fallback if scaling fails

    return best_overall_points


# EVOLVE-BLOCK-END
