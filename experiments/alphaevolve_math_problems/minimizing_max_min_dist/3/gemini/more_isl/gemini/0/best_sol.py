# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from numba import njit

# Numba-accelerated pairwise distance calculation (from inspiration)
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

# Numba-accelerated SA energy function (minimizes d_max / d_min, from inspiration)
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

    return d_max / d_min

# Objective function for scipy.minimize, minimizing d_max/d_min for unconstrained optimization
def _objective_function_scipy(points_flat: np.ndarray, n_points: int, n_dim: int) -> float:
    points = points_flat.reshape((n_points, n_dim))
    
    # Center the points for scale-invariant evaluation
    centered_points = points - np.mean(points, axis=0)
    
    distances = pdist(centered_points, 'euclidean')
    
    if distances.size == 0: return np.inf
    
    dmin, dmax = np.min(distances), np.max(distances)
    
    if dmax < 1e-9: return np.inf
    if dmin < 1e-9: return np.inf
    
    return dmax / dmin

# Force-directed pre-optimizer (from inspiration)
def _run_force_directed_pre_opt(initial_points: np.ndarray, num_points: int, dimensions: int) -> np.ndarray:
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

        force_magnitudes_per_dist_unit = 0.1 / (dists_clipped**5)
        
        repulsive_forces_matrix = diffs * force_magnitudes_per_dist_unit[:, :, np.newaxis]
        total_repulsive_forces = np.sum(repulsive_forces_matrix, axis=1)
        points += current_time_step * total_repulsive_forces
        
        # Dynamic centering and scaling
        points -= np.mean(points, axis=0)
        max_extent = np.max(np.abs(points))
        if max_extent > 1e-9:
            points /= (max_extent * 2)
        points += 0.5
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3D to maximize min/max distance ratio using an advanced multi-start hybrid approach
    inspired by high-performance solutions. It combines:
    1. Two distinct geometric initializations (Cube-centric, Fibonacci sphere).
    2. A force-directed pre-optimization step.
    3. Numba-accelerated Simulated Annealing for global search in unconstrained space.
    4. Unconstrained L-BFGS-B local refinement.
    """
    n_points = 14
    n_dim = 3
    n_vars = n_points * n_dim

    np.random.seed(42)

    # --- Generate Initial Config 1: Cube + Faces ---
    vertices = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    face_centers = np.array([[0.5,0.5,0],[0.5,0.5,1],[0.5,0,0.5],[0.5,1,0.5],[0,0.5,0.5],[1,0.5,0.5]])
    initial_points_1 = np.vstack((vertices, face_centers))
    initial_points_1 = _run_force_directed_pre_opt(initial_points_1, n_points, n_dim)

    # --- Generate Initial Config 2: Fibonacci Sphere ---
    phi = np.pi * (3. - np.sqrt(5.))
    sphere_points = np.zeros((n_points, n_dim))
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x, z = np.cos(theta) * radius, np.sin(theta) * radius
        sphere_points[i] = [x, y, z]
    initial_points_2 = _run_force_directed_pre_opt(sphere_points, n_points, n_dim)

    # --- Multi-Start Optimization ---
    all_initial_points = [initial_points_1, initial_points_2]
    best_overall_points = None
    best_overall_score = -np.inf

    # SA parameters from inspiration, tuned for Numba and unconstrained search
    temperature = 0.1
    perturbation_scale = 0.03
    iterations = 6500000
    cooling_rate = 0.999989225

    for run_idx, initial_points in enumerate(all_initial_points):
        sa_rng = np.random.default_rng(seed=43 + run_idx)
        
        current_points = initial_points.copy()
        current_energy = calculate_energy_jit(current_points)
        best_points_sa = current_points.copy()
        best_energy_sa = current_energy
        temp = temperature

        for i in range(iterations):
            new_points = current_points.copy()
            point_idx = sa_rng.integers(n_points)
            new_points[point_idx] += sa_rng.normal(0, perturbation_scale, n_dim)
            # NO CLIPPING: SA operates in unconstrained Euclidean space.
            new_energy = calculate_energy_jit(new_points)

            if new_energy < current_energy:
                current_points, current_energy = new_points, new_energy
                if new_energy < best_energy_sa:
                    best_points_sa, best_energy_sa = new_points.copy(), new_energy
            elif temp > 1e-7 and sa_rng.random() < np.exp(-(new_energy - current_energy) / temp):
                current_points, current_energy = new_points, new_energy
            
            temp *= cooling_rate
            if temp < 1e-7: break
        
        # --- Local Refinement (L-BFGS-B) in unconstrained space ---
        res = minimize(
            _objective_function_scipy,
            best_points_sa.flatten(),
            args=(n_points, n_dim),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        # Objective minimized d_max/d_min, so the ratio is 1/res.fun
        current_ratio = 1.0 / res.fun if res.success and res.fun > 1e-9 else 0.0
        
        if current_ratio > best_overall_score:
            best_overall_score = current_ratio
            refined_points = res.x.reshape((n_points, n_dim))
            
            # --- Final Scaling for the current best (center then scale) ---
            centered_points = refined_points - np.mean(refined_points, axis=0)
            final_distances = pdist(centered_points)
            
            if len(final_distances) > 0 and np.max(final_distances) > 1e-9:
                scaling_factor = 1.0 / np.max(final_distances)
                best_overall_points = centered_points * scaling_factor
            else:
                best_overall_points = refined_points

    return best_overall_points


# EVOLVE-BLOCK-END
