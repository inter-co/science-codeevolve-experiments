# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import njit

# --- Numba-jitted core functions for performance (inspired by Inspiration Programs 1 & 2) ---

@njit(cache=True)
def _get_min_max_dist_sq(points: np.ndarray):
    """
    Numba-accelerated function to find the squared minimum and maximum pairwise 
    distances and the indices of the points that form them. Working with squared 
    distances avoids costly sqrt operations inside the main simulation loop.
    """
    n, d = points.shape
    min_dist_sq = np.inf
    max_dist_sq = 0.0
    min_indices = (0, 1)
    max_indices = (0, 1)

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(d):
                dist_sq += (points[i, k] - points[j, k])**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                min_indices = (i, j)
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                max_indices = (i, j)
    
    return min_dist_sq, max_dist_sq, min_indices, max_indices

@njit(cache=True)
def _calculate_targeted_forces(points, min_indices, max_indices, min_dist_sq, repulsion_strength, attraction_strength):
    """
    Numba-accelerated function to calculate forces based on a targeted strategy:
    1. A repulsive force on the closest pair.
    2. An attractive (spring) force on the farthest pair.
    This implementation is sqrt-free for maximum efficiency.
    """
    forces = np.zeros_like(points)
    
    # 1. Repulsive force on the closest pair
    i_min, j_min = min_indices
    vec_min = points[i_min] - points[j_min]
    if min_dist_sq > 1e-24: # Safety check for collapsed points
        # Force magnitude is scaled to avoid using sqrt: F ~ vec / dist^4 = vec / dist_sq^2
        force_scalar = repulsion_strength / (min_dist_sq**2)
        force_rep = vec_min * force_scalar
        forces[i_min] += force_rep
        forces[j_min] -= force_rep

    # 2. Attractive force on the farthest pair (spring-like F = -k*x)
    i_max, j_max = max_indices
    vec_max = points[i_max] - points[j_max]
    force_attr_vec = attraction_strength * vec_max
    forces[i_max] -= force_attr_vec
    forces[j_max] += force_attr_vec
    
    return forces

def generate_fibonacci_sphere_points(n_points: int, radius: float = 0.45, center: tuple = (0.5, 0.5, 0.5), seed: int = None) -> np.ndarray:
    """
    Generates points on a sphere using the Fibonacci spiral method, then scales and translates them.
    A small random perturbation is added, and points are clipped to [0,1]^3.
    """
    if seed is not None:
        np.random.seed(seed)

    points = []
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
        r_at_y = np.sqrt(1 - y * y)  # Radius at y
        theta = phi * i
        x = np.cos(theta) * r_at_y
        z = np.sin(theta) * r_at_y
        points.append([x, y, z])

    points_np = np.array(points) * radius + np.array(center)
    
    perturbation_strength = 0.005 * radius
    points_np += np.random.uniform(-perturbation_strength, perturbation_strength, points_np.shape)
    points_np = np.clip(points_np, 0.0, 1.0)
    return points_np

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in a unit cube [0,1]³ using a numba-accelerated, targeted 
    force-directed simulation, synthesizing the best techniques from inspiration programs.

    The simulation involves:
    - Initialization: Points start on a Fibonacci sphere for a high-quality initial guess.
    - Targeted Forces: Repulsion on the closest pair, attraction on the farthest.
    - Performance: Core logic is JIT-compiled with numba and uses sqrt-free math.
    - Dynamics: Momentum, cosine annealing learning rate, and a random "kick" for robust exploration.
    """
    n, d, seed = 14, 3, 42
    np.random.seed(seed)

    # --- Hyperparameters (tuned from best-performing inspirations) ---
    n_iterations = 6_000_000 # Adjusted from 8M due to better starting point
    time_limit = 350 # Safety time limit
    
    initial_learning_rate, final_learning_rate = 0.01, 1e-6
    repulsion_strength, attraction_strength = 0.01, 0.05
    damping_factor = 0.95
    initial_perturbation, perturbation_decay = 0.01, 0.999997

    # --- Initialization ---
    points = generate_fibonacci_sphere_points(n, radius=0.45, center=(0.5, 0.5, 0.5), seed=seed)
    best_points = points.copy()
    
    min_d_sq, max_d_sq, _, _ = _get_min_max_dist_sq(points)
    best_ratio_sq = min_d_sq / max_d_sq if max_d_sq > 1e-24 else 0.0
    
    velocity = np.zeros_like(points)
    perturbation_magnitude = initial_perturbation
    start_time = time.time()

    # --- Main Force-Directed Simulation Loop ---
    for iteration in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        min_dist_sq, max_dist_sq, min_indices, max_indices = _get_min_max_dist_sq(points)
        if max_dist_sq < 1e-24: continue
        
        current_ratio_sq = min_dist_sq / max_dist_sq
        if current_ratio_sq > best_ratio_sq:
            best_ratio_sq = current_ratio_sq
            best_points = points.copy()

        # Cosine annealing for the learning rate
        t_norm = iteration / n_iterations
        learning_rate = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (1 + np.cos(np.pi * t_norm))

        # Calculate forces and update state
        forces = _calculate_targeted_forces(points, min_indices, max_indices, min_dist_sq, repulsion_strength, attraction_strength)
        velocity = damping_factor * velocity + learning_rate * forces
        random_kick = (np.random.rand(n, d) - 0.5) * 2 * perturbation_magnitude
        points += velocity + random_kick
        
        perturbation_magnitude *= perturbation_decay
        if perturbation_magnitude < 1e-8:
            perturbation_magnitude = 1e-8

        points = np.clip(points, 0, 1)

    return best_points


# EVOLVE-BLOCK-END
