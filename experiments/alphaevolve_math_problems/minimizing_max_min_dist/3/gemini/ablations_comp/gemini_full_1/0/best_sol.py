# EVOLVE-BLOCK-START
import numpy as np
from numba import jit

# --- High-performance helper functions from Inspirations 1 & 3 ---

@jit(nopython=True, fastmath=True)
def fast_pdist_stats(points: np.ndarray) -> (float, float):
    """
    A Numba-jitted function to calculate min and max pairwise distances, avoiding sqrt in the inner loop.
    This is drastically faster than scipy.spatial.distance.pdist.
    """
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0, 1.0

    d_min_sq = np.inf
    d_max_sq = -1.0
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d_sq = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - points[j, k]
                d_sq += diff * diff
            
            if d_sq < d_min_sq:
                d_min_sq = d_sq
            if d_sq > d_max_sq:
                d_max_sq = d_sq
                
    return np.sqrt(d_min_sq), np.sqrt(d_max_sq)

def fcc_lattice(n_points: int) -> np.ndarray:
    """
    Generates an initial guess based on the Face-Centered Cubic (FCC) lattice for 14 points.
    This configuration is a theoretically strong starting point with a min/max ratio of ~0.408.
    """
    if n_points != 14:
        raise ValueError("FCC lattice initialization is specifically for 14 points.")
    corners = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]
    ])
    face_centers = np.array([
        [0.5, 0.5, 0.], [0.5, 0.5, 1.],
        [0.5, 0., 0.5], [0.5, 1., 0.5],
        [0., 0.5, 0.5], [1., 0.5, 0.5]
    ])
    return np.vstack([corners, face_centers])

def _objective_function(points: np.ndarray) -> float:
    """ Objective function: computes min/max ratio using the fast Numba helper. """
    d_min, d_max = fast_pdist_stats(points)
    if d_max < 1e-9:
        return 0.0
    return d_min / d_max

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds an optimal arrangement of 14 points in a 3D unit cube by maximizing d_min/d_max.
    
    This implementation enhances the Simulated Annealing (SA) algorithm by:
    1.  Initializing with a Face-Centered Cubic (FCC) lattice, a near-optimal configuration.
    2.  Using a Numba-jitted function for ultra-fast distance calculations.
    3.  Employing a tuned annealing schedule to refine the excellent starting guess.
    """
    n = 14
    d = 3
    np.random.seed(42)

    # --- SA Parameters tuned for refining the high-quality FCC start ---
    MAX_ITERATIONS = 2_500_000
    INITIAL_TEMPERATURE = 0.002
    COOLING_RATE = 0.999995
    INITIAL_PERTURB_SCALE = 0.005
    MIN_PERTURB_SCALE = 1e-6

    # --- Initialize with the strong FCC lattice configuration ---
    current_points = fcc_lattice(n_points=n)
    current_ratio = _objective_function(current_points)
    
    best_points = current_points.copy()
    best_ratio = current_ratio

    temperature = INITIAL_TEMPERATURE

    for _ in range(MAX_ITERATIONS):
        # 1. Generate a neighbor: perturb one point with annealed step size
        point_idx = np.random.randint(n)
        candidate_points = current_points.copy()
        
        # Anneal perturbation scale for fine-tuning as temperature drops
        perturb_scale = MIN_PERTURB_SCALE + (INITIAL_PERTURB_SCALE) * (temperature / INITIAL_TEMPERATURE)
        perturbation = np.random.normal(0, perturb_scale, size=d)
        
        candidate_points[point_idx] += perturbation
        candidate_points[point_idx] = np.clip(candidate_points[point_idx], 0.0, 1.0)
        
        # 2. Evaluate the candidate
        candidate_ratio = _objective_function(candidate_points)

        # 3. Metropolis-Hastings criterion
        if candidate_ratio > current_ratio:
            current_points = candidate_points
            current_ratio = candidate_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()
        elif temperature > 1e-9:
            acceptance_prob = np.exp((candidate_ratio - current_ratio) / temperature)
            if np.random.rand() < acceptance_prob:
                current_points = candidate_points
                current_ratio = candidate_ratio
        
        # 4. Cool down
        temperature *= COOLING_RATE
        
        if temperature < 1e-9:
            break

    return best_points


# EVOLVE-BLOCK-END
