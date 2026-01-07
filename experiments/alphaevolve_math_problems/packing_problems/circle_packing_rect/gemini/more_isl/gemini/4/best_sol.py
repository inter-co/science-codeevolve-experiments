# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit
from scipy.spatial.distance import pdist, squareform # Added for vectorized distance calculations

N_CIRCLES = 21
PERIMETER = 4.0 # Global constant for container perimeter
EPSILON = 1e-7 # Tighter precision for numerical stability and constraints
MIN_DIST_SQ_TOL = EPSILON**2 # For squared distance checks to avoid division by zero

@njit
def _simulation_step(positions, velocities, radii, W, H, K_REPULSION, K_WALL, DT, DAMPING):
    """
    Performs one step of physics simulation with a centered coordinate system.
    Optimized for circle-circle force calculation (only sqrt if overlap)
    based on Inspiration 1, and added explicit handling for exact overlaps.
    Uses MIN_DIST_SQ_TOL for robustness.
    """
    forces = np.zeros_like(positions)
    n_circles = positions.shape[0]
    
    # Circle-circle repulsion forces
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            d_vec_x = positions[i, 0] - positions[j, 0]
            d_vec_y = positions[i, 1] - positions[j, 1]
            d_ij_sq = d_vec_x**2 + d_vec_y**2
            
            radii_sum = radii[i] + radii[j]
            radii_sum_sq = radii_sum**2

            if d_ij_sq < radii_sum_sq: # Check for overlap using squared distances
                # Only calculate sqrt if actual overlap
                dist = np.sqrt(d_ij_sq)
                if dist > MIN_DIST_SQ_TOL: # Avoid division by zero, use MIN_DIST_SQ_TOL
                    overlap = radii_sum - dist
                    force_magnitude = K_REPULSION * overlap
                    
                    force_vec_x = (force_magnitude / dist) * d_vec_x
                    force_vec_y = (force_magnitude / dist) * d_vec_y
                    
                    forces[i, 0] += force_vec_x
                    forces[i, 1] += force_vec_y
                    forces[j, 0] -= force_vec_x
                    forces[j, 1] -= force_vec_y
                else: # Handle exact overlap (circles at same point)
                    # Apply a small random force to separate them (Numba compatible random)
                    angle = np.random.uniform(0, 2 * np.pi) 
                    force_x, force_y = K_REPULSION * radii_sum * np.cos(angle), K_REPULSION * radii_sum * np.sin(angle)
                    forces[i, 0] += force_x
                    forces[i, 1] += force_y
                    forces[j, 0] -= force_x
                    forces[j, 1] -= force_y

    # Wall repulsion forces
    half_W, half_H = W / 2.0, H / 2.0
    for i in range(n_circles):
        r_i = radii[i]
        # Left boundary
        forces[i, 0] += K_WALL * max(0, (-half_W) - (positions[i, 0] - r_i))
        # Right boundary
        forces[i, 0] -= K_WALL * max(0, (positions[i, 0] + r_i) - half_W)
        # Bottom boundary
        forces[i, 1] += K_WALL * max(0, (-half_H) - (positions[i, 1] - r_i))
        # Top boundary
        forces[i, 1] -= K_WALL * max(0, (positions[i, 1] + r_i) - half_H)

    velocities += forces * DT
    velocities *= DAMPING
    positions += velocities * DT
    return positions, velocities

@njit
def _greedy_final_correction(positions, radii, W, H):
    """
    Iteratively shrinks radii to resolve overlaps in a centered coordinate system.
    Uses EPSILON for robustness.
    """
    n_circles = positions.shape[0]
    for _ in range(1500): # Increased iterations for even higher precision in final correction
        max_cc_overlap, max_cc_i, max_cc_j = 0.0, -1, -1
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                if dist > EPSILON: # Use EPSILON for robustness
                    overlap = (radii[i] + radii[j]) - dist
                    if overlap > max_cc_overlap:
                        max_cc_overlap, max_cc_i, max_cc_j = overlap, i, j
        
        max_wall_overlap, max_wall_i = 0.0, -1
        half_W, half_H = W / 2.0, H / 2.0
        for i in range(n_circles):
            r_i = radii[i]
            pos_i = positions[i]
            current_max = max(pos_i[0] + r_i - half_W, -half_W - (pos_i[0] - r_i),
                              pos_i[1] + r_i - half_H, -half_H - (pos_i[1] - r_i))
            if current_max > max_wall_overlap:
                max_wall_overlap, max_wall_i = current_max, i

        if max_cc_overlap < EPSILON and max_wall_overlap < EPSILON: break # Use EPSILON for tolerance
        
        if (max_cc_overlap / 2.0) > max_wall_overlap:
            if max_cc_i != -1:
                shrink_amount = (max_cc_overlap / 2.0) + EPSILON # Use EPSILON for robust shrinking
                radii[max_cc_i] -= shrink_amount
                radii[max_cc_j] -= shrink_amount
        else:
            if max_wall_i != -1:
                radii[max_wall_i] -= (max_wall_overlap + EPSILON) # Use EPSILON for robust shrinking
    return np.maximum(radii, 0)

# --- Vectorized objective and constraint functions for SLSQP (from Inspiration 1) ---
def _objective_slsqp(params):
    # Minimize the negative sum of radii to maximize the sum of radii.
    return -np.sum(params[2*N_CIRCLES : 3*N_CIRCLES])

def _constraints_radii_slsqp(params):
    # All radii must be positive (r_i >= EPSILON)
    return params[2*N_CIRCLES : 3*N_CIRCLES] - EPSILON

def _constraints_containment_slsqp(params):
    # Circles must be fully contained within the rectangle [ -W/2, W/2 ] x [ -H/2, H/2 ]
    # where H = PERIMETER/2 - W.
    pos = params[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii, width = params[2*N_CIRCLES : 3*N_CIRCLES], params[3*N_CIRCLES]
    height = PERIMETER/2 - width # Corrected to use PERIMETER/2

    # x-coordinates: x_i - r_i >= -W/2 and x_i + r_i <= W/2
    c1 = width/2 - pos[:, 0] - radii  # W/2 - x_i - r_i >= 0
    c2 = pos[:, 0] + width/2 - radii  # x_i + W/2 - r_i >= 0
    
    # y-coordinates: y_i - r_i >= -H/2 and y_i + r_i <= H/2
    c3 = height/2 - pos[:, 1] - radii # H/2 - y_i - r_i >= 0
    c4 = pos[:, 1] + height/2 - radii # y_i + H/2 - r_i >= 0
    
    return np.concatenate((c1, c2, c3, c4))

def _constraints_overlap_slsqp(params):
    # No circles overlap: dist(c_i, c_j)^2 >= (r_i + r_j)^2
    pos = params[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = params[2*N_CIRCLES : 3*N_CIRCLES]
    
    # Compute squared Euclidean distances between all circle centers
    dist_sq = squareform(pdist(pos, 'sqeuclidean'))
    # Compute squared sum of radii for all pairs
    radii_sum_sq = (radii[:, None] + radii[None, :])**2
    
    # The constraint is dist_sq - radii_sum_sq >= 0.
    # We only need the upper triangle (excluding diagonal) elements for unique pairs.
    return (dist_sq - radii_sum_sq)[np.triu_indices(N_CIRCLES, k=1)]


def _run_single_trial(initial_W, seed, sim_steps, total_growth, slsqp_maxiter):
    """Runs a single, fully integrated optimization trial."""
    np.random.seed(seed)
    n = N_CIRCLES
    
    W, H = initial_W, PERIMETER/2.0 - initial_W # Use PERIMETER constant
    K_REPULSION, K_WALL, DT, DAMPING = 1.0, 2.5, 0.01, 0.95
    growth_rate = total_growth / sim_steps if sim_steps > 0 else 0
    positions = (np.random.rand(n, 2) - 0.5) * np.array([W, H])
    velocities = np.zeros((n, 2))
    radii = np.full(n, 0.001)

    for _ in range(sim_steps):
        radii += growth_rate
        positions, velocities = _simulation_step(positions, velocities, radii, W, H, K_REPULSION, K_WALL, DT, DAMPING)
    
    for _ in range(sim_steps // 4):
        positions, velocities = _simulation_step(positions, velocities, radii, W, H, K_REPULSION, K_WALL, DT, DAMPING)
        if np.sum(velocities**2) < 1e-15: break

    initial_params = np.concatenate((positions.flatten(), radii, [W]))
    objective = _objective_slsqp # Use the new objective function
    
    # Use the new list of constraint dictionaries (from Inspiration 1)
    cons = [{'type': 'ineq', 'fun': _constraints_radii_slsqp},
            {'type': 'ineq', 'fun': _constraints_containment_slsqp},
            {'type': 'ineq', 'fun': _constraints_overlap_slsqp}]
    
    # Bounds for radii should use EPSILON, and W bounds should use PERIMETER/2 (from Inspiration 1)
    # Adjusted position bounds to be more robust for extreme aspect ratios
    bounds = [(-1.0 + EPSILON, 1.0 - EPSILON)] * (2*n) + [(EPSILON, 1.0)] * n + [(EPSILON, PERIMETER/2 - EPSILON)] 
    
    res = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=cons,
                     options={'maxiter': slsqp_maxiter, 'ftol': 1e-8, 'disp': False}) # Relaxed ftol from Inspiration 1

    p_final = res.x
    pos_centered = p_final[:2*n].reshape(n, 2)
    radii_final = p_final[2*n:3*n]
    W_final = p_final[3*n]
    H_final = PERIMETER/2 - W_final # Use PERIMETER constant

    radii_corrected = _greedy_final_correction(pos_centered, radii_final, W_final, H_final)
    pos_final = pos_centered + np.array([W_final / 2.0, H_final / 2.0])
    circles = np.hstack((pos_final, radii_corrected.reshape(n, 1)))
    return circles, np.sum(radii_corrected)

def circle_packing21() -> np.ndarray:
    """
    Finds an optimal packing using a multi-start, integrated optimization strategy.
    The container width 'W' is optimized simultaneously with circle parameters.
    """
    best_circles = np.zeros((N_CIRCLES, 3))
    best_sum_radii = -1.0

    # Generate a more diverse set of initial rectangle widths and seeds for multi-start.
    # This strategy increases exploration of the solution landscape, aiming to find
    # a higher global optimum within the given time budget.
    num_initial_W_points = 11 # Number of distinct W values to sample directly
    W_samples = np.linspace(0.5, 1.5, num_initial_W_points) # Sample W values in a common range
    
    trial_configs = []
    base_seed = 42 # Starting seed for reproducibility and diversity
    for i, initial_W in enumerate(W_samples):
        # Add the sampled W
        trial_configs.append({'initial_W': initial_W, 'seed': base_seed + i})
        # Add its complement (2.0 - W) to explore both aspect ratios
        # Ensure complement is not too close to the original W or outside bounds
        complement_W = PERIMETER/2.0 - initial_W # Use PERIMETER constant
        if EPSILON < complement_W < PERIMETER/2.0 - EPSILON and abs(complement_W - initial_W) > 0.05:
            trial_configs.append({'initial_W': complement_W, 'seed': base_seed + i + num_initial_W_points})
    
    # Tuned parameters for each trial, balancing depth of search with number of trials
    # to maximize overall performance within the 60-second limit.
    sim_steps_per_trial = 150000 # Adjusted from 180k to allow for more trials or faster execution
    slsqp_maxiter_per_trial = 10000 # Adjusted from 12k to allow for more trials or faster execution
    total_growth_factor = 0.130 # Consistent with inspirations

    for config in trial_configs:
        circles, sum_radii = _run_single_trial(
            initial_W=config['initial_W'],
            seed=config['seed'],
            sim_steps=sim_steps_per_trial,
            total_growth=total_growth_factor,
            slsqp_maxiter=slsqp_maxiter_per_trial
        )
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_circles = circles
    
    # Handle cases where optimization might fail and return a default empty array
    if best_sum_radii < 0.1: # A very small sum of radii indicates a failure
        return np.zeros((N_CIRCLES, 3))

    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
