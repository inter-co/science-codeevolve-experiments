# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit

N_CIRCLES = 21

@njit
def _simulation_step(positions, velocities, radii, W, H, K_REPULSION, K_WALL, DT, DAMPING):
    """Performs one step of physics simulation with a centered coordinate system."""
    forces = np.zeros_like(positions)
    n_circles = positions.shape[0]
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            direction = positions[i] - positions[j]
            dist_sq = direction[0]**2 + direction[1]**2
            radii_sum = radii[i] + radii[j]
            radii_sum_sq = radii_sum**2
            if dist_sq < radii_sum_sq and dist_sq > 1e-16:
                dist = np.sqrt(dist_sq)
                overlap = radii_sum - dist
                force_magnitude = K_REPULSION * overlap
                force_vec = (force_magnitude / dist) * direction
                forces[i] += force_vec
                forces[j] -= force_vec
    
    half_W, half_H = W / 2.0, H / 2.0
    for i in range(n_circles):
        r_i = radii[i]
        forces[i, 0] += K_WALL * max(0, (-half_W) - (positions[i, 0] - r_i)) # Left
        forces[i, 0] -= K_WALL * max(0, (positions[i, 0] + r_i) - half_W)   # Right
        forces[i, 1] += K_WALL * max(0, (-half_H) - (positions[i, 1] - r_i)) # Bottom
        forces[i, 1] -= K_WALL * max(0, (positions[i, 1] + r_i) - half_H)   # Top

    velocities += forces * DT
    velocities *= DAMPING
    positions += velocities * DT
    return positions, velocities

@njit
def _greedy_final_correction(positions, radii, W, H):
    """Iteratively shrinks radii to resolve overlaps in a centered coordinate system."""
    n_circles = positions.shape[0]
    for _ in range(1500): # Increased iterations for higher precision, matching inspiration.
        max_cc_overlap, max_cc_i, max_cc_j = 0.0, -1, -1
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                if dist > 1e-12:
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

        if max_cc_overlap < 1e-9 and max_wall_overlap < 1e-9: break
        
        if (max_cc_overlap / 2.0) > max_wall_overlap:
            if max_cc_i != -1:
                shrink_amount = (max_cc_overlap / 2.0) + 1e-11
                radii[max_cc_i] -= shrink_amount
                radii[max_cc_j] -= shrink_amount
        else:
            if max_wall_i != -1:
                radii[max_wall_i] -= (max_wall_overlap + 1e-11)
    return np.maximum(radii, 0)

@njit
def _slsqp_constraints(p):
    """Calculates all constraints for the integrated optimization problem (p includes W)."""
    n = N_CIRCLES
    positions = p[:2*n].reshape(n, 2)
    radii = p[2*n:3*n]
    W = p[3*n]
    H = 2.0 - W
    half_W, half_H = W / 2.0, H / 2.0

    num_constraints = n + 4 * n + n * (n - 1) // 2
    c = np.empty(num_constraints, dtype=np.float64)
    idx = 0

    for i in range(n):
        r_i, pos_i = radii[i], positions[i]
        c[idx] = r_i - 1e-6; idx += 1
        c[idx] = half_W - pos_i[0] - r_i; idx += 1 # x + r <= W/2
        c[idx] = pos_i[0] + half_W - r_i; idx += 1 # x - r >= -W/2
        c[idx] = half_H - pos_i[1] - r_i; idx += 1 # y + r <= H/2
        c[idx] = pos_i[1] + half_H - r_i; idx += 1 # y - r >= -H/2
    
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2
            c[idx] = dist_sq - (radii[i] + radii[j])**2; idx += 1
    return c

def _run_single_trial(initial_W, seed, sim_steps, total_growth, slsqp_maxiter):
    """Runs a single, fully integrated optimization trial."""
    np.random.seed(seed)
    n = N_CIRCLES
    
    W, H = initial_W, 2.0 - initial_W
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
    def objective(p): return -np.sum(p[2*n:3*n])
    cons = [{'type': 'ineq', 'fun': _slsqp_constraints}]
    
    bounds = [(-0.9, 0.9)] * (2*n) + [(1e-6, 1.0)] * n + [(0.2, 1.8)]
    
    res = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=cons,
                     options={'maxiter': slsqp_maxiter, 'ftol': 1e-9, 'disp': False})

    p_final = res.x
    pos_centered = p_final[:2*n].reshape(n, 2)
    radii_final = p_final[2*n:3*n]
    W_final = p_final[3*n]
    H_final = 2.0 - W_final

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
    
    # Multi-start trials with different initial W values and seeds,
    # adopted from Inspiration 1 for broader exploration.
    trials = [
        (1.0, 42),  # Square-like initial W
        (1.3, 0),   # Wider rectangle
        (0.7, 7),   # Taller rectangle
        (1.15, 13), # Another aspect ratio
        (1.05, 1),  # Slightly off square
        (0.95, 20), # Slightly off square
        (1.4, 33),  # Even wider
        (0.6, 55)   # Even taller
    ]
    
    # Parameters for the hybrid optimization, tuned based on Inspiration 1's successful run.
    # Reduced per-trial budget to accommodate more trials within the 60s limit,
    # while maintaining a good balance for exploration vs. refinement.
    sim_steps_per_trial = 150000
    slsqp_maxiter_per_trial = 15000
    total_growth_factor = 0.130

    for initial_W, seed in trials:
        circles, sum_radii = _run_single_trial(
            initial_W=initial_W,
            seed=seed,
            sim_steps=sim_steps_per_trial, 
            total_growth=total_growth_factor,
            slsqp_maxiter=slsqp_maxiter_per_trial
        )
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_circles = circles

    # Return empty array if the best sum of radii is suspiciously small (e.g., failed to pack)
    # This check is adopted from Inspiration 1 for robustness.
    if best_sum_radii < 0.1:
        return np.zeros((N_CIRCLES, 3))
        
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
