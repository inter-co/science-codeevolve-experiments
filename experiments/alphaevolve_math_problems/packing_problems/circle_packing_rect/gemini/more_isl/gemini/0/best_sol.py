# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from numba import njit

# Global constants adopted from the high-performing Inspiration Program 1.
NUM_CIRCLES = 21
PERIMETER = 4.0
EPSILON = 1e-6  # Minimum radius to maintain positivity.

@njit
def _simulation_step(positions, velocities, radii, W, H, K_REPULSION, K_WALL, DT, DAMPING):
    """
    Performs one step of a physics simulation using a centered coordinate system.
    This function, adapted from Inspiration Program 1, calculates repulsive forces
    between circles and from container walls to arrange the circles.
    """
    forces = np.zeros_like(positions)
    n_circles = positions.shape[0]
    # Circle-circle repulsion forces
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
    # Circle-boundary repulsion forces (for centered coordinate system)
    for i in range(n_circles):
        r_i = radii[i]
        forces[i, 0] += K_WALL * max(0, (-half_W) - (positions[i, 0] - r_i))
        forces[i, 0] -= K_WALL * max(0, (positions[i, 0] + r_i) - half_W)
        forces[i, 1] += K_WALL * max(0, (-half_H) - (positions[i, 1] - r_i))
        forces[i, 1] -= K_WALL * max(0, (positions[i, 1] + r_i) - half_H)

    # Update velocities and positions based on forces ( Verlet integration)
    velocities += forces * DT
    velocities *= DAMPING
    positions += velocities * DT
    return positions, velocities

@njit
def _greedy_correction_jit(positions, radii, W, H, min_radius_epsilon):
    """
    Post-processes the layout by iteratively finding the largest single overlap
    and shrinking only the involved circle(s) to resolve it. This is less
    destructive to the sum of radii than uniform scaling. Inspired by Insp. 1.
    """
    n = len(positions)
    half_W, half_H = W / 2.0, H / 2.0
    
    for _ in range(1000): # A sufficient number of iterations for convergence
        max_cc_overlap, max_cc_i, max_cc_j = 0.0, -1, -1
        # Find largest circle-circle overlap
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                overlap = (radii[i] + radii[j]) - dist
                if overlap > max_cc_overlap:
                    max_cc_overlap, max_cc_i, max_cc_j = overlap, i, j

        max_wall_overlap, max_wall_i = 0.0, -1
        # Find largest wall overlap (in centered coordinates)
        for i in range(n):
            r_i = radii[i]
            current_max = max(
                positions[i, 0] + r_i - half_W, -half_W - (positions[i, 0] - r_i),
                positions[i, 1] + r_i - half_H, -half_H - (positions[i, 1] - r_i)
            )
            if current_max > max_wall_overlap:
                max_wall_overlap, max_wall_i = current_max, i
        
        if max_cc_overlap < 1e-9 and max_wall_overlap < 1e-9: break
        
        # Correct the larger of the two types of overlaps
        if (max_cc_overlap / 2.0) > max_wall_overlap:
            if max_cc_i != -1:
                shrink_amount = (max_cc_overlap / 2.0) + 1e-11
                radii[max_cc_i] -= shrink_amount
                radii[max_cc_j] -= shrink_amount
        else:
            if max_wall_i != -1:
                radii[max_wall_i] -= (max_wall_overlap + 1e-11)
    
    return np.maximum(radii, min_radius_epsilon)

@njit
def _slsqp_constraints_jit(p):
    """
    Calculates all SLSQP inequality constraints in a single JIT-compiled function.
    This is a key performance optimization from Insp. 1. The optimizer expects c(x) >= 0.
    The parameter vector `p` includes positions, radii, AND the container width `W`.
    """
    n = NUM_CIRCLES
    positions = p[:2*n].reshape((n, 2))
    radii = p[2*n:3*n]
    W = p[3*n]
    H = PERIMETER / 2.0 - W
    half_W, half_H = W / 2.0, H / 2.0

    num_constraints = n + 4 * n + n * (n - 1) // 2
    c = np.empty(num_constraints, dtype=np.float64)
    idx = 0

    # 1. Positive radii: r_i - EPSILON >= 0
    for i in range(n):
        c[idx] = radii[i] - EPSILON; idx += 1

    # 2. Containment (centered coords): e.g., W/2 - x - r >= 0
    for i in range(n):
        r_i = radii[i]
        c[idx] = half_W - positions[i, 0] - r_i; idx += 1
        c[idx] = positions[i, 0] + half_W - r_i; idx += 1
        c[idx] = half_H - positions[i, 1] - r_i; idx += 1
        c[idx] = positions[i, 1] + half_H - r_i; idx += 1

    # 3. Non-overlap: dist_sq - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2
            c[idx] = dist_sq - (radii[i] + radii[j])**2; idx += 1
    
    return c

def _run_single_trial(initial_W, seed, sim_steps, total_growth, slsqp_maxiter):
    """
    Runs a single, fully integrated optimization trial, from simulation to final correction.
    This function structure is adopted from the successful strategy of Insp. 1.
    """
    np.random.seed(seed)
    n = NUM_CIRCLES
    
    # --- Stage 1: Physics Simulation for Initial Guess ---
    W_sim, H_sim = initial_W, PERIMETER/2.0 - initial_W
    K_REPULSION, K_WALL, DT, DAMPING = 1.0, 2.5, 0.01, 0.95 # Effective params from Insp. 1
    growth_rate = total_growth / sim_steps if sim_steps > 0 else 0
    
    positions = (np.random.rand(n, 2) - 0.5) * np.array([W_sim, H_sim])
    velocities = np.zeros((n, 2))
    radii = np.full(n, 0.001)

    for _ in range(sim_steps):
        radii += growth_rate
        positions, velocities = _simulation_step(positions, velocities, radii, W_sim, H_sim, K_REPULSION, K_WALL, DT, DAMPING)
    
    for _ in range(sim_steps // 4): # Settling phase
        positions, velocities = _simulation_step(positions, velocities, radii, W_sim, H_sim, K_REPULSION, K_WALL, DT, DAMPING)
        if np.sum(velocities**2) < 1e-15: break

    # --- Stage 2: SLSQP Optimization (Integrated W) ---
    initial_params = np.concatenate((positions.flatten(), radii, [W_sim]))
    
    def objective(p): return -np.sum(p[2*n:3*n]) # Maximize sum of radii
    
    cons = [{'type': 'ineq', 'fun': _slsqp_constraints_jit}]
    bounds = [(-1.5, 1.5)] * (2*n) + [(EPSILON, 1.0)] * n + [(0.2, 1.8)]
    
    res = minimize(objective, initial_params, method='SLSQP', bounds=bounds, constraints=cons,
                     options={'maxiter': slsqp_maxiter, 'ftol': 1e-9, 'disp': False})

    # --- Stage 3: Post-processing and Coordinate Conversion ---
    p_final = res.x
    pos_centered = p_final[:2*n].reshape(n, 2)
    radii_final = p_final[2*n:3*n]
    W_final = p_final[3*n]
    H_final = PERIMETER/2.0 - W_final

    radii_corrected = _greedy_correction_jit(pos_centered, radii_final, W_final, H_final, EPSILON)
    
    # Convert centered positions to the required output format [0, W]
    pos_final = pos_centered + np.array([W_final / 2.0, H_final / 2.0])
    circles = np.hstack((pos_final, radii_corrected.reshape(n, 1)))
    
    return circles, np.sum(radii_corrected)

def circle_packing21() -> np.ndarray:
    """
    Finds an optimal packing using a multi-start, integrated optimization strategy.
    This top-level function orchestrates multiple trials to find the best possible result,
    a strategy proven effective by Inspiration Program 1.
    """
    best_circles = np.zeros((NUM_CIRCLES, 3))
    best_sum_radii = -1.0
    
    # A set of diverse starting points for width and randomness, taken from Insp. 1.
    trials = [
        (1.0, 42), (1.3, 0), (0.7, 7), (1.15, 13),
        (1.05, 1), (0.95, 20), (1.4, 33), (0.6, 55)
    ]
    
    # Hyperparameters tuned for performance based on the benchmark-beating program.
    sim_steps_per_trial = 180000 
    slsqp_maxiter_per_trial = 18000
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

    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
