# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import time
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, physics-based energy minimization,
    and constrained optimization with improved initialization and parameter tuning.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective_function(x):
        """Objective function to minimize (negative of ratio to maximize ratio)."""
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize
        return -ratio
    
    def sphere_constraint(x):
        """Constraint to keep points on unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return difference from unit radius (should be close to 0)
        return norms - 1.0
    
    def energy_minimization(points, max_iter=1000, learning_rate=0.005, alpha=3.0):
        """Physics-based energy minimization with vectorized force computation."""
        points = points.copy()
        
        # Precompute all pairwise differences for efficiency
        for iteration in range(max_iter):
            # Compute all pairwise differences at once
            diff_matrix = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq_matrix = np.sum(diff_matrix**2, axis=2)
            
            # Avoid division by zero and self-interactions
            np.fill_diagonal(dist_sq_matrix, 1e-15)
            
            # Compute forces using inverse power law (repulsive)
            force_magnitudes = 1.0 / (dist_sq_matrix ** (alpha/2))
            np.fill_diagonal(force_magnitudes, 0)  # No self-force
            
            # Compute force vectors
            force_vectors = force_magnitudes[:, :, np.newaxis] * diff_matrix
            
            # Sum forces for each point
            forces = np.sum(force_vectors, axis=1)
            
            # Update positions
            points += learning_rate * forces
            
            # Project back to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
            
            # Early stopping if forces become very small
            if iteration > 50 and np.max(np.abs(forces)) < 1e-8:
                break
                
        return points
    
    # Generate multiple high-quality initial configurations
    best_ratio = 0.0
    best_solution = None
    
    # Strategy 1: Improved icosahedral-based initialization (highly symmetric)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
    
    # Add 2 more points to make 14 - use antipodal points for symmetry
    additional_points = np.array([[0, 0, 0.98], [0, 0, -0.98]])  # Slightly offset for better spread
    
    initial_points_1 = np.vstack([ico_vertices, additional_points])
    
    # Strategy 2: Fibonacci spiral on sphere (good distribution) - improved version
    fib_points = np.zeros((14, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(14):
        # Use Fibonacci spiral with better distribution
        y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = np.arccos(y)
        phi_val = (i * golden_ratio) % (2 * np.pi)
        fib_points[i] = [radius * np.sin(theta) * np.cos(phi_val),
                        radius * np.sin(theta) * np.sin(phi_val),
                        radius * np.cos(theta)]
    
    # Strategy 3: Random points with normalization
    np.random.seed(42)  # Fixed seed for reproducibility
    random_points = np.random.randn(14, 3)
    # Apply normalization
    norms = np.linalg.norm(random_points, axis=1)
    norms = np.where(norms == 0, 1, norms)
    random_points = random_points / norms[:, np.newaxis]
    
    # Strategy 4: Modified face-centered cubic lattice projection
    # Generate points that are more evenly distributed
    fc_points = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 1, 1], [1, 0.5, 1]
    ])
    # Take first 14 points and project to sphere
    fc_points = fc_points[:14]
    norms = np.linalg.norm(fc_points, axis=1)
    fc_points = fc_points / norms[:, np.newaxis]
    
    # Strategy 5: Better octahedral arrangement with polar points
    octahedral_points = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [0.707, 0.707, 0], [0.707, -0.707, 0], [-0.707, 0.707, 0], [-0.707, -0.707, 0],
        [0.707, 0, 0.707], [0.707, 0, -0.707], [-0.707, 0, 0.707], [-0.707, 0, -0.707]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(octahedral_points, axis=1)
    octahedral_points = octahedral_points / norms[:, np.newaxis]
    
    strategies = [initial_points_1, fib_points, random_points, fc_points, octahedral_points]
    
    # Track start time for timeout control
    start_time = time.time()
    
    # Try optimization with different strategies and multiple restarts
    for strategy_idx, initial_points in enumerate(strategies):
        # Multiple restarts for each strategy to increase chance of finding good local optima
        for restart in range(10):  # More restarts for better exploration
            # Check if we're running out of time
            if time.time() - start_time > 55:  # Leave 5 seconds for final steps
                break
                
            # Start with base configuration
            current_points = initial_points.copy()
            
            # Add small random perturbations for each restart
            np.random.seed(strategy_idx * 1000 + restart)  # Better seed mixing
            # Use smaller perturbations for more stable convergence
            perturbation = np.random.normal(0, 0.02, (14, 3))
            current_points += perturbation
            
            # Normalize to unit sphere again to maintain constraint
            norms = np.linalg.norm(current_points, axis=1)
            current_points = current_points / norms[:, np.newaxis]
            
            # Try physics-based refinement first with improved parameters
            refined_points = energy_minimization(current_points, max_iter=1000, learning_rate=0.005, alpha=3.0)
            
            # Then optimize with constrained optimization
            x0 = refined_points.flatten()
            
            # Define constraint for unit sphere
            constraint = {'type': 'eq', 'fun': sphere_constraint}
            
            try:
                # Use multiple optimization methods for robustness
                optimizers = ['trust-constr', 'SLSQP', 'L-BFGS-B']
                
                for method in optimizers:
                    try:
                        # More aggressive optimization settings for better results
                        result = minimize(
                            objective_function,
                            x0,
                            method=method,
                            constraints=constraint,
                            options={
                                'maxiter': 1000,   # More iterations
                                'ftol': 1e-12,     # Tighter tolerance
                                'gtol': 1e-12,     # Tighter tolerance
                                'disp': False
                            },
                            tol=1e-12
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 3)
                            ratio = compute_min_max_ratio(optimized_points)
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_solution = optimized_points.copy()
                        break  # Success, move to next strategy
                    except Exception:
                        # Continue with other methods if one fails
                        continue
                        
            except Exception:
                continue
    
    # Additional global optimization using differential evolution
    if best_solution is not None and best_ratio < 0.45 and time.time() - start_time < 50:
        try:
            from scipy.optimize import differential_evolution
            
            def de_objective(x_flat):
                points = x_flat.reshape(-1, 3)
                # Normalize to unit sphere
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                points = points / norms
                ratio = compute_min_max_ratio(points)
                return -ratio  # Negative for minimization
            
            bounds = [(-1.0, 1.0) for _ in range(3*14)]
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=1000,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                tol=1e-12,
                atol=1e-12
            )
            
            if de_result.success:
                final_points = de_result.x.reshape(-1, 3)
                # Normalize to unit sphere
                norms = np.linalg.norm(final_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                final_points = final_points / norms
                
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = final_points.copy()
                    
        except Exception:
            pass
    
    # If no good optimization found, return the best initial configuration
    if best_solution is None:
        # Return the first strategy as fallback
        return initial_points_1
    
    return best_solution


# EVOLVE-BLOCK-END
