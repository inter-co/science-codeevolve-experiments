# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import warnings
import math
import random
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, smooth approximations, 
    differential evolution for global search, and local optimization with simulated annealing refinement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    
    # Smooth approximation of min distance using log-sum-exp trick
    def smooth_min_distance(points_flat, k=100):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid self-distances
        np.fill_diagonal(distances, np.inf)
        # Use smooth approximation: -log(sum(exp(-k*d))) for large k
        smooth_min = -np.log(np.sum(np.exp(-k * distances))) / k
        return smooth_min
    
    # Smooth approximation of max distance
    def smooth_max_distance(points_flat, k=100):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to zero to avoid self-distances
        np.fill_diagonal(distances, 0)
        # Use smooth approximation: log(sum(exp(k*d))) / k for large k
        smooth_max = np.log(np.sum(np.exp(k * distances))) / k
        return smooth_max
    
    # Objective function with smoothing for differentiability
    def objective_with_smoothing(points_flat):
        min_dist = smooth_min_distance(points_flat)
        max_dist = smooth_max_distance(points_flat)
        # We want to maximize min_dist/max_dist, so we minimize -min_dist/max_dist
        # But we also want to penalize when max_dist approaches 0
        if max_dist < 1e-10:
            return 1e10
        return -min_dist / max_dist
    
    # Standard objective function for final evaluation (exact calculation)
    def objective_exact(x_flat):
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return float('inf')
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0 or d_min <= 0:
            return float('inf')
        return -d_min / d_max
    
    # Compute the ratio of minimum to maximum distance directly
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        # Find min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    # Initialize with a good starting configuration (hexagonal packing approximation)
    def generate_hexagonal_initial():
        # Create hexagonal grid pattern that approximates optimal distribution
        points = []
        rows = 4
        cols = 4
        spacing = 1.0
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing + (i % 2) * spacing/2
                y = i * spacing * np.sqrt(3)/2
                points.append([x, y])
        
        # Normalize to unit square [0,1] x [0,1]
        points = np.array(points[:n])
        # Handle edge cases more carefully
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        else:
            points[:, 0] = 0.5  # Set to middle if all same
            
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        else:
            points[:, 1] = 0.5  # Set to middle if all same
            
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        for i in range(n):
            points[i, 0] += (np.random.rand() - 0.5) * 0.02
            points[i, 1] += (np.random.rand() - 0.5) * 0.02
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Alternative initialization: golden spiral pattern
    def golden_spiral_initial():
        points = np.zeros((n, 2))
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1
            points[i] = [radius * np.cos(angle), radius * np.sin(angle)]
        # Normalize to [0,1] x [0,1] with careful edge case handling
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        else:
            points[:, 0] = 0.5
            
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        else:
            points[:, 1] = 0.5
            
        return points
    
    # Concentric circle pattern for better coverage
    def concentric_circles_initial():
        points = []
        # Create points in concentric circles
        radii = [0.1, 0.3, 0.5, 0.7, 0.9]
        points_per_ring = [1, 4, 8, 2, 1]  # Adjust for 16 points total
        
        idx = 0
        for r, count in zip(radii, points_per_ring):
            if idx + count > n:
                count = n - idx
            angles = np.linspace(0, 2*np.pi, count+1)[:-1]  # Avoid duplicate at 2pi
            for angle in angles:
                points.append([r * np.cos(angle) + 0.5, r * np.sin(angle) + 0.5])
                idx += 1
                if idx >= n:
                    break
            if idx >= n:
                break
        
        # Fill remaining points randomly
        while len(points) < n:
            points.append([np.random.rand(), np.random.rand()])
            
        return np.array(points[:n])
    
    # Simulated Annealing refinement (borrowed from Program 1)
    def simulated_annealing_refinement(initial_points):
        """Refine solution using simulated annealing for better convergence."""
        
        def neighbor_step(points, step_size=0.05):
            """Generate a neighboring solution by perturbing one point."""
            new_points = points.copy()
            # Choose a random point to perturb
            idx = random.randint(0, len(points) - 1)
            # Add small random perturbation
            new_points[idx] += np.random.normal(0, step_size, 2)
            # Keep points within [0,1] bounds
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            return new_points
        
        def acceptance_probability(old_energy, new_energy, temperature):
            """Calculate probability of accepting worse solution."""
            if new_energy < old_energy:
                return 1.0
            return math.exp((old_energy - new_energy) / temperature)
        
        # Initialize with the given points
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Use more precise cooling rate from successful approach (0.99985)
        temperature = 1.0
        min_temperature = 1e-12
        cooling_rate = 0.99985  # More precise cooling rate
        max_iterations = 55000  # More iterations for better convergence
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Generate neighbor solution
            new_points = neighbor_step(current_points, 0.05)
            
            # Calculate energies
            current_energy = objective_exact(current_points.flatten())
            new_energy = objective_exact(new_points.flatten())
            
            # Accept or reject new solution
            if acceptance_probability(current_energy, new_energy, temperature) > random.random():
                current_points = new_points
            
            # Update best solution
            current_ratio = compute_min_max_ratio(current_points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()
            
            # Cool down
            temperature *= cooling_rate
            
            # Stop if temperature gets too low
            if temperature < min_temperature:
                break
        
        # Final polishing stage with very fine adjustments
        if best_ratio > 0.05:
            # Fine-tune with even smaller steps for better results
            fine_tune_steps = 25000
            for _ in range(fine_tune_steps):
                new_points = neighbor_step(best_points, 0.0015)  # Even smaller steps
                new_energy = objective_exact(new_points.flatten())
                current_energy = objective_exact(best_points.flatten())
                
                # Accept with extremely low probability for very bad moves
                if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / (temperature * 0.0005)):
                    best_points = new_points
                    
                # Update best if we improved
                current_ratio = compute_min_max_ratio(best_points)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
        
        return best_points
    
    # Multi-start optimization with diverse initializations
    best_result = None
    best_ratio = -np.inf
    
    # Strategy 1: Differential Evolution for global search (from Program 2)
    try:
        bounds = [(0, 1) for _ in range(2 * n)]
        de_result = differential_evolution(
            objective_exact,
            bounds,
            maxiter=60,
            popsize=18,
            seed=42,
            disp=False,
            tol=1e-8
        )
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            ratio = compute_min_max_ratio(de_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_result = de_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Hexagonal initialization with local optimization
    try:
        initial_points = generate_hexagonal_initial()
        bounds = [(0, 1) for _ in range(2*n)]
        result = minimize(
            objective_with_smoothing,
            initial_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_result = final_points.copy()
    except Exception:
        pass
    
    # Strategy 3: Golden spiral initialization with local optimization
    try:
        initial_points = golden_spiral_initial()
        bounds = [(0, 1) for _ in range(2*n)]
        result = minimize(
            objective_with_smoothing,
            initial_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_result = final_points.copy()
    except Exception:
        pass
    
    # Strategy 4: Concentric circles initialization with local optimization
    try:
        initial_points = concentric_circles_initial()
        bounds = [(0, 1) for _ in range(2*n)]
        result = minimize(
            objective_with_smoothing,
            initial_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        if result.success:
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            ratio = compute_min_max_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_result = final_points.copy()
    except Exception:
        pass
    
    # Strategy 5: Multiple restarts with random initialization
    restart_count = 4  # Reduce from 5 to 4 to save time
    for restart in range(restart_count):
        try:
            # Generate random initial points
            initial_points = np.random.rand(n, 2)
            
            # Add some noise to make it more interesting
            np.random.seed(100 + restart)
            initial_points += np.random.normal(0, 0.01, initial_points.shape)
            initial_points = np.clip(initial_points, 0, 1)
            
            bounds = [(0, 1) for _ in range(2*n)]
            result = minimize(
                objective_with_smoothing,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = final_points.copy()
        except Exception:
            continue
    
    # Strategy 6: Final refinement using simulated annealing for better convergence
    # Only do this for solutions that are clearly good to save time
    if best_result is not None and best_ratio > 0.06:
        try:
            # Apply simulated annealing refinement
            refined_points = simulated_annealing_refinement(best_result)
            ratio = compute_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_result = refined_points
        except Exception:
            pass
    
    # If we have no valid results, fall back to hexagonal pattern
    if best_result is None:
        return generate_hexagonal_initial()
    
    return best_result


# EVOLVE-BLOCK-END
