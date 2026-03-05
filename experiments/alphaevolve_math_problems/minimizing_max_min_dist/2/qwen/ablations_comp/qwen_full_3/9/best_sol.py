# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import math
from typing import Tuple


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple algorithmic paradigms with enhanced optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points: np.ndarray) -> Tuple[float, float, float]:
        """Compute min/max distance ratio and actual values"""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
            
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        ratio = d_min / d_max if d_max > 0 else 0.0
        return ratio, d_min, d_max
    
    def objective(points_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = points_flat.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist  # Negative because we want to maximize
    
    def physics_based_approach() -> np.ndarray:
        """Enhanced physics-based approach with better force modeling"""
        # Initialize points with better distribution
        np.random.seed(42)
        points = np.random.rand(16, 2)
        
        # Physics parameters - optimized for faster convergence and better results
        max_iter = 300
        learning_rate = 0.03
        repulsion_strength = 3.0
        
        for iteration in range(max_iter):
            # Compute pairwise forces
            forces = np.zeros_like(points)
            
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    
                    if dist_sq > 0:
                        # More sophisticated repulsive force with better damping
                        force_magnitude = repulsion_strength / (dist_sq + 0.001)
                        force = force_magnitude * diff / np.sqrt(dist_sq)
                        forces[i] += force
                        forces[j] -= force
            
            # Apply forces with boundary constraints
            points += learning_rate * forces
            
            # Keep points within bounds
            points = np.clip(points, 0, 1)
            
            # Occasionally apply small random perturbations to escape local minima
            if iteration % 30 == 0 and iteration > 0:
                points += np.random.normal(0, 0.003, points.shape)
                points = np.clip(points, 0, 1)
        
        return points
    
    def grid_based_approach() -> np.ndarray:
        """Improved grid-based approach with strategic perturbations"""
        # Create a grid pattern with strategic perturbations
        points = np.zeros((16, 2))
        row_positions = np.linspace(0.1, 0.9, 4)
        col_positions = np.linspace(0.1, 0.9, 4)
        
        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if idx < 16:
                    # Add strategic jitter to avoid degenerate cases
                    jitter_x = (np.sin(i * 0.7) + np.cos(j * 0.5)) * 0.025
                    jitter_y = (np.cos(i * 0.5) + np.sin(j * 0.7)) * 0.025
                    
                    x = max(0.01, min(0.99, col + jitter_x))
                    y = max(0.01, min(0.99, row + jitter_y))
                    points[idx] = [x, y]
                    idx += 1
        return points
    
    def circular_arrangement() -> np.ndarray:
        """Enhanced circular arrangement with better spacing"""
        points = []
        radius = 0.4
        center = [0.5, 0.5]
        
        # Place points around circle with more even distribution
        for i in range(16):
            angle = 2 * math.pi * i / 16
            # Add slight perturbation for better dispersion
            angle += np.random.normal(0, 0.05)
            x = center[0] + radius * math.cos(angle) + np.random.normal(0, 0.01)
            y = center[1] + radius * math.sin(angle) + np.random.normal(0, 0.01)
            # Keep within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            points.append([x, y])
        return np.array(points)
    
    def fibonacci_spiral() -> np.ndarray:
        """Improved Fibonacci-like spiral pattern"""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        center = [0.5, 0.5]
        radius = 0.4
        
        for i in range(16):
            # Fibonacci spiral pattern with better distribution
            theta = i * 2 * math.pi / golden_ratio
            r = radius * math.sqrt(i / 15.0) if i > 0 else 0.01
            
            # Add some randomness to break symmetry
            theta += np.random.normal(0, 0.1)
            r += np.random.normal(0, 0.02)
            
            x = center[0] + r * math.cos(theta) + np.random.normal(0, 0.01)
            y = center[1] + r * math.sin(theta) + np.random.normal(0, 0.01)
            
            # Keep within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            points.append([x, y])
        return np.array(points)
    
    def hexagonal_pattern() -> np.ndarray:
        """Hexagonal pattern approach for better spacing"""
        points = []
        # Create points in a hexagonal lattice pattern
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Offset every other row for hexagonal packing
                offset = (i % 2) * 0.25
                x = offset + j * 0.25 + np.random.normal(0, 0.015)
                y = i * 0.25 + np.random.normal(0, 0.015)
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
        return np.array(points[:16])
    
    def regular_polygon_approach() -> np.ndarray:
        """Regular polygon approach with mathematical precision"""
        # Create points on a regular 16-gon
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([
            0.5 + radius * np.cos(angles),
            0.5 + radius * np.sin(angles)
        ])
        # Add small random perturbations to break symmetry and improve distribution
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        return points
    
    def global_optimization_approach() -> np.ndarray:
        """Use global optimization for best possible solution"""
        # Use differential evolution for global optimization - more robust than L-BFGS
        np.random.seed(42)
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,  # Increased iterations for better convergence
                popsize=15,   # Larger population for better exploration
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                tol=1e-6
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Fallback to random initialization if optimization fails
        return np.random.rand(16, 2)
    
    def simulated_annealing_approach(initial_points: np.ndarray) -> np.ndarray:
        """Enhanced simulated annealing with better cooling schedule"""
        points = initial_points.copy()
        current_ratio, _, _ = compute_min_max_ratio(points)
        
        # Improved annealing parameters - faster convergence
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 0.00001
        iterations_per_temp = 30
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make a small random change
                test_points = points.copy()
                idx = np.random.randint(0, len(test_points))
                test_points[idx] += np.random.normal(0, 0.008, 2)
                test_points = np.clip(test_points, 0, 1)
                
                # Evaluate
                test_ratio, _, _ = compute_min_max_ratio(test_points)
                
                # Accept or reject based on Metropolis criterion
                if test_ratio > current_ratio or np.random.random() < np.exp((test_ratio - current_ratio) / temperature):
                    points = test_points
                    current_ratio = test_ratio
                    
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = points.copy()
            
            temperature *= cooling_rate
        
        return best_points
    
    # Try multiple approaches and select the best
    approaches = [
        ("global_opt", global_optimization_approach),  # Use global optimization first
        ("regular_polygon", regular_polygon_approach),
        ("physics", physics_based_approach),
        ("grid", grid_based_approach),
        ("circular", circular_arrangement),
        ("fibonacci", fibonacci_spiral),
        ("hexagonal", hexagonal_pattern)
    ]
    
    best_ratio = 0.0
    best_points = None
    
    # First try global optimization which is more likely to find better solutions
    try:
        global_points = global_optimization_approach()
        global_ratio, _, _ = compute_min_max_ratio(global_points)
        if global_ratio > best_ratio:
            best_ratio = global_ratio
            best_points = global_points
    except Exception:
        pass
    
    # Then try other approaches with refinement
    for name, approach_func in approaches:
        if name == "global_opt":
            continue  # Skip since we already tried it
            
        try:
            points = approach_func()
            
            # Refine with simulated annealing for final optimization
            refined_points = simulated_annealing_approach(points)
            refined_ratio, _, _ = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points
                
        except Exception as e:
            continue
    
    # If no good solution was found, fall back to physics approach
    if best_points is None:
        points = physics_based_approach()
        best_points = simulated_annealing_approach(points)
    
    # Final refinement step using local optimization on the best result
    # This follows INSPIRATION PROGRAM 1's approach more closely
    if best_points is not None:
        try:
            from scipy.optimize import minimize
            
            def objective(x_flat):
                """Objective function to minimize (negative of min/max ratio)"""
                points = x_flat.reshape(-1, 2)
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 0
                return -min_dist / max_dist  # Negative because we want to maximize
            
            # Flatten the best configuration
            x0 = best_points.flatten()
            
            # Define bounds for optimization (points in [0,1] x [0,1])
            bounds = [(0, 1) for _ in range(32)]
            
            # Use multiple local optimization methods for robustness
            methods = ['L-BFGS-B', 'SLSQP']
            best_local_points = best_points
            best_local_ratio = best_ratio
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
                    )
                    
                    if result.success:
                        local_points = result.x.reshape(-1, 2)
                        local_ratio, _, _ = compute_min_max_ratio(local_points)
                        
                        if local_ratio > best_local_ratio:
                            best_local_ratio = local_ratio
                            best_local_points = local_points
                except Exception:
                    continue
            
            # Update if local optimization improved the result
            if best_local_ratio > best_ratio:
                best_ratio = best_local_ratio
                best_points = best_local_points
        except Exception:
            pass
    
    # Fallback to a mathematically sound configuration if nothing worked
    if best_points is None:
        # Use a regular polygon approach which provides good theoretical distribution
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        points = np.column_stack([
            0.5 + 0.4 * np.cos(angles),
            0.5 + 0.4 * np.sin(angles)
        ])
        best_points = points
    
    return best_points


# EVOLVE-BLOCK-END
