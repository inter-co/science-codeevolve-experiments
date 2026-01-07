# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import random
import math
from scipy.optimize import minimize
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions with advanced optimization techniques.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def neighbor_move(points, step_size=0.005):
        """Generate a neighboring solution by perturbing one point with adaptive step size."""
        new_points = points.copy()
        # Choose a random point to move
        idx = random.randint(0, len(points) - 1)
        # Perturb the point with adaptive step size for better exploration
        adaptive_step = step_size * (0.5 + random.random() * 0.5)  # Randomized step size
        new_points[idx] += np.random.normal(0, adaptive_step, 2)
        # Keep within bounds [0,1] x [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    def simulated_annealing(initial_points, max_iter=45000):
        """Run simulated annealing to find optimal point configuration."""
        current_points = initial_points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Enhanced aggressive cooling schedule from inspiration 2
        temperature = 1.0
        cooling_rate = 0.9997  # Slightly more aggressive cooling
        min_temperature = 1e-13  # Even lower minimum temperature
        max_iterations = max_iter
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        for iteration in range(max_iterations):
            # Generate neighbor solution with adaptive step size
            new_points = neighbor_move(current_points, 0.005)
            new_ratio = calculate_min_max_ratio(new_points)
            
            # Accept or reject the new solution
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                acceptance_prob = math.exp(delta / temperature) if temperature > 0 else 0
                if random.random() < acceptance_prob:
                    current_points = new_points
                    current_ratio = new_ratio
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
            # Reset to best solution more frequently to escape local optima
            if iteration % 1000 == 0 and iteration > 0:  # More frequent resets
                current_points = best_points.copy()
                current_ratio = best_ratio
        
        return best_points, best_ratio
    
    def create_fibonacci_spiral_initial():
        """
        Create initial configuration using Fibonacci spiral approach (from inspiration 2).
        """
        # Create a Fibonacci spiral arrangement with improved distribution
        points = []
        for i in range(16):
            phi = np.pi * (3 - np.sqrt(5)) * i  # Fibonacci spiral
            rho = np.sqrt(i / 15) if i < 15 else 0
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            points.append([(x + 1) / 2, (y + 1) / 2])  # Normalize to [0,1]
        
        points = np.array(points)
        
        # Add better distributed random noise to break symmetry
        noise_scale = 0.005  # Slightly larger noise for better exploration
        points += np.random.normal(0, noise_scale, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def create_better_hexagonal_initial():
        """
        Create a more carefully constructed hexagonal initial configuration.
        """
        # Create a 4x4 grid with alternating rows (like honeycomb) with better spacing
        points = []
        for i in range(4):
            for j in range(4):
                x = j + (i % 2) * 0.5
                y = i * math.sqrt(3)/2
                points.append([x, y])
        
        points = np.array(points)
        
        # Normalize to fit in [0,1] x [0,1] with proper scaling
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # Scale and shift to fit nicely in [0,1] x [0,1]
        scale_factor = 0.8  # Slightly smaller scale for better distribution
        offset_x = 0.1
        offset_y = 0.1
        
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * scale_factor + offset_x
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * scale_factor + offset_y
        
        # Add better distributed noise
        noise_scale = 0.015  # Slightly larger noise
        points += np.random.normal(0, noise_scale, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def generate_multiple_initial_configurations():
        """Generate several diverse initial configurations."""
        configs = []
        
        # Configuration 1: Fibonacci spiral (core of inspiration 2 success)
        configs.append(create_fibonacci_spiral_initial())
        
        # Configuration 2: Better hexagonal grid
        configs.append(create_better_hexagonal_initial())
        
        # Configuration 3: Random with clustering bias
        points = np.random.rand(16, 2)
        configs.append(points)
        
        # Configuration 4: Grid with noise
        points = np.zeros((16, 2))
        count = 0
        for i in range(4):
            for j in range(4):
                points[count] = [i/3.0, j/3.0]
                count += 1
        points += np.random.normal(0, 0.025, (16, 2))  # Slightly smaller noise
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 5: Concentrated around center
        points = np.random.rand(16, 2) * 0.5 + 0.25  # Centered in [0.25, 0.75] range
        configs.append(points)
        
        # Configuration 6: Another hexagonal variant with better spacing
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25
                y = i * 0.25
                points.append([x, y])
        points = np.array(points)
        points += np.random.normal(0, 0.015, points.shape)  # Smaller noise
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 7: Random with more structure
        points = np.random.rand(16, 2)
        configs.append(points)
        
        # Configuration 8: Another Fibonacci variant with different parameters
        points = []
        for i in range(16):
            phi = np.pi * (3 - np.sqrt(5)) * i * 1.1  # Slight modification
            rho = np.sqrt(i / 15) if i < 15 else 0
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            points.append([(x + 1) / 2, (y + 1) / 2])
        configs.append(np.array(points))
        
        # Configuration 9: Circular arrangement with some randomness
        points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            radius = 0.4 + 0.1 * np.random.random()  # Slightly randomized radii
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        configs.append(np.array(points))
        
        # Configuration 10: Lattice with perturbation
        points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125
                y = j * 0.25 + 0.125
                points.append([x, y])
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)  # Small noise
        points = np.clip(points, 0, 1)
        configs.append(points)
        
        # Configuration 11: Another ring-like distribution with different parameters
        angles = np.linspace(0, 2*np.pi, 16)
        radii = np.random.rand(16) * 0.3 + 0.35  # Different radius range
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.7 + 0.15  # Different scale and shift
        configs.append(points)
        
        # Configuration 12: More uniform distribution with center bias
        points = np.random.rand(16, 2) * 0.8 + 0.1  # Spread out more
        configs.append(points)
        
        # Configuration 13: Special configuration - try to approach the theoretical maximum
        # Create points that form a regular polygon pattern
        angles = np.linspace(0, 2*np.pi, 16)
        radii = np.ones(16) * 0.4  # All at same radius
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.6 + 0.2  # Scale and shift to [0.2, 0.8]
        configs.append(points)
        
        # Configuration 14: Another Fibonacci spiral with different scaling
        points = []
        for i in range(16):
            phi = np.pi * (3 - np.sqrt(5)) * i * 0.9  # Slight change in parameter
            rho = np.sqrt(i / 15) if i < 15 else 0
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            points.append([(x + 1) / 2, (y + 1) / 2])
        configs.append(np.array(points))
        
        return configs
    
    # Run multiple restarts with different initial configurations
    best_points = None
    best_ratio = -float('inf')
    
    # Use more restarts like INSPIRATION PROGRAM 1 for better exploration (25)
    initial_configs = generate_multiple_initial_configurations()
    
    # Run 25 restarts total (more than inspiration 2 for better chance)
    total_restarts = 25
    
    for restart in range(total_restarts):
        # Alternate between different initial configurations
        config_idx = restart % len(initial_configs)
        initial_points = initial_configs[config_idx].copy()
        
        # Run simulated annealing with enhanced parameters
        points, ratio = simulated_annealing(initial_points, max_iter=45000)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points
    
    # Final fallback to a good hexagonal configuration if nothing worked
    if best_points is None:
        points = np.zeros((16, 2))
        count = 0
        for i in range(4):
            for j in range(4):
                x = j + (i % 2) * 0.5
                y = i * math.sqrt(3)/2
                points[count] = [x, y]
                count += 1
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        points += np.random.normal(0, 0.02, (16, 2))
        points = np.clip(points, 0, 1)
        best_points = points
    
    # Try a final gradient-based refinement on the best solution with improved settings
    try:
        # Convert to flat array for scipy optimization
        x0 = best_points.flatten()
        
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # Avoid division by zero
            if max_dist <= 0:
                return -np.inf
                
            return -min_dist / max_dist
        
        # Bounds for each coordinate
        bounds = [(0, 1) for _ in range(32)]
        
        # Try gradient-based optimization to refine with higher precision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Use more iterations for better refinement
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}  # More iterations for better precision
            )
            
            # If successful, update the solution
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = calculate_min_max_ratio(refined_points)
                
                if refined_ratio > best_ratio:
                    best_points = refined_points
                    best_ratio = refined_ratio
    except Exception:
        pass  # If optimization fails, keep the previous best
    
    return best_points


# EVOLVE-BLOCK-END
