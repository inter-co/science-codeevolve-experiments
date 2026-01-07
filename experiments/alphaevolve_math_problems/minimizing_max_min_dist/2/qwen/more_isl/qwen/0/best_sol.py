# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import random
import math
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import qr
import cmath
from sklearn.cluster import KMeans


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions with advanced optimization techniques,
    including graph-theoretic methods and multiple restart strategies.

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
    
    def neighbor_move(points, step_size=0.01):
        """Generate a neighboring solution by perturbing one point."""
        new_points = points.copy()
        # Choose a random point to move
        idx = random.randint(0, len(points) - 1)
        # Perturb the point with very small step size for fine-tuning
        new_points[idx] += np.random.normal(0, step_size, 2)
        # Keep within bounds [0,1] x [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    def simulated_annealing(initial_points, max_iter=150000, cooling_rate=0.99985):
        """Run simulated annealing to find optimal point configuration."""
        current_points = initial_points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Very aggressive cooling schedule for better exploration
        temperature = 1.0
        min_temperature = 1e-12
        max_iterations = max_iter
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        for iteration in range(max_iterations):
            # Generate neighbor solution with very small steps for fine-tuning
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
                acceptance_prob = math.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_points = new_points
                    current_ratio = new_ratio
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
            # Reset to best solution periodically to escape local optima
            if iteration % 1500 == 0 and iteration > 0:
                current_points = best_points.copy()
                current_ratio = best_ratio
        
        return best_points
    
    def get_icosahedral_projection() -> np.ndarray:
        """
        Create initial configuration using icosahedral symmetry projection.
        This is one of the most mathematically principled approaches for point dispersion.
        """
        # Generate vertices of a regular icosahedron (12 vertices)
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        # Vertices of a regular icosahedron scaled to unit sphere
        ico_points = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        # Normalize to unit sphere
        norm_points = []
        for x, y, z in ico_points:
            norm = math.sqrt(x*x + y*y + z*z)
            norm_points.append((x/norm, y/norm, z/norm))
        
        # Project to 2D using stereographic projection from south pole
        proj_points = []
        for x, y, z in norm_points:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            proj_points.append((proj_x, proj_y))
        
        # Convert to array and adjust to 2D points
        proj_array = np.array(proj_points)
        
        # Add 4 more points strategically to reach 16
        # Use a pattern that maintains good dispersion
        additional_positions = [
            [0.25, 0.25],
            [0.75, 0.25], 
            [0.25, 0.75],
            [0.75, 0.75]
        ]
        
        # Combine all points
        points = []
        for i in range(12):
            # Scale to fit in [0,1] x [0,1]
            x = (proj_array[i][0] + 1) / 2
            y = (proj_array[i][1] + 1) / 2
            points.append([x, y])
        
        for pos in additional_positions:
            points.append(pos)
        
        points = np.array(points[:16])  # Ensure exactly 16 points
        
        # Add slight noise to break any remaining symmetry
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_hexagonal_grid() -> np.ndarray:
        """
        Create a hexagonal grid configuration which provides good baseline distribution.
        """
        # Create a 4x4 grid with alternating rows (like honeycomb)
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
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        # Add noise
        points += np.random.normal(0, 0.015, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def get_graph_theoretic_initial() -> np.ndarray:
        """
        Create an initial configuration using a graph-theoretic approach.
        This attempts to create a configuration where points are connected by edges
        with weights representing desired distances, mimicking a force-directed layout.
        """
        # Start with a hexagonal grid
        points = get_hexagonal_grid()
        
        # Apply a simple force-directed approach to spread points
        # This is a simplified version of a graph embedding approach
        for _ in range(100):  # Few iterations for speed
            # Calculate forces between all points
            distances = pdist(points)
            dist_matrix = np.zeros((16, 16))
            idx = 0
            for i in range(16):
                for j in range(i+1, 16):
                    dist_matrix[i,j] = distances[idx]
                    dist_matrix[j,i] = distances[idx]
                    idx += 1
            
            # Apply repulsive forces (inverse of distance squared)
            for i in range(16):
                for j in range(16):
                    if i != j and dist_matrix[i,j] > 0:
                        force_magnitude = 1.0 / (dist_matrix[i,j]**2 + 1e-8)
                        # Normalize direction
                        direction = points[j] - points[i]
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 0:
                            force = force_magnitude * direction / direction_norm
                            points[i] += force * 0.001
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    def get_fibonacci_spiral() -> np.ndarray:
        """
        Create points using Fibonacci spiral pattern which is known for good distribution.
        """
        points = []
        for i in range(16):
            phi = np.pi * (3 - np.sqrt(5)) * i
            rho = np.sqrt(i / 15) if i < 15 else 0
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            points.append([(x + 1) / 2, (y + 1) / 2])
        return np.array(points)
    
    # Generate multiple initial configurations using different approaches
    initial_configs = []
    
    # 1. Icosahedral projection (mathematical foundation)
    initial_configs.append(get_icosahedral_projection())
    
    # 2. Hexagonal grid (structured approach)
    initial_configs.append(get_hexagonal_grid())
    
    # 3. Graph-theoretic approach (novel methodology)
    initial_configs.append(get_graph_theoretic_initial())
    
    # 4. Fibonacci spiral (good distribution property)
    initial_configs.append(get_fibonacci_spiral())
    
    # 5. Random with clustering bias (exploration)
    initial_configs.append(np.random.rand(16, 2))
    
    # Run multiple restarts with different initial configurations
    best_points = None
    best_ratio = -float('inf')
    
    # Run 8 restarts total for better exploration (reduced from 10 to stay within time budget)
    for restart in range(8):
        # Select initial configuration based on restart number
        config_idx = restart % len(initial_configs)
        initial_points = initial_configs[config_idx].copy()
        
        # Run simulated annealing with optimized parameters
        points = simulated_annealing(initial_points, max_iter=100000, cooling_rate=0.99985)
        ratio = calculate_min_max_ratio(points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points
    
    # If no good solution was found, fallback to a well-tested hexagonal approach
    if best_points is None:
        points = get_hexagonal_grid()
        best_points = points
    
    # Apply a final multi-stage optimization to fine-tune the best solution
    final_points = best_points.copy()
    
    # Stage 1: Local optimization with L-BFGS-B (faster and more reliable)
    def objective_function(x):
        points = x.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = calculate_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    # Use L-BFGS-B for fast local refinement
    try:
        result = minimize(
            objective_function,
            final_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            final_ratio = calculate_min_max_ratio(optimized_points)
            
            if final_ratio > best_ratio:
                final_points = optimized_points
    except:
        pass
    
    # Stage 2: Differential evolution for global search (with reduced iterations)
    def de_objective(x):
        points = x.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = calculate_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    bounds_de = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    try:
        # Run with fewer iterations to stay within time budget
        de_result = differential_evolution(
            de_objective,
            bounds_de,
            maxiter=75,  # Reduced from 100 to save time
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_ratio = calculate_min_max_ratio(de_points)
            
            if de_ratio > best_ratio:
                final_points = de_points.copy()
    except:
        pass
    
    return final_points


# EVOLVE-BLOCK-END
