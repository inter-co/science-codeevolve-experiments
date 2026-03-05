# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import math
from scipy.optimize import differential_evolution
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a combination of geometric constructions, multi-start optimization, and physics-inspired approaches.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Strategy 1: Generate multiple high-quality initial configurations with physics-inspired methods
    def generate_initial_configurations():
        configs = []
        
        # Configuration 1: Golden spiral pattern (inspired by successful approaches)
        np.random.seed(42)
        points = []
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            angle = i * 2 * math.pi / phi
            radius = math.sqrt(i) / math.sqrt(15) if i > 0 else 0.5
            x = 0.5 + radius * math.cos(angle) * 0.45
            y = 0.5 + radius * math.sin(angle) * 0.45
            points.append([x, y])
        configs.append(("golden_spiral", np.array(points)))
        
        # Configuration 2: 3-ring hexagonal pattern (our best-performing approach)
        np.random.seed(43)
        points = np.zeros((16, 2))
        points[0] = [0.5, 0.5]  # Center point
        ring_radius1 = 0.25
        for i in range(6):
            angle = 2 * np.pi * i / 6
            points[i+1] = [
                0.5 + ring_radius1 * np.cos(angle),
                0.5 + ring_radius1 * np.sin(angle)
            ]
        ring_radius2 = 0.45
        for i in range(9):
            angle = 2 * np.pi * i / 9
            points[i+7] = [
                0.5 + ring_radius2 * np.cos(angle),
                0.5 + ring_radius2 * np.sin(angle)
            ]
        # Add small random perturbations
        for i in range(16):
            points[i, 0] += np.random.uniform(-0.02, 0.02)
            points[i, 1] += np.random.uniform(-0.02, 0.02)
        configs.append(("hex_3ring", np.clip(points, 0, 1)))
        
        # Configuration 3: Perturbed hexagonal grid (inspired by successful approaches)
        np.random.seed(44)
        points = []
        rows, cols = 4, 4
        spacing = 1.0 / (rows - 1)
        for i in range(rows):
            for j in range(cols):
                x = j * spacing
                if i % 2 == 1:
                    x += spacing * 0.5
                y = i * spacing
                x += (np.random.random() - 0.5) * 0.08
                y += (np.random.random() - 0.5) * 0.08
                points.append([x, y])
        configs.append(("hex_grid", np.clip(np.array(points), 0, 1)[:16]))
        
        # Configuration 4: Regular grid with jitter (robust baseline)
        np.random.seed(45)
        grid_x = np.linspace(0.05, 0.95, 4)
        grid_y = np.linspace(0.05, 0.95, 4)
        X, Y = np.meshgrid(grid_x, grid_y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        grid_points += np.random.normal(0, 0.03, grid_points.shape)
        configs.append(("grid", np.clip(grid_points, 0, 1)))
        
        # Configuration 5: Uniform random distribution
        np.random.seed(46)
        points = np.random.rand(16, 2)
        configs.append(("uniform_random", points))
        
        # Configuration 6: Circular pattern (inspiration from program 3)
        np.random.seed(49)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = []
        for angle in angles:
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        points = np.array(points) + np.random.normal(0, 0.03, (16, 2))
        configs.append(("circular", np.clip(points, 0, 1)))
        
        # Configuration 7: Physics-inspired repulsion model (new approach)
        # Start with random points and simulate electrostatic repulsion
        np.random.seed(50)
        repulsion_points = np.random.rand(16, 2)
        # Run simple physics simulation for stabilization
        for _ in range(300):  # More iterations for better stabilization (from INSPIRATION 1)
            dist_matrix = distance_matrix(repulsion_points, repulsion_points)
            # Avoid division by zero
            dist_matrix = np.maximum(dist_matrix, 1e-8)
            # Calculate forces (inverse square law)
            forces = np.zeros_like(repulsion_points)
            for i in range(16):
                for j in range(16):
                    if i != j:
                        diff = repulsion_points[i] - repulsion_points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 1e-8:
                            force_magnitude = 1.0 / (dist * dist)
                            forces[i] += force_magnitude * diff / dist
            
            # Update positions with small step size
            step_size = 0.001
            repulsion_points += step_size * forces
            # Keep within bounds
            repulsion_points = np.clip(repulsion_points, 0, 1)
        configs.append(("repulsion", repulsion_points))
        
        # Configuration 8: Optimized regular hexagon (inspired by sphere packing)
        np.random.seed(51)
        points = []
        # Create 3 concentric rings
        for k, radius in enumerate([0.2, 0.4, 0.6]):
            n_points = 6 if k == 0 else 12 if k == 1 else 16 - 6 - 12
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points + (k * 0.1)  # Small phase shift
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        # Trim to 16 points
        points = points[:16]
        # Add noise
        for i in range(16):
            points[i][0] += np.random.normal(0, 0.01)
            points[i][1] += np.random.normal(0, 0.01)
        configs.append(("hex_optimized", np.clip(np.array(points), 0, 1)))
        
        # Configuration 9: Concentric rings with varying density (additional diversity)
        np.random.seed(52)
        points = []
        # Add center point
        points.append([0.5, 0.5])
        # Add 3 rings with different densities
        for ring_idx, (radius, num_points) in enumerate([(0.25, 6), (0.45, 9), (0.65, 1)]):
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points + ring_idx * 0.2
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        # Make sure we have exactly 16 points
        points = points[:16]
        points = np.array(points) + np.random.normal(0, 0.02, (16, 2))
        configs.append(("concentric_rings", np.clip(np.array(points), 0, 1)))
        
        # Configuration 10: Spiral pattern with different parameters (more diversity)
        np.random.seed(53)
        points = []
        # Use a different spiral parameter
        for i in range(16):
            angle = i * 1.5 * math.pi / 2  # Different growth rate
            radius = math.sqrt(i) / math.sqrt(15) if i > 0 else 0.5
            x = 0.5 + radius * math.cos(angle) * 0.45
            y = 0.5 + radius * math.sin(angle) * 0.45
            points.append([x, y])
        configs.append(("spiral_alt", np.array(points)))
        
        # Configuration 11: Square grid with diagonal offsets (inspired by lattice structures)
        np.random.seed(54)
        points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * i / 3
                y = 0.1 + 0.8 * j / 3
                # Add diagonal offset for better distribution
                if (i + j) % 2 == 1:
                    x += 0.05
                    y += 0.05
                points.append([x, y])
        configs.append(("square_grid", np.clip(np.array(points), 0, 1)[:16]))
        
        # Configuration 12: Fibonacci-based arrangement (mathematically inspired)
        np.random.seed(55)
        points = []
        # Use Fibonacci sequence for better distribution
        fib = [1, 1]
        for i in range(14):
            fib.append(fib[-1] + fib[-2])
        
        for i in range(16):
            angle = 2 * np.pi * i / fib[-1]  # Using last Fibonacci number
            radius = 0.4 * np.sqrt(i / 15.0) if i > 0 else 0.0
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        configs.append(("fibonacci", np.array(points)))
        
        # Configuration 13: Octagonal pattern (inspired by higher dimensional constructions) - from INSPIRATION 1
        np.random.seed(56)
        points = []
        # Place points along octagon edges
        for i in range(16):
            if i < 8:
                # Outer ring
                angle = 2 * np.pi * i / 8
                x = 0.5 + 0.4 * np.cos(angle)
                y = 0.5 + 0.4 * np.sin(angle)
            else:
                # Inner ring
                angle = 2 * np.pi * (i - 8) / 8
                x = 0.5 + 0.2 * np.cos(angle)
                y = 0.5 + 0.2 * np.sin(angle)
            points.append([x, y])
        configs.append(("octagonal", np.clip(np.array(points), 0, 1)))
        
        # Configuration 14: Optimized circular arrangement with better radial distribution - from INSPIRATION 1
        np.random.seed(57)
        points = []
        # Distribute points more evenly in radial direction
        for i in range(16):
            angle = 2 * np.pi * i / 16
            # Use square root scaling for better distribution
            radius = 0.4 * np.sqrt(np.random.random()) 
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        configs.append(("radial_distribution", np.clip(np.array(points), 0, 1)))
        
        # Configuration 15: Fibonacci spiral with better distribution - from INSPIRATION 1
        np.random.seed(58)
        points = []
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            angle = i * 2 * math.pi / phi
            # Use logarithmic spiral with better radial distribution
            radius = 0.4 * (1 - np.exp(-i/5))  # Exponential decay for better spread
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        configs.append(("fibonacci_spiral", np.clip(np.array(points), 0, 1)))
        
        # Configuration 16: Improved concentric rings with better distribution
        np.random.seed(59)
        points = []
        # Add center point
        points.append([0.5, 0.5])
        # Add 3 rings with different densities and better angular distribution
        ring_radii = [0.2, 0.35, 0.5]
        ring_counts = [6, 8, 2]
        for ring_idx, (radius, num_points) in enumerate(zip(ring_radii, ring_counts)):
            for i in range(num_points):
                # Distribute points more uniformly
                angle = 2 * np.pi * i / num_points + ring_idx * 0.1
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        # Make sure we have exactly 16 points
        points = points[:16]
        points = np.array(points) + np.random.normal(0, 0.015, (16, 2))
        configs.append(("improved_concentric", np.clip(np.array(points), 0, 1)))
        
        # Configuration 17: Improved hexagonal arrangement with better spacing
        np.random.seed(60)
        points = []
        # Create 3 rings with better point distribution
        ring_radii = [0.2, 0.4, 0.6]
        ring_counts = [6, 10, 0]  # 16 total points
        point_idx = 0
        for ring_idx, (radius, num_points) in enumerate(zip(ring_radii, ring_counts)):
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points + ring_idx * 0.05  # Phase shift for better distribution
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
                point_idx += 1
        # Add remaining points
        if point_idx < 16:
            for i in range(16 - point_idx):
                # Add points in a circular pattern around the center
                angle = 2 * np.pi * i / (16 - point_idx)
                radius = 0.1
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        points = np.array(points)
        points = np.clip(points, 0, 1)
        configs.append(("improved_hex", points))
        
        return configs
    
    # Strategy 2: Enhanced objective function with better numerical handling
    def objective(x_flat):
        """Minimize negative of min/max distance ratio (i.e., maximize the ratio)"""
        points = x_flat.reshape(-1, 2)
        
        # Ensure points are within bounds [0,1] x [0,1]
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Filter out very small distances (numerical precision issues)
        # Use a more stringent filter to avoid numerical artifacts
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return float('inf')
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero - return large penalty if no valid distances
        if d_max <= 1e-12:
            return float('inf')
            
        # Return negative ratio to minimize (maximize the ratio)
        # Add small epsilon to avoid numerical issues with very close points
        # Use a more stable computation
        return -d_min / (d_max + 1e-15)
    
    # Strategy 3: Multi-start optimization with more aggressive approach - from INSPIRATION 1
    best_points = None
    best_ratio = 0
    
    initial_configs = generate_initial_configurations()
    
    # Track execution time to ensure we don't exceed time budget
    start_time = time.time()
    
    # Try each configuration with multiple optimization approaches - from INSPIRATION 1
    for config_name, initial_config in initial_configs:
        # Check time limit
        if time.time() - start_time > 55:  # Leave some margin for final refinement
            break
            
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
        
        # First, try local optimization with SLSQP with high tolerance - from INSPIRATION 1
        try:
            result = minimize(
                objective,
                initial_config.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-19, 'gtol': 1e-19},  # Higher tolerance from INSPIRATION 1
                tol=1e-19
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Evaluate the result
                distances = pdist(optimized_points)
                distances = distances[distances > 1e-12]
                
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
        except Exception:
            pass
        
        # If we're still not doing well, try differential evolution with even more aggressive settings
        if best_ratio < 0.095:  # Even lower threshold for DE to be more aggressive - from INSPIRATION 1
            try:
                # Use differential evolution for global optimization with better parameters
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=1500,  # More iterations from INSPIRATION 1
                    popsize=80,   # Larger population for better exploration - from INSPIRATION 1
                    mutation=(0.995, 1),  # Even higher mutation rate - from INSPIRATION 1
                    recombination=0.998,  # Even higher recombination - from INSPIRATION 1
                    seed=42,
                    atol=1e-19,
                    rtol=1e-19
                )
                
                if de_result.success:
                    de_points = de_result.x.reshape(-1, 2)
                    de_points = np.clip(de_points, 0, 1)
                    
                    # Evaluate the result
                    distances = pdist(de_points)
                    distances = distances[distances > 1e-12]
                    
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-12:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = de_points.copy()
                                
            except Exception:
                pass
    
    # Strategy 4: Additional refinement with multiple methods if needed - from INSPIRATION 1
    if best_points is not None:
        # Try additional refinement with different optimization methods
        np.random.seed(42)
        
        # Try L-BFGS-B with even higher precision and more iterations - from INSPIRATION 1
        try:
            bounds = [(0, 1) for _ in range(32)]
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1200, 'ftol': 1e-19, 'gtol': 1e-19}  # More iterations from INSPIRATION 1
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                
                # Evaluate the refined result
                distances = pdist(refined_points)
                distances = distances[distances > 1e-12]
                
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-12:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
        except Exception:
            pass
        
        # Try a few more local optimizations with different starting points - from INSPIRATION 1
        for _ in range(15):  # More attempts from INSPIRATION 1
            # Check time limit
            if time.time() - start_time > 55:  # Leave some margin for final refinement
                break
                
            # Add small random perturbations to current best
            test_points = best_points + np.random.normal(0, 0.001, best_points.shape)
            test_points = np.clip(test_points, 0, 1)
            
            try:
                result = minimize(
                    objective,
                    test_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-18, 'gtol': 1e-18}  # Higher tolerance from INSPIRATION 1
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 2)
                    refined_points = np.clip(refined_points, 0, 1)
                    
                    # Evaluate the refined result
                    distances = pdist(refined_points)
                    distances = distances[distances > 1e-12]
                    
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-12:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = refined_points.copy()
            except Exception:
                continue
    
    # Strategy 5: Physics-inspired refinement approach - from INSPIRATION 1
    if best_points is not None:
        # Apply physics-based refinement with more iterations and better force calculation - from INSPIRATION 1
        try:
            # Run a more intensive physics simulation
            physics_points = best_points.copy()
            for iteration in range(1500):  # More iterations from INSPIRATION 1
                # Check time limit
                if time.time() - start_time > 55:  # Leave some margin for final refinement
                    break
                    
                dist_matrix = distance_matrix(physics_points, physics_points)
                # Avoid division by zero
                dist_matrix = np.maximum(dist_matrix, 1e-8)
                # Calculate forces (inverse square law)
                forces = np.zeros_like(physics_points)
                for i in range(16):
                    for j in range(16):
                        if i != j:
                            diff = physics_points[i] - physics_points[j]
                            dist = np.linalg.norm(diff)
                            if dist > 1e-8:
                                force_magnitude = 1.0 / (dist * dist)
                                forces[i] += force_magnitude * diff / dist
                
                # Update positions with damping and adaptive step size - from INSPIRATION 1
                step_size = 0.001 * (1 - iteration/2000)  # Decreasing step size
                physics_points += step_size * forces
                # Keep within bounds
                physics_points = np.clip(physics_points, 0, 1)
                
                # Early stopping if improvement is minimal - from INSPIRATION 1
                if iteration > 500 and iteration % 100 == 0:
                    distances = pdist(physics_points)
                    distances = distances[distances > 1e-12]
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-12:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = physics_points.copy()
            
            # Final evaluation
            distances = pdist(physics_points)
            distances = distances[distances > 1e-12]
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = physics_points.copy()
        except Exception:
            pass
    
    # Strategy 6: If no optimization succeeded, return the best initial configuration
    if best_points is None:
        # Try a more sophisticated physics-based initialization first - from INSPIRATION 1
        np.random.seed(42)
        points = np.random.rand(16, 2)
        # Run more iterations of physics simulation for better stabilization - from INSPIRATION 1
        for _ in range(1200):  # More iterations from INSPIRATION 1
            # Check time limit
            if time.time() - start_time > 55:  # Leave some margin for final refinement
                break
                
            dist_matrix = distance_matrix(points, points)
            # Avoid division by zero
            dist_matrix = np.maximum(dist_matrix, 1e-8)
            # Calculate forces (inverse square law)
            forces = np.zeros_like(points)
            for i in range(16):
                for j in range(16):
                    if i != j:
                        diff = points[i] - points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 1e-8:
                            force_magnitude = 1.0 / (dist * dist)
                            forces[i] += force_magnitude * diff / dist
            
            # Update positions with small step size
            step_size = 0.001
            points += step_size * forces
            # Keep within bounds
            points = np.clip(points, 0, 1)
        return points
    
    return best_points


# EVOLVE-BLOCK-END
