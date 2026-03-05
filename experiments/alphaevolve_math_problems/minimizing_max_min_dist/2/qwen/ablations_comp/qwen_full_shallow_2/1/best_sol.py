# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a combination of geometric constructions, multi-start optimization, and hybrid approaches.
    Includes mathematical programming and energy minimization approaches.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Strategy 1: Generate multiple high-quality initial configurations
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
        
        # Configuration 5: Concentrated around center (better for ratio optimization)
        np.random.seed(46)
        points = []
        for i in range(16):
            # Concentrate points towards center with some randomness
            r = np.random.random() * 0.4
            theta = np.random.random() * 2 * np.pi
            x = 0.5 + r * np.cos(theta) * 0.8
            y = 0.5 + r * np.sin(theta) * 0.8
            points.append([x, y])
        configs.append(("center_concentrated", np.array(points)))
        
        # Configuration 6: Uniform random distribution
        np.random.seed(47)
        points = np.random.rand(16, 2)
        configs.append(("uniform_random", points))
        
        # Configuration 7: Circular pattern (inspiration from program 3)
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
        
        # Configuration 8: Concentrated on edges (for variety)
        np.random.seed(50)
        points = []
        for i in range(16):
            # Concentrate points near edges
            edge = np.random.randint(0, 4)  # 0=top, 1=right, 2=bottom, 3=left
            if edge == 0:  # top
                x = np.random.random()
                y = 0.95 + np.random.random() * 0.05
            elif edge == 1:  # right
                x = 0.95 + np.random.random() * 0.05
                y = np.random.random()
            elif edge == 2:  # bottom
                x = np.random.random()
                y = np.random.random() * 0.05
            else:  # left
                x = np.random.random() * 0.05
                y = np.random.random()
            points.append([x, y])
        configs.append(("edge_concentrated", np.array(points)))
        
        # Configuration 9: Concentrated in a cluster (even more aggressive)
        np.random.seed(51)
        points = []
        # Cluster points in center region
        for i in range(16):
            # Very tight clustering near center
            r = np.random.random() * 0.2
            theta = np.random.random() * 2 * np.pi
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            points.append([x, y])
        configs.append(("tight_cluster", np.array(points)))
        
        # Configuration 10: Grid with center point (balanced approach)
        np.random.seed(52)
        points = [[0.5, 0.5]]  # Center point
        # Add points in a grid-like pattern
        for i in range(3):
            for j in range(3):
                x = 0.2 + i * 0.3
                y = 0.2 + j * 0.3
                points.append([x, y])
        # Fill remaining spots with random points near center
        for i in range(16 - len(points)):
            x = 0.5 + np.random.normal(0, 0.05)
            y = 0.5 + np.random.normal(0, 0.05)
            points.append([x, y])
        configs.append(("grid_center", np.clip(np.array(points), 0, 1)[:16]))
        
        # Configuration 11: Energy-based initialization (repulsion model)
        np.random.seed(53)
        # Start with random points
        energy_points = np.random.rand(16, 2)
        # Simple energy minimization heuristic: repel points from each other
        for _ in range(100):
            for i in range(16):
                force = np.zeros(2)
                for j in range(16):
                    if i != j:
                        diff = energy_points[i] - energy_points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 1e-10:  # Avoid division by zero
                            force += diff / (dist**3 + 1e-10)
                # Move point in direction of negative force (repulsion)
                energy_points[i] += 0.001 * force
                # Keep within bounds
                energy_points[i] = np.clip(energy_points[i], 0, 1)
        configs.append(("energy_init", energy_points))
        
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
        distances = distances[distances > 1e-15]
        
        if len(distances) == 0:
            return float('inf')
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero - return large penalty if no valid distances
        if d_max <= 1e-15:
            return float('inf')
            
        # Return negative ratio to minimize (maximize the ratio)
        # Add small epsilon to avoid numerical issues with very close points
        return -d_min / (d_max + 1e-15)
    
    # Strategy 3: Multi-start optimization with both local and global methods
    best_points = None
    best_ratio = 0
    
    initial_configs = generate_initial_configurations()
    
    # Try each configuration with optimization - more aggressive approach
    for config_name, initial_config in initial_configs:
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
        
        # First, try differential evolution for global optimization with aggressive settings
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=700,      # More iterations
                popsize=40,       # Larger population for better exploration
                mutation=(0.95, 1),  # Higher mutation rate for more exploration
                recombination=0.98,  # Higher recombination rate
                seed=42,
                atol=1e-16,       # Tighter absolute tolerance
                rtol=1e-16       # Tighter relative tolerance
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 2)
                de_points = np.clip(de_points, 0, 1)
                
                # Evaluate the result
                distances = pdist(de_points)
                distances = distances[distances > 1e-15]
                
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = de_points.copy()
                            
        except Exception:
            pass
        
        # If still not good enough, try SLSQP as backup with even higher tolerance
        if best_ratio < 0.09 and best_points is None:
            try:
                result = minimize(
                    objective,
                    initial_config.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1200, 'ftol': 1e-16, 'gtol': 1e-16},
                    tol=1e-16
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    
                    # Evaluate the result
                    distances = pdist(optimized_points)
                    distances = distances[distances > 1e-15]
                    
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-15:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                
            except Exception:
                pass
    
    # Strategy 4: Final refinement with local search if needed
    if best_points is not None:
        # Try additional refinement with multiple methods
        np.random.seed(42)
        
        # Try L-BFGS-B with even more aggressive settings
        try:
            bounds = [(0, 1) for _ in range(32)]
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                
                # Evaluate the refined result
                distances = pdist(refined_points)
                distances = distances[distances > 1e-15]
                
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = refined_points.copy()
        except Exception:
            pass
        
        # Try several more local optimizations with different starting points
        for _ in range(3):  # More attempts
            # Add small random perturbations to current best
            test_points = best_points + np.random.normal(0, 0.0005, best_points.shape)
            test_points = np.clip(test_points, 0, 1)
            
            try:
                result = minimize(
                    objective,
                    test_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 2)
                    refined_points = np.clip(refined_points, 0, 1)
                    
                    # Evaluate the refined result
                    distances = pdist(refined_points)
                    distances = distances[distances > 1e-15]
                    
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-15:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = refined_points.copy()
            except Exception:
                continue
    
    # Strategy 5: If no optimization succeeded, return the best initial configuration
    if best_points is None:
        # Return the golden spiral configuration as final fallback
        np.random.seed(42)
        points = []
        phi = (1 + math.sqrt(5)) / 2
        for i in range(16):
            angle = i * 2 * math.pi / phi
            radius = math.sqrt(i) / math.sqrt(15) if i > 0 else 0.5
            x = 0.5 + radius * math.cos(angle) * 0.45
            y = 0.5 + radius * math.sin(angle) * 0.45
            points.append([x, y])
        return np.array(points)
    
    return best_points


# EVOLVE-BLOCK-END
