# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach combining mathematical initialization and efficient optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given point configuration"""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    # Generate high-quality initial configurations
    def generate_initial_configs():
        configs = []
        
        # Config 1: Hexagonal lattice (efficient packing)
        hex_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25 * np.sqrt(3)/2
                hex_points.append([x, y])
        configs.append(np.array(hex_points[:16]))
        
        # Config 2: Circle arrangement with internal symmetry
        circle_points = []
        # Place 12 points on outer circle
        for i in range(12):
            angle = i * 2 * np.pi / 12
            circle_points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        
        # Add 4 inner points arranged symmetrically
        inner_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        for angle in inner_angles:
            circle_points.append([0.5 + 0.15 * np.cos(angle), 0.5 + 0.15 * np.sin(angle)])
        
        configs.append(np.array(circle_points[:16]))
        
        # Config 3: Golden spiral distribution
        golden_points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            theta = np.arccos(-1 + (2 * i) / 15)
            phi_val = np.sqrt(i / 15) * 2 * np.pi
            x = np.sin(theta) * np.cos(phi_val)
            y = np.sin(theta) * np.sin(phi_val)
            golden_points.append([(x + 1) / 2, (y + 1) / 2])
        configs.append(np.array(golden_points))
        
        # Config 4: Optimized grid with symmetry breaking
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125 + 0.005 * np.sin(i * np.pi/2 + j * np.pi/3)
                y = i * 0.25 + 0.125 + 0.005 * np.cos(j * np.pi/2 + i * np.pi/3)
                grid_points.append([x, y])
        configs.append(np.array(grid_points[:16]))
        
        # Config 5: Random with fixed seed
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        configs.append(random_points)
        
        return configs
    
    # Robust optimization with multiple strategies
    def robust_optimization(initial_points, max_iter=600):
        """Optimize using multiple strategies for maximum robustness"""
        
        def objective(x):
            # Reshape x into points
            points = x.reshape(-1, 2)
            
            # Compute pairwise distances
            distances = pdist(points)
            
            # Minimize negative of ratio (equivalent to maximizing ratio)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 0  # Avoid division by zero
                
            return -min_dist / max_dist
        
        # Set bounds for optimization (0 to 1 in both dimensions)
        bounds = [(0, 1) for _ in range(32)]
        
        best_ratio = -np.inf
        best_solution = None
        
        # Try multiple optimization methods with different seeds
        strategies = [
            ('trust-constr', [42, 123, 456]),
            ('L-BFGS-B', [42, 123, 456]),
            ('SLSQP', [42, 123])
        ]
        
        for method, seeds in strategies:
            for seed in seeds:
                try:
                    # Start with the initial configuration plus small perturbation
                    np.random.seed(seed)
                    x0 = initial_points.flatten()
                    # Add small random perturbation
                    x0 += np.random.normal(0, 0.005, x0.shape)
                    x0 = np.clip(x0, 0, 1)
                    
                    # Optimize
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9}
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        ratio = compute_min_max_ratio(final_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_solution = result.x.copy()
                            
                except Exception:
                    continue
        
        # Return best solution found or original if none
        if best_solution is not None:
            return best_solution.reshape(-1, 2)
        else:
            return initial_points
    
    # Enhanced refinement with better convergence criteria
    def enhanced_refinement(initial_points, max_iter=100):
        """Enhanced refinement with better convergence detection"""
        points = initial_points.copy()
        best_ratio = compute_min_max_ratio(points)
        best_points = points.copy()
        
        # Track improvement for early stopping
        last_improvement = 0
        improvement_threshold = 1e-8
        
        for iteration in range(max_iter):
            improved = False
            total_improvement = 0
            
            # Try small perturbations to each point
            for i in range(len(points)):
                current_point = points[i].copy()
                best_point = current_point.copy()
                best_local_ratio = best_ratio
                
                # Try several small perturbations
                for _ in range(2):
                    # Small random perturbation
                    delta = np.random.normal(0, 0.002, 2)
                    new_point = current_point + delta
                    # Keep within bounds
                    new_point = np.clip(new_point, 0, 1)
                    
                    # Test this change
                    test_points = points.copy()
                    test_points[i] = new_point
                    
                    ratio = compute_min_max_ratio(test_points)
                    if ratio > best_local_ratio:
                        best_local_ratio = ratio
                        best_point = new_point
                        improved = True
                        total_improvement += (ratio - best_local_ratio)
                
                # Update if improvement was found
                if improved:
                    points[i] = best_point
                    if best_local_ratio > best_ratio:
                        best_ratio = best_local_ratio
                        best_points = points.copy()
            
            # Early stopping if improvement is negligible
            if total_improvement < improvement_threshold:
                last_improvement += 1
                if last_improvement > 10:
                    break
            else:
                last_improvement = 0
                
        return best_points
    
    # Main optimization process with systematic approach
    best_points = None
    best_ratio = 0
    
    # Generate multiple configurations
    configs = generate_initial_configs()
    
    # Try each configuration with optimization
    for i, config in enumerate(configs):
        try:
            # Apply robust optimization
            optimized_points = robust_optimization(config, max_iter=600)
            
            # Apply enhanced refinement
            refined_points = enhanced_refinement(optimized_points, max_iter=80)
            
            # Check quality
            ratio = compute_min_max_ratio(refined_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
                
        except Exception as e:
            continue
    
    # If no good solution was found, use a strong fallback
    if best_points is None:
        # Try the hexagonal configuration with more intensive optimization
        hex_config = generate_initial_configs()[0]
        try:
            best_points = robust_optimization(hex_config, max_iter=800)
        except Exception:
            # Final fallback: simple grid
            points = []
            for i in range(4):
                for j in range(4):
                    x = j * 0.25 + 0.125
                    y = i * 0.25 + 0.125
                    points.append([x, y])
            best_points = np.array(points)
    
    # Final verification and minor optimization
    try:
        # Try one final round with slightly different parameters
        final_points = robust_optimization(best_points, max_iter=300)
        final_ratio = compute_min_max_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
