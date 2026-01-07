# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Use a more sophisticated initialization based on inspiration 2 with improved bounds
    def generate_initial_config():
        """Generate a good initial configuration using a hexagonal packing heuristic"""
        circles = np.zeros((n, 3))
        
        # Create a hexagonal packing pattern with better spacing
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Make sure we have enough space
        while rows * cols < n:
            rows += 1
            cols = int(np.ceil(n / rows))
        
        # Create initial positions in a hexagonal lattice with better centering
        spacing_x = 0.9 / cols  # Leave margin for better packing
        spacing_y = 0.9 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x  # Center the pattern
                y = 0.05 + (i + 0.5) * spacing_y
                
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                
                # Adjust to stay within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius based on proximity to edges - more conservative to allow optimization
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                circles[idx] = [x, y, min_dist_to_edge * 0.25]
                idx += 1
        
        # Ensure we have exactly 32 circles
        if idx < n:
            # Fill remaining positions with small random circles but with better positioning
            for i in range(idx, n):
                # Use a more strategic placement for remaining circles
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                circles[i] = [x, y, 0.04]
        
        return circles

    # Optimized constraint functions using vectorization and numerical stability
    def constraint_containment(circles):
        """Constraint function for containment"""
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        # r <= x <= 1-r and r <= y <= 1-r
        # We want: x >= r, 1-x >= r, y >= r, 1-y >= r
        # Which means: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        return np.column_stack([
            x - r,           # x >= r
            1 - x - r,       # x <= 1-r  
            y - r,           # y >= r
            1 - y - r        # y <= 1-r
        ]).flatten()

    def constraint_overlap(circles):
        """Constraint function for non-overlap"""
        n = len(circles)
        cons = []
        
        # Vectorized computation of distances for better performance
        centers = circles[:, :2]
        distances = cdist(centers, centers)
        
        # Only check upper triangle to avoid double counting
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r_i, r_j = circles[i, 2], circles[j, 2]
                # We want dist >= r_i + r_j, so we add the constraint: dist - r_i - r_j >= 0
                # Add small epsilon to handle numerical precision issues
                cons.append(dist - r_i - r_j)
        
        return np.array(cons)

    # Objective function to minimize (negative of sum of radii)
    def objective(circles):
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Set up bounds for optimization
    def setup_bounds():
        bounds = []
        for i in range(n):
            # x coordinate bounds: 0.001 to 0.999 (with small buffer)
            # y coordinate bounds: 0.001 to 0.999 (with small buffer)  
            # r coordinate bounds: 0.001 to 0.499 (max radius to prevent overlap issues)
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        return bounds
    
    # Refine using optimization with better parameters and error handling
    def optimize_circles(circles):
        """Refine the circle configuration using optimization"""
        # Flatten the circles array for optimization
        initial_vars = circles.flatten()
        
        bounds = setup_bounds()
        
        # Define constraints
        def constraint_func(vars):
            # Reconstruct circles from flattened vars
            reconstructed = vars.reshape(-1, 3)
            
            # Check containment constraints
            containment = constraint_containment(reconstructed)
            
            # Check overlap constraints  
            overlap = constraint_overlap(reconstructed)
            
            return np.concatenate([containment, overlap])
        
        # Try multiple optimization attempts with different settings for robustness
        try:
            # First attempt with strict tolerances
            result = minimize(
                lambda x: objective(x.reshape(-1, 3)),
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 2500, 'ftol': 1e-10, 'eps': 1e-8, 'disp': False}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Clip values to ensure they're within bounds
                for i in range(len(optimized_circles)):
                    x, y, r = optimized_circles[i]
                    optimized_circles[i] = [
                        np.clip(x, r, 1-r),
                        np.clip(y, r, 1-r),
                        np.clip(r, 0.001, 0.499)
                    ]
                return optimized_circles
        except Exception as e:
            # If optimization fails, try with moderate tolerances
            try:
                result = minimize(
                    lambda x: objective(x.reshape(-1, 3)),
                    initial_vars,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1500, 'ftol': 1e-8, 'eps': 1e-7, 'disp': False}
                )
                
                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    # Clip values to ensure they're within bounds
                    for i in range(len(optimized_circles)):
                        x, y, r = optimized_circles[i]
                        optimized_circles[i] = [
                            np.clip(x, r, 1-r),
                            np.clip(y, r, 1-r),
                            np.clip(r, 0.001, 0.499)
                        ]
                    return optimized_circles
            except Exception as e2:
                # If everything fails, try with looser tolerances
                try:
                    result = minimize(
                        lambda x: objective(x.reshape(-1, 3)),
                        initial_vars,
                        method='SLSQP',
                        bounds=bounds,
                        constraints={'type': 'ineq', 'fun': constraint_func},
                        options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6, 'disp': False}
                    )
                    
                    if result.success:
                        optimized_circles = result.x.reshape(-1, 3)
                        # Clip values to ensure they're within bounds
                        for i in range(len(optimized_circles)):
                            x, y, r = optimized_circles[i]
                            optimized_circles[i] = [
                                np.clip(x, r, 1-r),
                                np.clip(y, r, 1-r),
                                np.clip(r, 0.001, 0.499)
                            ]
                        return optimized_circles
                except Exception as e3:
                    # If everything fails, return original circles
                    pass
        
        return circles
    
    # Enhanced multi-start optimization with better diversity and smarter restarts
    def enhanced_multistart_optimization(initial_circles):
        """Run multiple optimization runs with different starting points"""
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])
        
        # Try several random restarts with different perturbation schemes
        for restart in range(15):  # More restarts for better exploration
            # Create perturbed version with different intensities and probabilities
            perturbed = initial_circles.copy()
            
            # Different perturbation strategies based on restart number
            if restart < 4:
                # Strong perturbation with high probability
                prob = 0.9
                scale = 0.04
            elif restart < 8:
                # Medium perturbation with medium probability
                prob = 0.7
                scale = 0.025
            else:
                # Weak perturbation with lower probability
                prob = 0.5
                scale = 0.015
            
            # Apply perturbations to each circle
            for i in range(n):
                if np.random.rand() < prob:
                    perturbed[i, 0] += np.random.normal(0, scale)
                    perturbed[i, 1] += np.random.normal(0, scale)
                    perturbed[i, 2] += np.random.normal(0, scale * 0.5)
                    
                    # Keep within bounds
                    r = perturbed[i, 2]
                    perturbed[i, 0] = np.clip(perturbed[i, 0], r, 1-r)
                    perturbed[i, 1] = np.clip(perturbed[i, 1], r, 1-r)
                    perturbed[i, 2] = np.clip(r, 0.001, 0.499)
            
            # Optimize the perturbed version
            optimized = optimize_circles(perturbed)
            current_sum = np.sum(optimized[:, 2])
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized
        
        return best_circles
    
    # Additional aggressive local search for final refinement
    def aggressive_local_search(circles):
        """Perform an aggressive local search to fine-tune the solution"""
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Multiple rounds of aggressive perturbation and optimization
        for round_num in range(3):  # Fewer rounds for efficiency
            # Even more aggressive perturbations in early rounds
            if round_num < 2:
                scale = 0.03
                prob = 0.85
            else:
                scale = 0.015
                prob = 0.6
            
            # Many iterations in early rounds
            iterations = 50 if round_num < 2 else 30
            
            for _ in range(iterations):
                # Create perturbed version
                perturbed = best_circles.copy()
                
                # Perturb with higher probability in early rounds
                for i in range(n):
                    if np.random.rand() < prob:
                        perturbed[i, 0] += np.random.normal(0, scale)
                        perturbed[i, 1] += np.random.normal(0, scale)
                        perturbed[i, 2] += np.random.normal(0, scale * 0.5)
                        
                        # Keep within bounds
                        r = perturbed[i, 2]
                        perturbed[i, 0] = np.clip(perturbed[i, 0], r, 1-r)
                        perturbed[i, 1] = np.clip(perturbed[i, 1], r, 1-r)
                        perturbed[i, 2] = np.clip(r, 0.001, 0.499)
                
                # Optimize the perturbed version
                optimized = optimize_circles(perturbed)
                current_sum = np.sum(optimized[:, 2])
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized
        
        return best_circles
    
    # Generate initial configuration
    circles = generate_initial_config()
    
    # Run enhanced multistart optimization
    best_circles = enhanced_multistart_optimization(circles)
    
    # Apply aggressive local search for fine-tuning
    best_circles = aggressive_local_search(best_circles)
    
    # Final refinement step
    final_circles = optimize_circles(best_circles)
    final_sum = np.sum(final_circles[:, 2])
    
    # If we got a better solution from final optimization, use it
    if final_sum > np.sum(best_circles[:, 2]):
        best_circles = final_circles
    
    return best_circles


# EVOLVE-BLOCK-END
