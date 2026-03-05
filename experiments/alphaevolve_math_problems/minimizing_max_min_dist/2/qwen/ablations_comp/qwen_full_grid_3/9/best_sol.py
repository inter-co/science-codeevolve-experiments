# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric insights with global optimization and local refinement.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return 0
            
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to maximize (negative because scipy minimizes)."""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize
        return -ratio
    
    # Strategy 1: Better geometric initialization using Fibonacci-like spiral
    def generate_fibonacci_initial_guess():
        """Generate initial configuration using Fibonacci-like spiral for better distribution."""
        points = []
        
        # Use Fibonacci spiral pattern for even better point distribution
        # Inspired by optimal sphere packing and golden ratio properties
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(16):
            # Map index to spiral coordinates with better spacing
            theta = i * 2.4  # Golden angle approximation
            r = np.sqrt(i / 15.0) if i > 0 else 0
            
            # Convert to Cartesian coordinates
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            
            points.append([x, y])
        
        # Convert to numpy array
        points = np.array(points)
        
        # Ensure all points are within [0,1] x [0,1]
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        # Add small random perturbations to escape local optima
        np.random.seed(42)
        perturbations = np.random.uniform(-0.005, 0.005, (16, 2))
        points += perturbations
        
        # Ensure all points are still within bounds after perturbation
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    # Strategy 2: Enhanced optimization with better parameters from inspirations
    def optimize_with_enhanced_de(initial_points):
        """Use differential evolution with optimized parameters."""
        # Flatten initial points
        x0 = initial_points.flatten()
        
        # Define bounds for all coordinates [0,1]
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Use differential evolution with enhanced settings from inspiration 1
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42,
                maxiter=120,   # More iterations for better convergence
                popsize=25,    # Larger population for better exploration
                mutation=(0.5, 1),
                recombination=0.8,
                tol=1e-8,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                # Ensure all points are within bounds
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        
        # Return initial points if optimization fails
        return initial_points
    
    # Strategy 3: Enhanced local refinement with adaptive approach
    def enhanced_local_improvement(points):
        """Enhanced local improvement with adaptive search strategy."""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Try more rounds of improving individual points with adaptive steps
        for round_num in range(100):  # More iterations for better improvement
            improved = False
            for i in range(16):
                # Save current point
                original_point = current_points[i].copy()
                
                # Try moving this point to improve the minimum distance
                best_move = original_point.copy()
                best_ratio = current_ratio
                
                # Adaptive step size based on iteration progress
                if round_num < 50:
                    step_size = 0.01
                else:
                    step_size = 0.005
                
                # Try several random moves with adaptive step size
                for _ in range(25):  # More attempts per point for better search
                    # Generate random perturbation with adaptive step size
                    delta = np.random.uniform(-step_size, step_size, 2)
                    new_point = original_point + delta
                    
                    # Keep within bounds
                    new_point = np.clip(new_point, 0, 1)
                    
                    # Test if this move improves the ratio
                    test_points = current_points.copy()
                    test_points[i] = new_point
                    
                    new_ratio = compute_min_max_ratio(test_points)
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_move = new_point.copy()
                
                # If we found an improvement, apply it
                if not np.array_equal(best_move, original_point):
                    current_points[i] = best_move
                    improved = True
            
            # If no improvement found, stop early
            if not improved:
                break
        
        return current_points
    
    # Strategy 4: Multi-start approach with better selection criteria
    def multi_start_optimization():
        """Try multiple initial configurations and pick the best result."""
        best_points = None
        best_ratio = 0
        
        # Generate better initial guesses
        initial_guesses = []
        
        # Fibonacci spiral (best geometric initialization)
        initial_guesses.append(generate_fibonacci_initial_guess())
        
        # Hexagonal grid with perturbations
        points1 = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points1.append([x, y])
        points1 = np.array(points1)
        np.random.seed(123)
        perturbations = np.random.uniform(-0.01, 0.01, (16, 2))
        points1 += perturbations
        points1[:, 0] = np.clip(points1[:, 0], 0, 1)
        points1[:, 1] = np.clip(points1[:, 1], 0, 1)
        initial_guesses.append(points1)
        
        # Random uniform distribution
        np.random.seed(456)
        points2 = np.random.uniform(0.05, 0.95, (16, 2))
        initial_guesses.append(points2)
        
        for i, initial_points in enumerate(initial_guesses):
            try:
                # Global optimization
                optimized_points = optimize_with_enhanced_de(initial_points)
                
                # Enhanced local improvement
                refined_points = enhanced_local_improvement(optimized_points)
                
                # Evaluate quality
                ratio = compute_min_max_ratio(refined_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
                    
            except Exception:
                continue
        
        # If no success, return the first guess
        if best_points is not None:
            return best_points
        else:
            # Fallback to first initial guess
            return initial_guesses[0]
    
    try:
        # Use multi-start approach for better results
        final_points = multi_start_optimization()
        return final_points
        
    except Exception as e:
        # Fallback to simple hexagonal arrangement if anything goes wrong
        points_hex = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points_hex.append([x, y])
        points_hex = np.array(points_hex)
        points_hex[:, 0] = np.clip(points_hex[:, 0], 0, 1)
        points_hex[:, 1] = np.clip(points_hex[:, 1], 0, 1)
        return points_hex


# EVOLVE-BLOCK-END
