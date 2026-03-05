# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import math
import random
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an enhanced hybrid approach combining geometric initialization with advanced optimization
    techniques inspired by the best performing methods.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def create_concentric_circles_initialization():
        """Create high-quality initial configuration using concentric circles"""
        points = []
        
        # Concentric circles with offset angles - optimized from best inspirations
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        # Inner circle with better radius
        radius_inner = 0.32
        for i in range(8):
            angle = angles[i]
            x = 0.5 + radius_inner * np.cos(angle)
            y = 0.5 + radius_inner * np.sin(angle)
            points.append([x, y])
        
        # Outer circle (offset to break symmetry) with better radius
        radius_outer = 0.48
        for i in range(8):
            angle = angles[i] + np.pi/8  # Offset by π/8
            x = 0.5 + radius_outer * np.cos(angle)
            y = 0.5 + radius_outer * np.sin(angle)
            points.append([x, y])
        
        # Add some corner points for better coverage
        corner_points = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        for i in range(4):
            if len(points) < 16:
                points.append(corner_points[i % 4])
        
        # Ensure exactly 16 points
        while len(points) < 16:
            points.append([0.5, 0.5])
        
        points = np.array(points[:16])
        
        # Add more substantial random perturbations to break symmetry
        np.random.seed(42)
        points += np.random.normal(0, 0.035, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def create_better_hexagonal_initialization():
        """Create a better hexagonal initialization pattern"""
        points = []
        sqrt3_over_2 = np.sqrt(3) / 2
        
        # Create a hexagonal grid with 4 rows and 4 columns
        for i in range(4):
            for j in range(4):
                if len(points) >= 16:
                    break
                x = j + 0.5 * (i % 2)  # Offset every other row
                y = i * sqrt3_over_2
                points.append([x, y])
        
        points = np.array(points)
        
        # Normalize to fit in [0,1]x[0,1]
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        
        # Scale to fit nicely within unit square
        max_coord = max(points[:, 0].max(), points[:, 1].max())
        if max_coord > 0:
            points /= max_coord
        
        # Center in unit square
        points[:, 0] += (1.0 - points[:, 0].max() + points[:, 0].min()) / 2
        points[:, 1] += (1.0 - points[:, 1].max() + points[:, 1].min()) / 2
        
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.015, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def create_perturbed_circle_initialization():
        """Create points on a circle with random perturbations"""
        # Points on a circle with better radius and spacing
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.42
        points = np.column_stack([
            0.5 + radius * np.cos(angles),
            0.5 + radius * np.sin(angles)
        ])
        
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.025, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def create_grid_initialization():
        """Create a regular grid pattern"""
        points = []
        for i in range(4):
            for j in range(4):
                points.append([j * 0.25 + 0.125, i * 0.25 + 0.125])
        return np.array(points)
    
    def create_random_initialization():
        """Simple random initialization."""
        return np.random.rand(16, 2)
    
    def create_corner_initialization():
        """Initialize points using strategic corner and center arrangements."""
        # Corner points plus center points for better distribution
        points = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
            [0.5, 0.5], [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
            [0.15, 0.5], [0.85, 0.5], [0.5, 0.15], [0.5, 0.85]
        ])
        points = np.clip(points, 0, 1)
        points = points[:16]  # Ensure exactly 16 points
        
        # Add small random perturbations
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        
        return points
    
    def enhanced_simulated_annealing(initial_points, max_iter=70000):
        """Enhanced simulated annealing with better cooling schedule"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Even more aggressive cooling schedule - optimized from inspirations
        T = 1.0
        T_min = 1e-16
        alpha = 0.998  # Even more aggressive cooling
        iter_without_improvement = 0
        max_no_improve = 1000  # Early stopping condition (more aggressive)
        
        for i in range(max_iter):
            # Perturb one point randomly
            idx = random.randint(0, 15)
            new_points = current_points.copy()
            
            # Adaptive perturbation with dynamic temperature
            temp_factor = max(0.0015, T * 0.8)
            delta = np.random.normal(0, temp_factor, 2)
            new_points[idx] = current_points[idx] + delta
            
            # Keep within bounds
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            # Calculate new ratio
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or random.random() < math.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio
                iter_without_improvement = 0
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            else:
                iter_without_improvement += 1
            
            # Cool down more aggressively
            T = max(T * alpha, T_min)
            
            # Early stopping if no improvement for too long
            if iter_without_improvement > max_no_improve:
                break
            
        return best_points
    
    def improved_local_refinement(points, max_iter=3000):
        """Improved local refinement with better search strategy"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Strategy: Enhanced gradient-free hill climbing with more aggressive search
        for iteration in range(max_iter // 10):
            improved = False
            
            # Try moving each point to improve the ratio
            for i in range(16):
                best_point = current_points[i].copy()
                best_ratio = current_ratio
                
                # Try several moves with different step sizes - increased attempts
                for _ in range(70):  # More attempts per point
                    # Different step sizes for exploration
                    step_size = 0.004 if iteration < max_iter // 30 else 0.001
                    move = np.random.normal(0, step_size, 2)
                    new_point = current_points[i] + move
                    new_point = np.clip(new_point, 0, 1)
                    
                    # Test this move
                    test_points = current_points.copy()
                    test_points[i] = new_point
                    
                    new_ratio = compute_min_max_ratio(test_points)
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_point = new_point.copy()
                
                # Apply the best move if it improves the ratio
                if best_ratio > current_ratio:
                    current_points[i] = best_point
                    current_ratio = best_ratio
                    improved = True
            
            # Early stopping if no improvements
            if not improved:
                break
        
        return current_points
    
    def adaptive_local_search(points, max_iter=1000):
        """Adaptive local search that tries different strategies"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Try several different local search approaches with more diversity
        for _ in range(15):  # More attempts
            # Random point perturbation with varying intensities
            test_points = current_points.copy()
            num_moves = random.randint(5, 20)  # More moves
            
            for _ in range(num_moves):
                idx = random.randint(0, 15)
                # Variable step size based on current ratio
                step_size = 0.001 + 0.003 * (current_ratio * 10)  # Larger steps when ratio is low
                move = np.random.normal(0, step_size, 2)
                test_points[idx] = test_points[idx] + move
                test_points[idx] = np.clip(test_points[idx], 0, 1)
            
            new_ratio = compute_min_max_ratio(test_points)
            if new_ratio > current_ratio:
                current_points = test_points
                current_ratio = new_ratio
        
        return current_points
    
    # Generate diverse initial configurations with emphasis on best ones
    initial_configs = [
        create_concentric_circles_initialization(),  # Best approach from inspirations
        create_better_hexagonal_initialization(),
        create_perturbed_circle_initialization(), 
        create_grid_initialization(),
        create_random_initialization(),
        create_corner_initialization()
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Try multiple strategies with different restarts - reduced but more focused
    for strategy_idx, initial_config in enumerate(initial_configs):
        try:
            # Strategy 1: Enhanced Simulated Annealing (even longer run)
            sa_points = enhanced_simulated_annealing(initial_config, 70000)
            sa_ratio = compute_min_max_ratio(sa_points)
            
            # Strategy 2: Local refinement after SA with better refinement
            refined_points = improved_local_refinement(sa_points, 3000)
            refined_ratio = compute_min_max_ratio(refined_points)
            
            # Strategy 3: Adaptive local search
            adaptive_points = adaptive_local_search(sa_points, 1000)
            adaptive_ratio = compute_min_max_ratio(adaptive_points)
            
            # Strategy 4: Direct optimization from initial config
            direct_points = initial_config.copy()
            direct_ratio = compute_min_max_ratio(direct_points)
            
            # Select the best of these four approaches for this initial config
            candidates = [
                (sa_points, sa_ratio),
                (refined_points, refined_ratio),
                (adaptive_points, adaptive_ratio),
                (direct_points, direct_ratio)
            ]
            
            # Pick the best among the four
            best_candidate = max(candidates, key=lambda x: x[1])
            
            if best_candidate[1] > best_ratio:
                best_ratio = best_candidate[1]
                best_points = best_candidate[0].copy()
                
        except Exception as e:
            # Continue with next initial config if this one fails
            continue
    
    # Final refinement step with more aggressive optimization
    if best_points is not None:
        # Multiple rounds of refinement to polish the solution
        refined_points = improved_local_refinement(best_points, 2000)
        refined_ratio = compute_min_max_ratio(refined_points)
        
        if refined_ratio > best_ratio:
            best_points = refined_points
            best_ratio = refined_ratio
        
        # One final adaptive search
        final_points = adaptive_local_search(best_points, 1000)
        final_ratio = compute_min_max_ratio(final_points)
        
        if final_ratio > best_ratio:
            best_points = final_points
    
    # If still no good solution, return the best among initial configurations
    if best_points is None:
        # Return the best among initial configurations
        best_initial_ratio = -np.inf
        for config in initial_configs:
            ratio = compute_min_max_ratio(config)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_points = config.copy()
    
    return best_points


# EVOLVE-BLOCK-END
