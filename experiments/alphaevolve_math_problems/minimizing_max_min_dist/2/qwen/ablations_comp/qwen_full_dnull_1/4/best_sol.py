# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.spatial.distance import pdist
import warnings
import math
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach inspired by the best-performing methods from multiple inspirations.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances efficiently
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Handle edge case where all points are coincident
        if max_dist <= 1e-12:
            return 0.0
            
        return min_dist / max_dist
    
    def create_concentric_circles_initialization():
        """Create high-quality initial configuration using concentric circles"""
        points = []
        
        # Concentric circles with offset angles
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        # Inner circle
        radius_inner = 0.35
        for i in range(8):
            angle = angles[i]
            x = 0.5 + radius_inner * np.cos(angle)
            y = 0.5 + radius_inner * np.sin(angle)
            points.append([x, y])
        
        # Outer circle (offset to break symmetry)
        radius_outer = 0.45
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
        points += np.random.normal(0, 0.03, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_hexagonal_grid():
        """Generate points in a hexagonal grid pattern"""
        points = []
        sqrt3_over_2 = math.sqrt(3) / 2
        
        # Create a hexagonal grid with 4 rows and 4 columns
        for i in range(4):
            for j in range(4):
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
        points += np.random.normal(0, 0.015, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_perturbed_circle():
        """Generate points on a circle with random perturbations"""
        # Points on a circle
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([
            0.5 + radius * np.cos(angles),
            0.5 + radius * np.sin(angles)
        ])
        
        # Add small random perturbations
        points += np.random.normal(0, 0.02, (16, 2))
        points = np.clip(points, 0, 1)
        return points
    
    def generate_spiral_distribution():
        """Generate points using a spiral pattern"""
        points = []
        # Golden angle spiral
        golden_angle = math.pi * (3 - math.sqrt(5))
        
        for i in range(16):
            radius = 0.4 * math.sqrt(i / 15.0)  # Spiral radius
            angle = i * golden_angle
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        return np.array(points)
    
    def generate_corner_distribution():
        """Generate points in corners and center for better spread"""
        points = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
            [0.5, 0.5], [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
            [0.15, 0.5], [0.85, 0.5], [0.5, 0.15], [0.5, 0.85]
        ])
        points = np.clip(points, 0, 1)
        return points[:16]  # Ensure exactly 16 points
    
    def enhanced_simulated_annealing(initial_points, max_iter=100000):
        """Enhanced simulated annealing with better cooling schedule"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Enhanced cooling schedule from top performers - even more aggressive
        T = 1.0
        T_min = 1e-15
        alpha = 0.998  # Even more aggressive cooling - inspired by INSPIRATION 2
        iter_without_improvement = 0
        max_no_improve = 1000  # Early stopping condition
        
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
    
    def local_gradient_free_refinement(points, max_iter=4000):
        """Enhanced local refinement with better search strategy"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Strategy: Gradient-free hill climbing with more thorough exploration
        for iteration in range(max_iter // 10):
            improved = False
            
            # Try moving each point to improve the ratio
            for i in range(16):
                best_point = current_points[i].copy()
                best_ratio = current_ratio
                
                # Try several moves with different step sizes for better exploration
                step_sizes = [0.004, 0.003, 0.002, 0.001]  # More aggressive steps
                for step_size in step_sizes:
                    for _ in range(25):  # More attempts per step size
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
    
    def multi_strategy_optimization():
        """Run multiple strategies and return the best result"""
        best_points = None
        best_ratio = -np.inf
        
        # Generate diverse initial configurations
        initial_configs = [
            create_concentric_circles_initialization(),  # Best approach from inspirations
            generate_hexagonal_grid(),
            generate_perturbed_circle(), 
            generate_spiral_distribution(),  # Add spiral for variety
            generate_corner_distribution()   # Add corner distribution for spread
        ]
        
        # Try multiple strategies with different restarts
        for strategy_idx, initial_config in enumerate(initial_configs):
            try:
                # Strategy 1: Enhanced Simulated Annealing (even longer run)
                sa_points = enhanced_simulated_annealing(initial_config, 100000)
                sa_ratio = compute_min_max_ratio(sa_points)
                
                # Strategy 2: Local refinement after SA
                refined_points = local_gradient_free_refinement(sa_points, 4000)
                refined_ratio = compute_min_max_ratio(refined_points)
                
                # Strategy 3: Direct optimization from initial config
                direct_points = initial_config.copy()
                direct_ratio = compute_min_max_ratio(direct_points)
                
                # Select the best of these three approaches for this initial config
                candidates = [
                    (sa_points, sa_ratio),
                    (refined_points, refined_ratio),
                    (direct_points, direct_ratio)
                ]
                
                # Pick the best among the three
                best_candidate = max(candidates, key=lambda x: x[1])
                
                if best_candidate[1] > best_ratio:
                    best_ratio = best_candidate[1]
                    best_points = best_candidate[0].copy()
                    
            except Exception as e:
                continue
        
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
    
    # Main optimization procedure - use the most successful multi-strategy approach
    try:
        # Run multi-strategy optimization with the best techniques from inspirations
        final_points = multi_strategy_optimization()
        
        # Final refinement step with even more iterations
        final_points = local_gradient_free_refinement(final_points, 2000)
        final_ratio = compute_min_max_ratio(final_points)
        
        return final_points
        
    except Exception:
        # Fallback to robust geometric approach
        try:
            # Try concentric circles approach (best from inspirations)
            fallback_points = create_concentric_circles_initialization()
            return fallback_points
        except Exception:
            # Last resort: random points
            return np.random.rand(16, 2)


# EVOLVE-BLOCK-END
