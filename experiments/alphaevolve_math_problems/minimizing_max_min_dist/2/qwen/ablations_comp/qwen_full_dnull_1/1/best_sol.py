# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import math
import random
from typing import Tuple
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an enhanced hybrid approach combining geometric initialization, simulated annealing, 
    and robust multi-start optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
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
    
    def objective_function(x_flat):
        """Objective function to minimize (negative of ratio to maximize ratio)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    def create_best_initialization():
        """Create the highest quality initial configuration inspired by best practices"""
        # Strategy: Mix of circular, grid, and corner distributions
        points = []
        
        # 1. Concentric circles (inspired by Program 1)
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
            angle = angles[i] + np.pi/8
            x = 0.5 + radius_outer * np.cos(angle)
            y = 0.5 + radius_outer * np.sin(angle)
            points.append([x, y])
        
        # 2. Add some corner points for better coverage
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
    
    def enhanced_simulated_annealing(initial_points, max_iter=50000):
        """Enhanced simulated annealing with better cooling and stopping criteria"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Enhanced cooling schedule from Program 1
        T = 1.0
        T_min = 1e-12
        alpha = 0.996  # Slightly more aggressive cooling
        iter_without_improvement = 0
        max_no_improve = 2000  # Early stopping condition
        
        for i in range(max_iter):
            # Perturb one point randomly
            idx = random.randint(0, 15)
            new_points = current_points.copy()
            
            # Adaptive perturbation with dynamic temperature
            temp_factor = max(0.003, T * 0.5)
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
    
    def local_refinement_enhanced(points, max_iter=2000):
        """Enhanced local refinement with multiple strategies"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Strategy 1: Gradient-free hill climbing with multiple iterations
        for iteration in range(max_iter // 10):
            improved = False
            
            # Try moving each point to improve the ratio
            for i in range(16):
                best_point = current_points[i].copy()
                best_ratio = current_ratio
                
                # Try several moves with different step sizes
                for _ in range(30):
                    # Different step sizes for exploration
                    step_size = 0.002 if iteration < max_iter // 20 else 0.0005
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
    
    def robust_multi_start_optimization():
        """Run robust multi-start optimization with enhanced strategies"""
        best_points = None
        best_ratio = -np.inf
        
        # Create multiple high-quality initial configurations
        initial_configs = [
            create_best_initialization(),           # Best initialization
            np.random.rand(16, 2),                  # Pure random
            np.array([[i//4 * 0.25 + 0.125, (i%4) * 0.25 + 0.125] for i in range(16)]),  # Grid
        ]
        
        # Also try some geometric patterns
        try:
            # Hexagonal grid
            hex_points = []
            sqrt3_over_2 = math.sqrt(3) / 2
            for i in range(4):
                for j in range(4):
                    x = j + 0.5 * (i % 2)
                    y = i * sqrt3_over_2
                    hex_points.append([x, y])
            hex_points = np.array(hex_points)
            if hex_points[:, 0].max() > hex_points[:, 0].min():
                hex_points[:, 0] = (hex_points[:, 0] - hex_points[:, 0].min()) / (hex_points[:, 0].max() - hex_points[:, 0].min())
            if hex_points[:, 1].max() > hex_points[:, 1].min():
                hex_points[:, 1] = (hex_points[:, 1] - hex_points[:, 1].min()) / (hex_points[:, 1].max() - hex_points[:, 1].min())
            hex_points *= 0.8
            hex_points[:, 0] += 0.1
            hex_points[:, 1] += 0.1
            hex_points += np.random.normal(0, 0.02, (16, 2))
            hex_points = np.clip(hex_points, 0, 1)
            initial_configs.append(hex_points)
        except Exception:
            pass
        
        # Run multiple strategies with different restarts
        for strategy_idx, initial_config in enumerate(initial_configs):
            try:
                # Strategy 1: Enhanced Simulated Annealing (longer run)
                sa_points = enhanced_simulated_annealing(initial_config, 50000)
                sa_ratio = compute_min_max_ratio(sa_points)
                
                # Strategy 2: Local refinement after SA
                refined_points = local_refinement_enhanced(sa_points, 2000)
                refined_ratio = compute_min_max_ratio(refined_points)
                
                # Strategy 3: Multiple Differential Evolution restarts
                de_best_points = None
                de_best_ratio = -np.inf
                for restart in range(3):
                    try:
                        bounds = [(0, 1) for _ in range(32)]
                        de_result = differential_evolution(
                            objective_function,
                            bounds,
                            maxiter=500,
                            popsize=40,
                            seed=42 + restart,
                            disp=False,
                            atol=1e-12,
                            rtol=1e-12
                        )
                        
                        if de_result.success:
                            de_points = de_result.x.reshape(-1, 2)
                            de_points = np.clip(de_points, 0, 1)
                            de_ratio = compute_min_max_ratio(de_points)
                            
                            if de_ratio > de_best_ratio:
                                de_best_ratio = de_ratio
                                de_best_points = de_points.copy()
                    except Exception:
                        continue
                
                # Strategy 4: Multiple SLSQP restarts with better perturbations
                slsqp_best_points = None
                slsqp_best_ratio = -np.inf
                for restart in range(3):
                    try:
                        # Start with a perturbed version
                        if de_best_points is not None and restart == 0:
                            perturbed_points = de_best_points + np.random.normal(0, 0.02, (16, 2))
                        else:
                            perturbed_points = initial_config + np.random.normal(0, 0.04, (16, 2))
                        
                        perturbed_points = np.clip(perturbed_points, 0, 1)
                        
                        # Use SLSQP method with tighter tolerances
                        result = minimize(
                            objective_function,
                            perturbed_points.flatten(),
                            method='SLSQP',
                            bounds=[(0, 1) for _ in range(32)],
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                        )
                        
                        if result.success:
                            slsqp_points = result.x.reshape(-1, 2)
                            slsqp_points = np.clip(slsqp_points, 0, 1)
                            slsqp_ratio = compute_min_max_ratio(slsqp_points)
                            
                            if slsqp_ratio > slsqp_best_ratio:
                                slsqp_best_ratio = slsqp_ratio
                                slsqp_best_points = slsqp_points.copy()
                    except Exception:
                        continue
                
                # Select the best among all strategies for this initial config
                candidates = [
                    (sa_points, sa_ratio),
                    (refined_points, refined_ratio),
                    (de_best_points, de_best_ratio) if de_best_points is not None else None,
                    (slsqp_best_points, slsqp_best_ratio) if slsqp_best_points is not None else None
                ]
                
                # Filter out None values
                valid_candidates = [c for c in candidates if c is not None]
                
                if valid_candidates:
                    best_candidate = max(valid_candidates, key=lambda x: x[1])
                    
                    if best_candidate[1] > best_ratio:
                        best_ratio = best_candidate[1]
                        best_points = best_candidate[0].copy()
                        
            except Exception as e:
                warnings.warn(f"Strategy {strategy_idx} failed: {e}")
                continue
        
        # If still no good solution, return the best initial configuration
        if best_points is None:
            # Return the best among initial configurations
            best_initial_ratio = -np.inf
            for config in initial_configs:
                ratio = compute_min_max_ratio(config)
                if ratio > best_initial_ratio:
                    best_initial_ratio = ratio
                    best_points = config.copy()
        
        return best_points
    
    # Main optimization process
    try:
        # Run robust multi-start optimization
        optimized_points = robust_multi_start_optimization()
        
        # Final local refinement
        final_points = local_refinement_enhanced(optimized_points, 1500)
        
        # Validate and return
        final_ratio = compute_min_max_ratio(final_points)
        if final_ratio <= 0:
            return create_best_initialization()
            
        return final_points
        
    except Exception as e:
        warnings.warn(f"Main optimization failed: {e}")
        # Fallback to best initial configuration
        return create_best_initialization()


# EVOLVE-BLOCK-END
