# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations
from numba import jit
import warnings
from scipy.spatial import ConvexHull
import random


@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using Numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distances[idx] = np.sqrt(dx * dx + dy * dy)
            idx += 1
    return distances


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid optimization approach with geometric insights.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of the ratio we want to maximize)"""
        points = x_flat.reshape(-1, 2)
        
        # Use optimized distance computation
        try:
            distances = compute_distances_jit(points)
        except:
            # Fallback to standard method
            distances = pdist(points)
        
        # Avoid division by zero and handle edge cases
        if len(distances) == 0 or np.allclose(distances, 0):
            return 0
            
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Handle edge case where all points are the same
        if max_dist <= 1e-12:
            return 0
            
        # Return negative ratio since we're minimizing
        return -min_dist / max_dist if max_dist > 0 else 0
    
    def generate_initial_config():
        """Generate high-quality initial configurations based on geometric principles"""
        configs = []
        
        # Configuration 1: Optimized hexagonal lattice (more structured)
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < 16:
                    # Proper hexagonal packing with offset rows
                    offset = 0.25 if i % 2 == 0 else 0.375
                    x = j * 0.25 + offset
                    y = i * 0.25
                    points.append([x, y])
        hex_points = np.array(points[:16])
        # Add moderate noise for better exploration
        hex_points = hex_points + (np.random.random((16, 2)) - 0.5) * 0.03
        hex_points = np.clip(hex_points, 0, 1)
        configs.append(hex_points.flatten())
        
        # Configuration 2: Circle with perturbations (better distribution)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.ones(16) * 0.35  # Slightly smaller radius for better spread
        # Add variation to radii
        radii = radii + (np.random.random(16) - 0.5) * 0.1
        circle_points = np.column_stack([
            0.5 + radii * np.cos(angles),
            0.5 + radii * np.sin(angles)
        ])
        configs.append(circle_points.flatten())
        
        # Configuration 3: Improved grid with better spacing
        grid_points = np.array([[i/3.5, j/3.5] for i in range(4) for j in range(4) if i*4+j < 16])
        # Add more controlled perturbations
        perturbed = grid_points + (np.random.random((16, 2)) - 0.5) * 0.02
        perturbed = np.clip(perturbed, 0, 1)
        configs.append(perturbed.flatten())
        
        # Configuration 4: Spiral with golden ratio (more structured)
        golden_angle = 2.399963229728653  # 2π(1 - 1/φ) where φ is golden ratio
        golden_spiral_points = []
        for i in range(16):
            radius = i * 0.035  # Slightly smaller step
            angle = i * golden_angle
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            golden_spiral_points.append([x, y])
        golden_spiral_points = np.array(golden_spiral_points)
        golden_spiral_points = np.clip(golden_spiral_points, 0, 1)
        configs.append(golden_spiral_points.flatten())
        
        # Configuration 5: Square grid with perturbations
        square_points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + 0.125
                y = j * 0.25 + 0.125
                square_points.append([x, y])
        square_points = np.array(square_points[:16])
        square_points = square_points + (np.random.random((16, 2)) - 0.5) * 0.04
        square_points = np.clip(square_points, 0, 1)
        configs.append(square_points.flatten())
        
        # Configuration 6: Pseudo-random with better distribution (using low-discrepancy)
        # Generate points using a low-discrepancy sequence approach
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        points = []
        for i in range(16):
            x = (i / 16) * 0.8 + 0.1
            y = ((i * phi) % 1) * 0.8 + 0.1
            points.append([x, y])
        pseudo_random = np.array(points)
        configs.append(pseudo_random.flatten())
        
        return configs
    
    def local_search_improved(points, max_iter=100):
        """Enhanced local optimization with better gradient estimation and smarter moves"""
        points = points.reshape(-1, 2).copy()
        n = points.shape[0]
        
        # Try different optimization strategies
        for attempt in range(3):  # Multiple attempts
            try:
                # Use L-BFGS-B with bounds and better parameters
                def obj_func(x):
                    return -objective(x)
                
                result = minimize(
                    obj_func,
                    points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    new_points = result.x.reshape(-1, 2)
                    # Ensure points are within bounds
                    new_points = np.clip(new_points, 0, 1)
                    return new_points.flatten()
            except:
                pass
        
        # Fall back to more robust hill climbing with adaptive step sizes
        for iter_count in range(max_iter):
            # Compute current distances
            distances = compute_distances_jit(points)
            if len(distances) == 0 or np.allclose(distances, 0):
                break
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 1e-12:
                break
                
            # Adaptive step size based on current performance
            step_size = max(0.001, min(0.05, 1.0 / (iter_count + 10)))
            
            # For each point, try small moves
            for i in range(n):
                best_move = None
                best_ratio = min_dist / max_dist if max_dist > 0 else 0
                
                # Try several small moves
                for _ in range(20):
                    move = (np.random.random(2) - 0.5) * step_size
                    new_point = points[i] + move
                    
                    # Keep within bounds
                    new_point = np.clip(new_point, 0, 1)
                    
                    # Test this move
                    old_point = points[i].copy()
                    points[i] = new_point
                    
                    # Check if this improves the ratio
                    try:
                        new_distances = compute_distances_jit(points)
                        if len(new_distances) > 0 and not np.allclose(new_distances, 0):
                            new_min = np.min(new_distances)
                            new_max = np.max(new_distances)
                            if new_max > 0 and new_max > 1e-12:
                                new_ratio = new_min / new_max
                                if new_ratio > best_ratio:
                                    best_ratio = new_ratio
                                    best_move = move.copy()
                    except:
                        pass
                    
                    # Restore
                    points[i] = old_point
                
                # Apply best move if found
                if best_move is not None:
                    points[i] = np.clip(points[i] + best_move, 0, 1)
        
        return points.flatten()
    
    def improved_simulated_annealing(start_points, max_iter=500):
        """Improved simulated annealing with better cooling schedule and acceptance criteria"""
        points = start_points.reshape(-1, 2).copy()
        current_points = points.copy()
        
        # Initial evaluation
        current_distances = compute_distances_jit(current_points)
        if len(current_distances) > 0 and not np.allclose(current_distances, 0):
            current_min = np.min(current_distances)
            current_max = np.max(current_distances)
            current_ratio = current_min / current_max if current_max > 0 and current_max > 1e-12 else 0
        else:
            current_ratio = 0
            
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Improved cooling schedule
        temperature = 0.05
        cooling_rate = 0.995
        min_temperature = 1e-10
        
        # Track acceptance rate for adaptive cooling
        accepted_moves = 0
        total_moves = 0
        
        for iteration in range(max_iter):
            # Perturb one point randomly
            idx = np.random.randint(0, 16)
            # Adaptive step size based on temperature
            step_size = temperature * 0.5
            new_point = current_points[idx] + (np.random.random(2) - 0.5) * step_size
            new_point = np.clip(new_point, 0, 1)
            
            # Create new configuration
            new_points = current_points.copy()
            new_points[idx] = new_point
            
            # Evaluate new configuration
            try:
                new_distances = compute_distances_jit(new_points)
                if len(new_distances) > 0 and not np.allclose(new_distances, 0):
                    new_min = np.min(new_distances)
                    new_max = np.max(new_distances)
                    new_ratio = new_min / new_max if new_max > 0 and new_max > 1e-12 else 0
                    
                    # Accept or reject based on Metropolis criterion
                    delta = new_ratio - current_ratio
                    if delta > 0 or np.random.random() < np.exp(delta / temperature):
                        current_points = new_points
                        current_ratio = new_ratio
                        accepted_moves += 1
                        
                        if new_ratio > best_ratio:
                            best_points = new_points.copy()
                            best_ratio = new_ratio
            except:
                pass
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
            total_moves += 1
            
        return best_points.flatten()
    
    # Generate multiple initial configurations
    initial_configs = generate_initial_config()
    
    best_ratio = -float('inf')
    best_points = None
    
    # Try each initial configuration with enhanced local search
    for i, config in enumerate(initial_configs):
        try:
            # Apply enhanced local search to improve this configuration
            improved_config = local_search_improved(config, max_iter=100)
            
            # Evaluate the improved configuration
            points = improved_config.reshape(-1, 2)
            distances = compute_distances_jit(points)
            
            if len(distances) > 0 and not np.allclose(distances, 0):
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0 and max_dist > 1e-12:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
                        
        except Exception as e:
            continue
    
    # If no good result from local searches, use a more systematic approach
    if best_points is None:
        # Use a hybrid approach with multiple restarts
        best_result = None
        best_obj_value = float('inf')
        
        for restart in range(8):  # Reduced restarts for efficiency
            np.random.seed(42 + restart)
            
            # Start with a good configuration
            if restart == 0:
                # Try hexagonal arrangement first
                points = []
                for i in range(4):
                    for j in range(4):
                        x = j * 0.25 + (0.125 if i % 2 == 0 else 0.375)
                        y = i * 0.25
                        points.append([x, y])
                start_points = np.array(points[:16])
            elif restart == 1:
                # Try circle arrangement
                angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                radii = np.ones(16) * 0.35
                start_points = np.column_stack([
                    0.5 + radii * np.cos(angles),
                    0.5 + radii * np.sin(angles)
                ])
            else:
                # Random configuration with better seed control
                start_points = np.random.random((16, 2))
            
            # Apply improved simulated annealing
            try:
                sa_result = improved_simulated_annealing(start_points.flatten(), max_iter=300)
                points = sa_result.reshape(-1, 2)
                distances = compute_distances_jit(points)
                
                if len(distances) > 0 and not np.allclose(distances, 0):
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0 and max_dist > 1e-12:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = points.copy()
            except Exception:
                continue
        
        # If still no good result, return a reasonable default
        if best_points is None:
            # Return a better hexagonal grid configuration
            points = []
            for i in range(4):
                for j in range(4):
                    x = j * 0.25 + (0.125 if i % 2 == 0 else 0.375)
                    y = i * 0.25
                    points.append([x, y])
            best_points = np.array(points[:16])
    
    # Final refinement with local search
    if best_points is not None:
        final_config = local_search_improved(best_points.flatten(), max_iter=50)
        final_points = final_config.reshape(-1, 2)
        distances = compute_distances_jit(final_points)
        if len(distances) > 0 and not np.allclose(distances, 0):
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0 and max_dist > 1e-12:
                final_ratio = min_dist / max_dist
                if final_ratio > best_ratio:
                    best_points = final_points
    
    return best_points


# EVOLVE-BLOCK-END
