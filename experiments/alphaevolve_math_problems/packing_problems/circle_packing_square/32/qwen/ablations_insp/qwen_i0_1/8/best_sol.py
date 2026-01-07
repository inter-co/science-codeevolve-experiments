# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Fixed seed for reproducibility
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, gradient-based optimization,
    and constraint-aware refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    def generate_better_initial_placement():
        """Generate initial configuration using a more sophisticated approach based on known dense packings"""
        # Use a systematic approach inspired by known optimal packings for 32 circles
        # Try a 6x6 grid pattern with slight modifications for 32 circles
        
        # Start with a 6x6 grid (36 positions) but remove 4 circles to get 32
        rows, cols = 6, 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Hexagonal packing parameters
        hex_spacing_x = spacing_x * 0.866  # sqrt(3)/2
        hex_spacing_y = spacing_y * 0.75   # 3/4
        
        circles = []
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * hex_spacing_x
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Start with a reasonable initial radius
                r = min(spacing_x / 3, 0.12)
                circles.append([x, y, r])
                count += 1
        
        # Take exactly n circles
        circles = circles[:n]
        
        # Add small random perturbations to break symmetries
        for i in range(len(circles)):
            # Add small random displacement
            circles[i][0] += np.random.normal(0, 0.001) * spacing_x
            circles[i][1] += np.random.normal(0, 0.001) * spacing_y
            # Ensure within bounds
            circles[i][0] = np.clip(circles[i][0], circles[i][2], 1 - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], 1 - circles[i][2])
            
        return np.array(circles)
    
    def compute_constraints_fast(circles):
        """Fast constraint computation using vectorized operations"""
        points = circles[:, :2]
        radii = circles[:, 2]
        
        # Check containment constraints (vectorized)
        containment_mask = (
            (points[:, 0] - radii < 0) | 
            (points[:, 0] + radii > 1) | 
            (points[:, 1] - radii < 0) | 
            (points[:, 1] + radii > 1)
        )
        containment_violations = np.where(containment_mask)[0].tolist()
        
        # Use KDTree for efficient overlap checking
        tree = cKDTree(points)
        # Find all pairs that might be overlapping (using max radius for safety)
        max_radius = np.max(radii) if len(radii) > 0 else 0.1
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
        
        overlap_violations = []
        for i, j in pairs:
            if i < j:  # Avoid duplicate pairs
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                distance = sqrt(dx*dx + dy*dy)
                if distance < radii[i] + radii[j]:
                    overlap_violations.append((i, j))
        
        return containment_violations, overlap_violations
    
    def compute_violation_penalty(circles, penalty_weight=1000.0):
        """Calculate penalty for constraint violations"""
        containment_violations, overlap_violations = compute_constraints_fast(circles)
        
        # Penalty for containment violations (each violation adds significant penalty)
        containment_penalty = len(containment_violations) * penalty_weight * 10000
        
        # Penalty for overlap violations (quadratic penalty for severity)
        overlap_penalty = 0
        for i, j in overlap_violations:
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            distance = sqrt(dx*dx + dy*dy)
            overlap = (circles[i, 2] + circles[j, 2]) - distance
            # Quadratic penalty for overlap severity with stronger penalty
            overlap_penalty += max(0, overlap)**2 * penalty_weight * 100
            
        return containment_penalty + overlap_penalty
    
    def objective_with_penalty(x_flat, circles=None):
        """Objective function with penalty for constraints"""
        # Reshape flat array back to circles
        circles = x_flat.reshape(-1, 3)
        
        # Objective: negative sum of radii (we want to maximize)
        obj_value = -np.sum(circles[:, 2])
        
        # Add penalty for constraint violations
        penalty = compute_violation_penalty(circles)
        
        return obj_value + penalty
    
    def optimize_with_scipy(initial_circles, maxiter=1000):
        """Use scipy optimization for better results with multiple attempts"""
        # Flatten the initial configuration
        x0 = initial_circles.flatten()
        
        # Define bounds for each parameter (x, y, r) for each circle
        bounds = []
        for i in range(n):
            # Bounds for x coordinate
            bounds.append((initial_circles[i, 2], 1 - initial_circles[i, 2]))
            # Bounds for y coordinate  
            bounds.append((initial_circles[i, 2], 1 - initial_circles[i, 2]))
            # Bounds for radius (positive, bounded)
            bounds.append((0.001, 0.25))  # Increased upper bound for better optimization
        
        # Try multiple optimization methods
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        best_result = None
        best_value = float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective_with_penalty,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10},
                    callback=lambda x: None  # No callback needed
                )
                
                if result.success:
                    # Check if this result is better
                    current_value = objective_with_penalty(result.x)
                    if current_value < best_value:
                        best_value = current_value
                        best_result = result
                        
            except Exception as e:
                continue
        
        if best_result is not None:
            optimized_circles = best_result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Fallback to initial configuration if optimization fails
            return initial_circles
    
    def enhanced_local_improvement(circles, max_iter=200):
        """Enhanced local improvement with better constraint handling"""
        refined_circles = circles.copy()
        
        for iteration in range(max_iter):
            improved = False
            
            # Try to increase radii where possible
            for i in range(len(refined_circles)):
                old_radius = refined_circles[i, 2]
                # Try to increase radius more aggressively
                test_radius = min(old_radius + 0.008, 0.25)
                
                # Test if we can increase this radius
                refined_circles[i, 2] = test_radius
                
                # Check constraints
                containment_violations, overlap_violations = compute_constraints_fast(refined_circles)
                if len(containment_violations) == 0 and len(overlap_violations) == 0:
                    improved = True
                else:
                    # Revert if constraints violated
                    refined_circles[i, 2] = old_radius
            
            # Try position improvements with better step size
            for i in range(len(refined_circles)):
                old_x, old_y = refined_circles[i, 0], refined_circles[i, 1]
                old_radius = refined_circles[i, 2]
                
                # Try larger position adjustments
                delta_x = np.random.normal(0, 0.01)
                delta_y = np.random.normal(0, 0.01)
                
                new_x = np.clip(old_x + delta_x, old_radius, 1 - old_radius)
                new_y = np.clip(old_y + delta_y, old_radius, 1 - old_radius)
                
                refined_circles[i, 0] = new_x
                refined_circles[i, 1] = new_y
                
                # Check if this improves the configuration
                containment_violations, overlap_violations = compute_constraints_fast(refined_circles)
                if len(containment_violations) == 0 and len(overlap_violations) == 0:
                    improved = True
                else:
                    # Revert if constraints violated
                    refined_circles[i, 0] = old_x
                    refined_circles[i, 1] = old_y
            
            # Early stopping if no improvement
            if not improved and iteration > 100:
                break
                
        return refined_circles
    
    def adaptive_repulsion_refinement(circles, max_iter=150):
        """Improved physics-inspired refinement with adaptive parameters"""
        refined_circles = circles.copy()
        
        for iteration in range(max_iter):
            points = refined_circles[:, :2]
            radii = refined_circles[:, 2]
            
            # Compute pairwise distances efficiently
            distances = cdist(points, points)
            
            # Compute forces
            forces = np.zeros_like(points)
            
            # Vectorized computation of repulsion forces with better scaling
            for i in range(len(refined_circles)):
                # Compute distances to all other circles
                dists = distances[i, :]
                # Find neighbors that are too close
                close_mask = (dists < (radii[i] + radii)) & (dists > 0)
                if np.any(close_mask):
                    # Compute repulsion forces
                    dx = points[i, 0] - points[close_mask, 0]
                    dy = points[i, 1] - points[close_mask, 1]
                    dists_close = dists[close_mask]
                    
                    # Normalize and compute forces with better scaling
                    norm = np.sqrt(dx*dx + dy*dy) + 1e-10
                    force_magnitude = ((radii[i] + radii[close_mask]) - dists_close) / (norm + 1e-10)
                    
                    forces[i, 0] -= np.sum(force_magnitude * dx / norm) * 0.1
                    forces[i, 1] -= np.sum(force_magnitude * dy / norm) * 0.1
            
            # Apply forces with boundary constraints
            for i in range(len(refined_circles)):
                # Apply force to position
                refined_circles[i, 0] = np.clip(refined_circles[i, 0] + forces[i, 0], 
                                              refined_circles[i, 2], 1 - refined_circles[i, 2])
                refined_circles[i, 1] = np.clip(refined_circles[i, 1] + forces[i, 1], 
                                              refined_circles[i, 2], 1 - refined_circles[i, 2])
            
            # Try to increase radii slightly with better strategy
            for i in range(len(refined_circles)):
                old_radius = refined_circles[i, 2]
                # Increase radius more aggressively but carefully
                test_radius = min(old_radius + 0.005, 0.25)
                refined_circles[i, 2] = test_radius
                
                # Check constraints after radius change
                containment_violations, overlap_violations = compute_constraints_fast(refined_circles)
                if len(containment_violations) > 0 or len(overlap_violations) > 0:
                    refined_circles[i, 2] = old_radius
        
        return refined_circles
    
    def improved_simulated_annealing(initial_circles, max_iter=300):
        """Improved simulated annealing with better acceptance criteria"""
        current_circles = initial_circles.copy()
        best_circles = current_circles.copy()
        best_sum = np.sum(current_circles[:, 2])
        
        # Adaptive cooling schedule with faster cooling
        temperature = 0.1
        cooling_rate = 0.995
        
        for iteration in range(max_iter):
            # Make a small random change
            new_circles = current_circles.copy()
            
            # Pick a random circle to modify
            idx = np.random.randint(0, n)
            
            # Modify either position or radius with different probabilities
            if np.random.random() < 0.7:  # 70% chance to modify position
                # Modify position with larger steps
                new_circles[idx, 0] += np.random.normal(0, 0.015)
                new_circles[idx, 1] += np.random.normal(0, 0.015)
                # Clip to valid range
                new_circles[idx, 0] = np.clip(new_circles[idx, 0], 
                                            new_circles[idx, 2], 1 - new_circles[idx, 2])
                new_circles[idx, 1] = np.clip(new_circles[idx, 1], 
                                            new_circles[idx, 2], 1 - new_circles[idx, 2])
            else:
                # Modify radius with higher probability for growth
                delta_r = np.random.normal(0, 0.01)
                new_circles[idx, 2] = np.clip(new_circles[idx, 2] + delta_r, 
                                            0.001, 0.3)
            
            # Check constraints
            containment_violations, overlap_violations = compute_constraints_fast(new_circles)
            
            if len(containment_violations) == 0 and len(overlap_violations) == 0:
                # Accept if better or with probability based on temperature
                new_sum = np.sum(new_circles[:, 2])
                delta = new_sum - best_sum
                
                # Better acceptance criterion
                if delta > 0 or np.random.random() < np.exp(delta / (temperature + 1e-10)):
                    current_circles = new_circles
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = new_circles.copy()
            
            # Cool down
            temperature *= cooling_rate
            
        return best_circles
    
    def global_search_optimization(initial_circles, max_iter=100):
        """Perform a global search to find better starting points"""
        best_circles = initial_circles.copy()
        best_sum = np.sum(initial_circles[:, 2])
        
        # Try several different random starting configurations
        for _ in range(8):  # More attempts for better exploration
            # Create a new random configuration with different initialization
            np.random.seed(np.random.randint(10000))
            random_config = generate_better_initial_placement()
            
            # Run optimization on this configuration
            optimized = optimize_with_scipy(random_config, maxiter=300)
            
            # Run refinement steps
            refined = enhanced_local_improvement(optimized, max_iter=100)
            refined = adaptive_repulsion_refinement(refined, max_iter=50)
            
            current_sum = np.sum(refined[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = refined.copy()
        
        return best_circles
    
    def advanced_geometric_initialization():
        """Use a more sophisticated geometric initialization based on known good packings"""
        # Try a more structured approach: place circles in concentric rings
        circles = []
        
        # Place some circles in the center area
        center_radius = 0.1
        circles.append([0.5, 0.5, center_radius])
        
        # Place remaining circles around the perimeter in a pattern
        remaining_count = n - 1
        
        # Distribute remaining circles around the edges in a strategic way
        # Using a spiral-like pattern to avoid clustering
        angle_step = 2 * np.pi / remaining_count
        radius_step = 0.3 / remaining_count
        
        for i in range(remaining_count):
            # Position along a spiral pattern near the edges
            angle = i * angle_step
            radius = 0.1 + i * radius_step
            
            # Distribute along perimeter and edges
            if i < remaining_count // 4:
                # Top edge
                x = 0.1 + i * 0.8 / (remaining_count // 4)
                y = 0.9
            elif i < 2 * remaining_count // 4:
                # Right edge
                x = 0.9
                y = 0.9 - (i - remaining_count // 4) * 0.8 / (remaining_count // 4)
            elif i < 3 * remaining_count // 4:
                # Bottom edge
                x = 0.9 - (i - 2 * remaining_count // 4) * 0.8 / (remaining_count // 4)
                y = 0.1
            else:
                # Left edge
                x = 0.1
                y = 0.1 + (i - 3 * remaining_count // 4) * 0.8 / (remaining_count // 4)
            
            # Add some randomness to break symmetry
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            
            # Ensure within bounds and create reasonable radius
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            r = 0.05 + np.random.random() * 0.05
            
            circles.append([x, y, r])
        
        # Ensure we have exactly n circles
        circles = circles[:n]
        
        # Ensure all circles are within bounds
        for i in range(len(circles)):
            circles[i][0] = np.clip(circles[i][0], circles[i][2], 1 - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], 1 - circles[i][2])
            
        return np.array(circles)
    
    def hybrid_approach_initialization():
        """Create even better initial configuration using a hybrid approach"""
        # Start with a better hexagonal packing pattern
        circles = []
        
        # Create a regular hexagonal lattice
        rows = 6
        cols = 6
        spacing = 0.15  # Spacing between centers
        radius = 0.05  # Initial radius
        
        # Generate hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * spacing * 0.5
                x = 0.1 + j * spacing + x_offset
                y = 0.1 + i * spacing * 0.866  # Vertical spacing for hexagon
                
                # Clip to avoid going out of bounds
                x = np.clip(x, radius, 1-radius)
                y = np.clip(y, radius, 1-radius)
                
                circles.append([x, y, radius])
        
        # Trim to exactly n circles
        circles = circles[:n]
        
        # Randomize positions slightly and adjust radii
        for i in range(len(circles)):
            # Add small random perturbation
            circles[i][0] += np.random.normal(0, 0.005)
            circles[i][1] += np.random.normal(0, 0.005)
            
            # Adjust radius to be more reasonable
            circles[i][2] = np.clip(circles[i][2] + np.random.normal(0, 0.005), 0.01, 0.15)
            
            # Ensure within bounds
            circles[i][0] = np.clip(circles[i][0], circles[i][2], 1 - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], 1 - circles[i][2])
        
        return np.array(circles)
    
    def advanced_optimization_pipeline(initial_circles):
        """Run a more comprehensive optimization pipeline"""
        current = initial_circles.copy()
        
        # Multiple refinement passes with increasing intensity
        for pass_num in range(3):
            # Pass 1: Global optimization with various methods
            if pass_num == 0:
                # More aggressive optimization
                optimized = optimize_with_scipy(current, maxiter=800)
                current = optimized
                
            # Pass 2: Local refinement with enhanced strategies
            elif pass_num == 1:
                # Enhanced refinement
                refined = enhanced_local_improvement(current, max_iter=200)
                refined = adaptive_repulsion_refinement(refined, max_iter=100)
                current = refined
                
            # Pass 3: Simulated annealing for global exploration
            elif pass_num == 2:
                # Final global search
                sa_result = improved_simulated_annealing(current, max_iter=300)
                current = sa_result
        
        return current
    
    # Main algorithm - use improved approach
    # Step 1: Try advanced hybrid initialization
    initial_config = hybrid_approach_initialization()
    
    # Step 2: Global search to find better starting points
    initial_config = global_search_optimization(initial_config, max_iter=30)
    
    # Step 3: Run comprehensive optimization pipeline
    final_config = advanced_optimization_pipeline(initial_config)
    
    # Step 4: Final refinement with enhanced techniques
    final_config = enhanced_local_improvement(final_config, max_iter=100)
    final_config = adaptive_repulsion_refinement(final_config, max_iter=50)
    
    # Final validation
    containment_violations, overlap_violations = compute_constraints_fast(final_config)
    if len(containment_violations) > 0 or len(overlap_violations) > 0:
        # Fall back to initial configuration if there are issues
        final_config = initial_config
    
    return final_config


# EVOLVE-BLOCK-END
