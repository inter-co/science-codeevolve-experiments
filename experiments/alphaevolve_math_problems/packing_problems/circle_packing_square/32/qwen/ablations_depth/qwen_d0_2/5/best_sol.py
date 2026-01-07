# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import time
from typing import Tuple, List
import warnings
from itertools import combinations
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining:
    1. Advanced grid initialization with better space utilization
    2. Physics-inspired particle system for overlap resolution
    3. Improved optimization with better constraint handling
    4. Multi-start strategy with diverse initializations
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    timeout_seconds = 55  # Leave some buffer for cleanup
    
    # Strategy 1: Advanced grid initialization for better space utilization
    def advanced_grid_initialization():
        """Initialize with an optimized grid pattern"""
        # Use a more sophisticated approach: try to place circles in a way that maximizes
        # the minimum distance between centers while maintaining good coverage
        
        # Create a coarse grid and then optimize positions
        circles = []
        
        # Try different grid configurations
        grid_configs = [
            (5, 7),  # 5 rows, 7 columns
            (6, 6),  # 6 rows, 6 columns  
            (7, 5),  # 7 rows, 5 columns
        ]
        
        best_config = None
        best_density = 0
        
        for rows, cols in grid_configs:
            # Calculate spacing
            spacing_x = 1.0 / cols
            spacing_y = 1.0 / rows
            
            # Create initial positions
            temp_circles = []
            for i in range(rows):
                for j in range(cols):
                    # Offset every other row
                    x_offset = (i % 2) * spacing_x / 2
                    x = x_offset + (j + 0.5) * spacing_x
                    y = (i + 0.5) * spacing_y
                    
                    # Skip if out of bounds
                    if x < 0 or x > 1 or y < 0 or y > 1:
                        continue
                        
                    temp_circles.append([x, y, 0])  # Placeholder radius
                    
                    if len(temp_circles) >= n:
                        break
                if len(temp_circles) >= n:
                    break
            
            if len(temp_circles) >= n:
                # Calculate density (approximate)
                area_covered = len(temp_circles) * (spacing_x * spacing_y) / 4  # Approximate
                density = area_covered / 1.0
                
                if density > best_density:
                    best_density = density
                    best_config = temp_circles[:n]
        
        # If we don't have enough circles, fill with random ones
        if len(best_config) < n:
            best_config = []
            for i in range(n):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                best_config.append([x, y, 0])
        
        # Now assign initial radii based on proximity to neighbors
        circles_array = np.array(best_config)
        
        # Initialize radii to avoid overlaps
        for i in range(len(circles_array)):
            x, y, _ = circles_array[i]
            
            # Find nearest neighbors
            distances = []
            for j in range(len(circles_array)):
                if i != j:
                    x2, y2, _ = circles_array[j]
                    dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                    distances.append(dist)
            
            # Set radius to be as large as possible without overlapping neighbors
            if distances:
                min_dist = min(distances)
                # Radius should be less than half the distance to nearest neighbor
                radius = min_dist / 2 * 0.95  # Slightly conservative
            else:
                radius = 0.1
                
            # Ensure radius respects boundary constraints
            radius = min(radius, x, 1-x, y, 1-y)
            
            circles_array[i, 2] = max(radius, 0.005)
        
        return circles_array
    
    # Strategy 2: Physics-based particle system for overlap resolution
    def physics_based_correction(circles: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Use physics-inspired approach to resolve overlaps"""
        corrected = circles.copy()
        
        # Create spatial index for efficient neighbor queries
        tree = KDTree(corrected[:, :2])
        
        # Spring constants and parameters
        spring_constant = 0.1
        repulsion_strength = 10.0
        dt = 0.01
        
        for iter_num in range(iterations):
            # Calculate forces
            forces = np.zeros_like(corrected[:, :2])
            
            # Repulsion forces from overlaps
            pairs = tree.query_pairs(r=0.001)  # Get all close pairs
            for i, j in pairs:
                x1, y1, r1 = corrected[i]
                x2, y2, r2 = corrected[j]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0 and dist < (r1 + r2):
                    # Overlapping - apply repulsion force
                    force_magnitude = repulsion_strength * (r1 + r2 - dist) / (dist + 1e-8)
                    forces[i, 0] -= force_magnitude * dx / dist
                    forces[i, 1] -= force_magnitude * dy / dist
                    forces[j, 0] += force_magnitude * dx / dist
                    forces[j, 1] += force_magnitude * dy / dist
            
            # Boundary forces (push back into bounds)
            for i in range(len(corrected)):
                x, y, r = corrected[i]
                # Force to stay within bounds
                if x < r:
                    forces[i, 0] += (r - x) * spring_constant
                elif x > 1 - r:
                    forces[i, 0] += (1 - r - x) * spring_constant
                    
                if y < r:
                    forces[i, 1] += (r - y) * spring_constant
                elif y > 1 - r:
                    forces[i, 1] += (1 - r - y) * spring_constant
            
            # Update positions
            for i in range(len(corrected)):
                x, y, r = corrected[i]
                corrected[i, 0] = np.clip(x + forces[i, 0] * dt, r, 1-r)
                corrected[i, 1] = np.clip(y + forces[i, 1] * dt, r, 1-r)
            
            # Rebuild spatial index for next iteration
            tree = KDTree(corrected[:, :2])
        
        return corrected
    
    # Strategy 3: Improved optimization with better constraint handling
    def improved_local_optimization(circles: np.ndarray) -> np.ndarray:
        """Use local optimization with better constraint handling"""
        def objective(params):
            total_radius = 0
            for i in range(n):
                total_radius += params[3*i+2]  # Add radius (index 2 of each circle)
            return -total_radius  # Negative because we minimize
        
        def constraint_func(params):
            # Return list of constraint violations (negative values mean violation)
            violations = []
            
            # Containment constraints - ensure each circle fits within bounds
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Circle must fit within boundaries (with margin for numerical stability)
                violations.append(x - r - 1e-6)      # x - r >= 0
                violations.append(1 - x - r - 1e-6)  # 1 - x - r >= 0
                violations.append(y - r - 1e-6)      # y - r >= 0
                violations.append(1 - y - r - 1e-6)  # 1 - y - r >= 0
            
            # Non-overlap constraints - ensure no overlaps
            for i, j in combinations(range(n), 2):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                # Distance between centers minus sum of radii should be >= 0
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraint = dist - (r1 + r2) - 1e-6  # Small margin
                violations.append(constraint)
            
            return np.array(violations)
        
        # Flatten for optimization
        initial_params = []
        for x, y, r in circles:
            initial_params.extend([x, y, r])
        
        # Set up bounds - tighter bounds for better convergence
        bounds = []
        for i in range(n):
            bounds.extend([(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-4, 0.499)])
        
        # Set up constraints - only inequality constraints
        cons = [{'type': 'ineq', 'fun': lambda p: constraint_func(p)}]
        
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-5, 'eps': 1e-3}
            )
            
            if result.success:
                # Extract final result
                final_circles = []
                for i in range(n):
                    x = result.x[3*i]
                    y = result.x[3*i+1]
                    r = result.x[3*i+2]
                    final_circles.append([x, y, r])
                return np.array(final_circles)
        except Exception as e:
            # If optimization fails, return original circles
            pass
        
        return circles
    
    # Strategy 4: Multi-start optimization approach with diverse strategies
    def multi_start_optimization():
        """Try multiple starting points to find better solutions"""
        best_circles = None
        best_sum = 0
        
        # Try different initializations with varying strategies
        initializations = [
            lambda: advanced_grid_initialization(),
            lambda: advanced_grid_initialization() + np.random.normal(0, 0.01, (32, 3)),
            lambda: advanced_grid_initialization() * (1 + np.random.uniform(-0.05, 0.05, (32, 3))),
        ]
        
        # Add some random initializations
        for _ in range(3):
            random_circles = []
            for _ in range(32):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = min(0.2, x, 1-x, y, 1-y) * np.random.uniform(0.5, 1.0)
                random_circles.append([x, y, max(r, 0.01)])
            initializations.append(lambda c=random_circles: np.array(c))
        
        for i, init_func in enumerate(initializations):
            try:
                circles = init_func()
                
                # Apply physics-based correction first
                circles = physics_based_correction(circles, iterations=30)
                
                # Apply local optimization
                circles = improved_local_optimization(circles)
                
                # Apply final physics correction
                circles = physics_based_correction(circles, iterations=20)
                
                # Calculate sum of radii
                sum_radii = np.sum(circles[:, 2])
                
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_circles = circles.copy()
                    
            except Exception as e:
                continue  # Skip this attempt if it fails
        
        # If no good solution found, use default initialization
        if best_circles is None:
            circles = advanced_grid_initialization()
            circles = physics_based_correction(circles, iterations=50)
            circles = improved_local_optimization(circles)
            best_circles = circles
        
        return best_circles
    
    # Main algorithm workflow
    start_time = time.time()
    
    # Step 1: Multi-start optimization with diverse strategies
    circles = multi_start_optimization()
    
    # Step 2: Additional refinement with local optimization
    if time.time() - start_time < timeout_seconds * 0.8:
        circles = improved_local_optimization(circles)
        circles = physics_based_correction(circles, iterations=30)
    
    # Step 3: Final validation and fine-tuning
    circles = physics_based_correction(circles, iterations=50)
    
    return circles


# EVOLVE-BLOCK-END
