# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import time
from typing import Tuple, List
import warnings
from itertools import combinations

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining:
    1. Hexagonal grid initialization for good coverage
    2. Local optimization with gradient-based methods
    3. Constraint satisfaction with efficient spatial data structures
    4. Multi-start strategy for better global exploration
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    timeout_seconds = 55  # Leave some buffer for cleanup
    
    # Strategy 1: Hexagonal grid initialization for good distribution
    def hexagonal_initialization():
        """Initialize with a hexagonal packing pattern"""
        # Create hexagonal grid pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x_offset = (i % 2) * spacing_x / 2
                x = x_offset + (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Skip if out of bounds
                if x < 0 or x > 1 or y < 0 or y > 1:
                    continue
                    
                # Initial radius estimate based on spacing
                radius = min(spacing_x, spacing_y) / 2 * 0.8
                
                # Adjust radius to respect boundaries
                radius = min(radius, x, 1-x, y, 1-y)
                
                if radius > 0.01:
                    circles.append([x, y, radius])
                
                if len(circles) >= n:
                    break
            if len(circles) >= n:
                break
        
        # If we don't have enough circles, fill with random ones
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            radius = min(0.2, x, 1-x, y, 1-y)
            if radius > 0.01:
                circles.append([x, y, max(radius, 0.01)])
        
        return np.array(circles[:n])
    
    # Strategy 2: Local optimization with improved constraints
    def local_optimization(circles: np.ndarray) -> np.ndarray:
        """Use local optimization to refine the solution"""
        def objective(params):
            total_radius = 0
            for i in range(n):
                total_radius += params[3*i+2]  # Add radius (index 2 of each circle)
            return -total_radius  # Negative because we minimize
        
        def constraint_func(params):
            # Return list of constraint violations (negative values mean violation)
            violations = []
            
            # Containment constraints
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Circle must fit within boundaries
                violations.append(x - r)      # x - r >= 0
                violations.append(1 - x - r)  # 1 - x - r >= 0
                violations.append(y - r)      # y - r >= 0
                violations.append(1 - y - r)  # 1 - y - r >= 0
            
            # Non-overlap constraints
            for i, j in combinations(range(n), 2):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                # Distance between centers minus sum of radii should be >= 0
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraint = dist - (r1 + r2)
                violations.append(constraint)
            
            return np.array(violations)
        
        # Flatten for optimization
        initial_params = []
        for x, y, r in circles:
            initial_params.extend([x, y, r])
        
        # Set up bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Set up constraints - only inequality constraints
        cons = [{'type': 'ineq', 'fun': lambda p: constraint_func(p)}]
        
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-4}
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
    
    # Strategy 3: Improved constraint validation and correction
    def validate_and_correct(circles: np.ndarray) -> np.ndarray:
        """Validate constraints and correct violations efficiently"""
        # Make a copy to work with
        corrected = circles.copy()
        
        # Fix containment issues first
        for i in range(len(corrected)):
            x, y, r = corrected[i]
            # Adjust position to fit within bounds
            corrected[i, 0] = np.clip(x, r, 1-r)
            corrected[i, 1] = np.clip(y, r, 1-r)
        
        # Fix overlaps using a more efficient approach
        # Create spatial index for faster neighbor searches
        tree = KDTree(corrected[:, :2])
        
        # Try to resolve overlaps systematically
        max_iterations = 100
        for iteration in range(max_iterations):
            # Find all pairs that are overlapping
            pairs = tree.query_pairs(r=0.001)  # Very small threshold to catch all overlaps
            overlaps = []
            
            for i, j in pairs:
                x1, y1, r1 = corrected[i]
                x2, y2, r2 = corrected[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < (r1 + r2):
                    overlaps.append((i, j, dist, r1 + r2))
            
            if not overlaps:
                break
                
            # Sort by overlap severity (deepest overlap first)
            overlaps.sort(key=lambda x: x[3] - x[2], reverse=True)
            
            # Resolve the most severe overlaps
            resolved_count = 0
            for i, j, dist, sum_radii in overlaps[:min(10, len(overlaps))]:  # Resolve up to 10
                if resolved_count >= 5:  # Limit per iteration
                    break
                    
                x1, y1, r1 = corrected[i]
                x2, y2, r2 = corrected[j]
                
                # Move circles apart along the line connecting their centers
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0.001:  # Avoid division by zero
                    # Calculate how much to move each circle
                    overlap = sum_radii - dist
                    move_amount = overlap / 2
                    
                    # Normalize direction vector
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    # Move circles away from each other
                    corrected[i, 0] -= dx_norm * move_amount
                    corrected[i, 1] -= dy_norm * move_amount
                    corrected[j, 0] += dx_norm * move_amount
                    corrected[j, 1] += dy_norm * move_amount
                    
                    # Keep within bounds
                    corrected[i, 0] = np.clip(corrected[i, 0], r1, 1-r1)
                    corrected[i, 1] = np.clip(corrected[i, 1], r1, 1-r1)
                    corrected[j, 0] = np.clip(corrected[j, 0], r2, 1-r2)
                    corrected[j, 1] = np.clip(corrected[j, 1], r2, 1-r2)
                    
                    resolved_count += 1
        
        return corrected
    
    # Strategy 4: Multi-start optimization approach
    def multi_start_optimization():
        """Try multiple starting points to find better solutions"""
        best_circles = None
        best_sum = 0
        
        # Try different initializations
        initializations = [
            hexagonal_initialization,
            lambda: hexagonal_initialization() + np.random.normal(0, 0.02, (32, 3)),
        ]
        
        for i, init_func in enumerate(initializations):
            try:
                circles = init_func()
                # Apply local optimization
                circles = local_optimization(circles)
                # Validate and correct
                circles = validate_and_correct(circles)
                
                # Calculate sum of radii
                sum_radii = np.sum(circles[:, 2])
                
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_circles = circles.copy()
                    
            except Exception as e:
                continue  # Skip this attempt if it fails
        
        # If no good solution found, use default initialization
        if best_circles is None:
            circles = hexagonal_initialization()
            circles = local_optimization(circles)
            circles = validate_and_correct(circles)
            best_circles = circles
        
        return best_circles
    
    # Main algorithm workflow
    start_time = time.time()
    
    # Step 1: Multi-start optimization with different strategies
    circles = multi_start_optimization()
    
    # Step 2: Additional refinement with local optimization
    if time.time() - start_time < timeout_seconds * 0.8:
        circles = local_optimization(circles)
        circles = validate_and_correct(circles)
    
    # Step 3: Final validation
    circles = validate_and_correct(circles)
    
    return circles


# EVOLVE-BLOCK-END
