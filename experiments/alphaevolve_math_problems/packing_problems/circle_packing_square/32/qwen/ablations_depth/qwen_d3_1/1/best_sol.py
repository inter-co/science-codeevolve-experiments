# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from scipy.spatial import cKDTree
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining simulated annealing initialization and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initialization using simulated annealing-inspired approach
    def generate_better_initialization():
        """Generate initial configuration using a more systematic approach"""
        # Start with a coarse grid and then refine
        circles = []
        
        # Try a more structured approach: place circles in a way that maximizes initial coverage
        # Use a spiral-like pattern or adaptive grid
        
        # Create a coarse grid first
        grid_size = 6
        spacing = 1.0 / (grid_size + 1)
        
        # Fill grid points with some randomness to avoid perfect patterns
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                # Add jitter to positions
                x = spacing * (j + 1) + (random.random() - 0.5) * spacing * 0.3
                y = spacing * (i + 1) + (random.random() - 0.5) * spacing * 0.3
                if 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
        
        # If we don't have enough positions, fill with random points
        while len(positions) < n:
            x = random.random()
            y = random.random()
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append([x, y])
        
        positions = positions[:n]
        
        # Initialize with reasonable radii based on proximity to edges
        radii = []
        for i, (x, y) in enumerate(positions):
            # Maximum radius considering boundaries
            max_r = min(x, 1-x, y, 1-y)
            # Start with a small fraction to allow optimization
            r = max_r * 0.25
            radii.append(r)
        
        # Now perform a simple local optimization to reduce overlaps
        # This helps create a better starting point for the full optimization
        for _ in range(50):  # Some local iterations
            improved = False
            for i in range(n):
                # Try to increase radius while maintaining constraints
                x, y = positions[i]
                r = radii[i]
                max_r = min(x, 1-x, y, 1-y)
                
                # Try to increase radius
                new_r = min(max_r, r + 0.01)
                
                # Check if this change would cause overlaps
                valid = True
                for j in range(n):
                    if i != j:
                        x_j, y_j = positions[j]
                        r_j = radii[j]
                        dist_sq = (x - x_j)**2 + (y - y_j)**2
                        dist = np.sqrt(dist_sq)
                        if dist < new_r + r_j:
                            valid = False
                            break
                
                if valid and new_r > r:
                    radii[i] = new_r
                    improved = True
            
            if not improved:
                break
        
        return [(positions[i][0], positions[i][1], radii[i]) for i in range(n)]
    
    # Generate initial configuration
    initial_circles = generate_better_initialization()
    
    # Phase 2: Set up optimization variables
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    initial_vars = []
    for circle in initial_circles:
        x, y, r = circle
        initial_vars.extend([x, y, r])
    
    # Phase 3: Define constraint functions with improved efficiency
    def contain_constraints(vars):
        """Ensure all circles are contained in the unit square"""
        cons = []
        for i in range(n):
            idx = 3*i
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            # Circle must be contained in unit square: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
            cons.append(x - r)  # x - r >= 0
            cons.append(1 - x - r)  # 1 - x - r >= 0
            cons.append(y - r)  # y - r >= 0
            cons.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(cons)
    
    def overlap_constraints(vars):
        """Ensure no overlapping circles - more efficient implementation"""
        cons = []
        
        # Instead of using KDTree for every constraint evaluation, compute all pairwise distances
        # But only when needed for performance - use a smarter approach
        
        # Convert to numpy array for easier access
        positions = np.array([(vars[3*i], vars[3*i+1]) for i in range(n)])
        radii = np.array([vars[3*i+2] for i in range(n)])
        
        # Check all pairs efficiently
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1, r2 = radii[i], radii[j]
                
                # Distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                dist = np.sqrt(dist_sq)
                
                # Constraint: distance >= r1 + r2 (non-overlapping)
                # So we want: dist - r1 - r2 >= 0
                cons.append(dist - r1 - r2)
        
        return np.array(cons)
    
    # Phase 4: Define objective function
    def objective(vars):
        """Maximize sum of radii (minimize negative sum)"""
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius  # Negative because we minimize
    
    # Phase 5: Run optimization with enhanced robustness
    try:
        # Define bounds for variables with tighter constraints
        bounds = []
        for i in range(n):
            # x bounds: [r, 1-r] 
            bounds.append((0.001, 0.999))  # x coordinate
            bounds.append((0.001, 0.999))  # y coordinate
            bounds.append((0.001, 0.499))  # radius (limit to prevent too large circles)
        
        # Create constraint dictionaries
        constraints = [
            {'type': 'ineq', 'fun': contain_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        # Try multiple optimization methods with different starting points
        methods_to_try = ['trust-constr', 'SLSQP']
        best_result = None
        best_value = float('-inf')
        
        # Multiple random restarts for better convergence
        for restart in range(5):
            # Slightly perturb initial solution for diversity
            perturbed_vars = initial_vars.copy()
            
            # Add more significant perturbation for better exploration
            for i in range(0, len(perturbed_vars), 3):
                if i+0 < len(perturbed_vars):
                    perturbed_vars[i] += (random.random() - 0.5) * 0.1
                    perturbed_vars[i] = max(0.001, min(0.999, perturbed_vars[i]))
                if i+1 < len(perturbed_vars):
                    perturbed_vars[i+1] += (random.random() - 0.5) * 0.1
                    perturbed_vars[i+1] = max(0.001, min(0.999, perturbed_vars[i+1]))
                if i+2 < len(perturbed_vars):
                    perturbed_vars[i+2] += (random.random() - 0.5) * 0.05
                    perturbed_vars[i+2] = max(0.001, min(0.499, perturbed_vars[i+2]))
            
            for method in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        perturbed_vars,
                        method=method,
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        # Calculate the actual sum of radii
                        current_sum = -objective(result.x)
                        if current_sum > best_value:
                            best_value = current_sum
                            best_result = result
                except:
                    continue
        
        # If we found a good result, use it; otherwise fall back to initial
        if best_result is not None and best_result.success:
            final_vars = best_result.x
        else:
            # Fall back to initial configuration with more aggressive refinement
            final_vars = initial_vars.copy()
            
            # More sophisticated refinement using local search with better logic
            try:
                # Perform more iterations of local optimization
                for iter in range(50):
                    improved = False
                    # Try to improve each circle individually
                    for i in range(n):
                        # Save current state
                        old_x, old_y, old_r = final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]
                        
                        # Try to increase radius while maintaining constraints
                        x, y, r = old_x, old_y, old_r
                        
                        # Find maximum possible radius
                        max_r = min(x, 1-x, y, 1-y)
                        
                        # Check what we can increase to
                        new_r = min(max_r, r + 0.01)
                        
                        # Check overlap constraints with neighbors
                        valid = True
                        for j in range(n):
                            if i != j:
                                x_j, y_j, r_j = final_vars[3*j], final_vars[3*j+1], final_vars[3*j+2]
                                dist = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                                if dist < new_r + r_j:
                                    valid = False
                                    break
                        
                        if valid and new_r > r:
                            final_vars[3*i + 2] = new_r
                            improved = True
                    
                    if not improved:
                        break
            except:
                pass
        
        # Extract final result
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]]
            
        return circles
        
    except Exception as e:
        # Fallback: return the initial configuration if optimization fails
        circles = np.zeros((n, 3))
        for i in range(n):
            x, y, r = initial_circles[i]
            circles[i] = [x, y, r]
        return circles


# EVOLVE-BLOCK-END
