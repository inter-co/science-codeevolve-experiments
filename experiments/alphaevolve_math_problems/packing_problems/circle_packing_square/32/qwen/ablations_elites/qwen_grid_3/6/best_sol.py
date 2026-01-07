# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import warnings

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def get_initial_config() -> np.ndarray:
    """Generate initial configuration using a better geometric initialization"""
    n = 32
    circles = np.zeros((n, 3))
    
    # Use a more strategic initial placement inspired by inspiration program 2
    # Better approach: use a more systematic grid with adaptive spacing
    rows = 5
    cols = 7
    
    # Calculate spacing
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Create a more efficient grid pattern
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= 32:
                break
            # Position with slight jitter for better distribution
            x = (j + 1) * spacing_x + np.random.uniform(-0.1 * spacing_x, 0.1 * spacing_x)
            y = (i + 1) * spacing_y + np.random.uniform(-0.1 * spacing_y, 0.1 * spacing_y)
            
            # Initial radius - use a value that allows good packing
            r = min(spacing_x, spacing_y) * 0.4
            
            # Ensure circle fits in square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[count] = [x, y, r]
            count += 1
            
        if count >= 32:
            break
    
    # For remaining circles, place them strategically
    for i in range(count, 32):
        # Try to place in a way that avoids immediate conflicts
        placed = False
        max_attempts = 1000
        attempts = 0
        while not placed and attempts < max_attempts:
            # Try placing near corners or edges for better spread
            if i < 4:  # First few go to corners
                corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
                corner_idx = i % len(corner_positions)
                x_base, y_base = corner_positions[corner_idx]
                x = x_base + np.random.uniform(-0.1, 0.1)
                y = y_base + np.random.uniform(-0.1, 0.1)
            else:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
            r = np.random.uniform(0.01, 0.1)
            
            # Check boundary constraints
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                # Check overlap with existing circles
                valid = True
                for j in range(i):
                    x_prev, y_prev, r_prev = circles[j]
                    distance = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                    if distance < r + r_prev:
                        valid = False
                        break
                if valid:
                    circles[i] = [x, y, r]
                    placed = True
            attempts += 1
    
    return circles

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate total sum of radii"""
    return np.sum(circles[:, 2])

def check_constraints(circles: np.ndarray) -> Tuple[bool, list]:
    """Check if all circles satisfy constraints and return violations"""
    n = len(circles)
    violations = []
    
    # Check containment constraints efficiently using vectorized operations
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    
    # Check containment constraints
    containment_violations = np.where((x_coords - radii < 0) | 
                                     (x_coords + radii > 1) | 
                                     (y_coords - radii < 0) | 
                                     (y_coords + radii > 1))[0]
    
    for i in containment_violations:
        violations.append(f"Circle {i} violates containment constraints")
    
    # Check overlap constraints efficiently using pairwise distance matrix
    if len(violations) == 0 and n > 1:
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create mask for upper triangle (avoid double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        distance_matrix = distances[mask]
        min_distance_matrix = (radii[:, None] + radii[None, :])[mask]
        
        # Find overlapping pairs - with small tolerance for numerical precision
        overlap_indices = np.where(distance_matrix < min_distance_matrix - 1e-10)[0]
        
        if len(overlap_indices) > 0:
            # Get the corresponding circle indices
            for idx in overlap_indices:
                i = np.triu_indices(n, k=1)[0][idx]
                j = np.triu_indices(n, k=1)[1][idx]
                violations.append(f"Circles {i} and {j} overlap")
    
    return len(violations) == 0, violations

def optimize_with_scipy(initial_circles: np.ndarray) -> np.ndarray:
    """Use scipy optimization to improve the configuration - inspired by inspiration program 2"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_vars = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_vars.extend([x, y, r])
    
    # Define objective function to maximize sum of radii (negative because we minimize)
    def objective(vars_flat):
        circles = np.array(vars_flat).reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Define constraints with better numerical stability
    def boundary_constraints(vars_flat):
        """Ensure all circles are within the unit square"""
        circles = np.array(vars_flat).reshape(-1, 3)
        constraints = []
        
        for i in range(n):
            x, y, r = circles[i]
            # Circle must fit entirely within square with margin for numerical stability
            constraints.append(x - r - 1e-6)      # x - r >= 1e-6
            constraints.append(1 - x - r - 1e-6)  # 1 - x - r >= 1e-6
            constraints.append(y - r - 1e-6)      # y - r >= 1e-6
            constraints.append(1 - y - r - 1e-6)  # 1 - y - r >= 1e-6
            
        return np.array(constraints)
    
    def overlap_constraints(vars_flat):
        """Ensure no two circles overlap with numerical tolerance"""
        circles = np.array(vars_flat).reshape(-1, 3)
        constraints = []
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                
                # Add small tolerance to avoid numerical issues
                constraints.append(distance_sq - min_distance_sq - 1e-10)
                
        return np.array(constraints)
    
    # Set up bounds for variables with tighter ranges
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds
        bounds.append((0.001, 0.999))
        # r bounds - slightly smaller upper bound for safety
        bounds.append((0.001, 0.49))
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    # Try multiple optimization methods for robustness
    methods_to_try = ['SLSQP', 'trust-constr']
    best_result = None
    best_sum = -np.inf
    
    for method in methods_to_try:
        try:
            result = minimize(objective, initial_vars, method=method, 
                             bounds=bounds, constraints=cons, 
                             options={'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-7})
            
            if result.success:
                # Evaluate the result
                circles = result.x.reshape(-1, 3)
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If any method succeeded, return the best result
    if best_result is not None:
        optimized_circles = best_result.x.reshape(-1, 3)
        return optimized_circles
    else:
        # If all optimization failed, return initial configuration
        return initial_circles

def force_based_optimization(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply force-based optimization to improve packing"""
    n = len(circles)
    positions = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    # Parameters for force calculation - fine-tuned for better convergence
    k_repel = 1500.0
    k_containment = 1500.0
    dt = 0.0005
    
    for iteration in range(max_iter):
        forces = np.zeros_like(positions)
        
        # Repulsion forces between circles
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                if dist > 0 and dist < (radii[i] + radii[j]):
                    # Repulsive force with smoother decay
                    force_magnitude = k_repel * (radii[i] + radii[j] - dist) / (dist + 1e-8)
                    forces[i, 0] += force_magnitude * dx / dist
                    forces[i, 1] += force_magnitude * dy / dist
                    forces[j, 0] -= force_magnitude * dx / dist
                    forces[j, 1] -= force_magnitude * dy / dist
        
        # Containment forces (push back into bounds)
        for i in range(n):
            # Push away from boundaries with stronger force near edges
            boundary_forces = np.array([
                max(0, radii[i] - positions[i, 0]),  # left boundary
                max(0, radii[i] - positions[i, 1]),  # bottom boundary
                max(0, positions[i, 0] + radii[i] - 1),  # right boundary
                max(0, positions[i, 1] + radii[i] - 1)   # top boundary
            ])
            
            forces[i, 0] += k_containment * boundary_forces[0] - k_containment * boundary_forces[2]
            forces[i, 1] += k_containment * boundary_forces[1] - k_containment * boundary_forces[3]
        
        # Update positions
        positions += dt * forces
        
        # Keep positions within bounds
        positions[:, 0] = np.clip(positions[:, 0], radii, 1-radii)
        positions[:, 1] = np.clip(positions[:, 1], radii, 1-radii)
    
    # Create updated circles array
    updated_circles = np.column_stack([positions, radii])
    return updated_circles

def enhanced_local_search(circles: np.ndarray) -> np.ndarray:
    """Enhanced local search with more sophisticated strategies"""
    n = len(circles)
    current_circles = circles.copy()
    improved = True
    
    iteration = 0
    max_iterations = 1000  # Increased iterations for better convergence
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Strategy: Try to increase radii systematically with better compromise logic
        # Sort by current radius to focus on smaller ones first
        sorted_indices = np.argsort(current_circles[:, 2])
        
        for i in sorted_indices:
            # Try to increase radius of circle i
            original_radius = current_circles[i, 2]
            test_radius = min(0.4, original_radius + 0.003)  # Smaller increments for fine tuning
            
            # Test if we can increase this radius
            test_circles = current_circles.copy()
            test_circles[i, 2] = test_radius
            
            # Check constraints
            valid, _ = check_constraints(test_circles)
            if valid:
                test_sum = calculate_radius_sum(test_circles)
                if test_sum > calculate_radius_sum(current_circles):
                    current_circles = test_circles
                    improved = True
            else:
                # Try to make a compromise with neighbors
                # Try decreasing some nearby radii to make room
                for j in range(n):
                    if i != j and current_circles[j, 2] > 0.02:
                        test_circles = current_circles.copy()
                        test_circles[j, 2] = max(0.01, test_circles[j, 2] - 0.001)
                        test_circles[i, 2] = test_radius
                        
                        valid, _ = check_constraints(test_circles)
                        if valid:
                            test_sum = calculate_radius_sum(test_circles)
                            if test_sum > calculate_radius_sum(current_circles):
                                current_circles = test_circles
                                improved = True
                                break
    
    return current_circles

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Improve initial configuration using multiple techniques"""
    n = len(initial_circles)
    
    # Start with better optimization approach from inspiration program 2
    optimized_circles = optimize_with_scipy(initial_circles)
    
    # Apply force-based optimization for fine-tuning
    optimized_circles = force_based_optimization(optimized_circles, 120)
    
    # Apply enhanced local search
    optimized_circles = enhanced_local_search(optimized_circles)
    
    # Try several local search approaches with different perturbations
    best_circles = optimized_circles.copy()
    best_sum = calculate_radius_sum(best_circles)
    
    # Multiple restarts with different perturbations - more aggressive
    for restart in range(8):
        # Perturb the configuration differently
        perturbed = initial_circles.copy()
        
        # Perturb a varying number of circles
        num_perturbed = max(1, n//3 + restart*2)  # Varying perturbation
        perturb_indices = np.random.choice(n, size=min(num_perturbed, n), replace=False)
        
        for i in perturb_indices:
            # Different perturbation sizes for different restarts
            perturbation_size = 0.03 + restart * 0.015  # Increase with restart count
            perturbed[i, 0] += random.uniform(-perturbation_size, perturbation_size)
            perturbed[i, 1] += random.uniform(-perturbation_size, perturbation_size)
            perturbed[i, 2] += random.uniform(-0.02, 0.02)
            
            # Ensure bounds
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0.05, 0.95)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0.05, 0.95)
            perturbed[i, 2] = np.clip(perturbed[i, 2], 0.01, 0.4)
        
        # Apply optimization to perturbed version
        improved_circles = optimize_with_scipy(perturbed)
        
        # Apply force-based optimization to perturbed version
        improved_circles = force_based_optimization(improved_circles, 100)
        
        # Apply enhanced local search
        improved_circles = enhanced_local_search(improved_circles)
        
        # Keep the best result
        current_sum = calculate_radius_sum(improved_circles)
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = improved_circles
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Generate initial configuration
    initial_config = get_initial_config()
    
    # Optimize the configuration
    optimized_circles = optimize_circles(initial_config)
    
    # Final validation and cleanup
    valid, violations = check_constraints(optimized_circles)
    if not valid:
        # If constraints are violated, revert to initial config with small radii
        print(f"Constraints violated: {violations}")
        final_circles = get_initial_config()
        final_circles[:, 2] = 0.02  # Small equal radii
    else:
        final_circles = optimized_circles
    
    return final_circles


# EVOLVE-BLOCK-END
