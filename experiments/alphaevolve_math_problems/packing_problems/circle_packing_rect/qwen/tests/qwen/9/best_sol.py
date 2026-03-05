# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import random
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Implements the most effective approach from Inspiration Program 1, which achieved the highest score of 2.3595858432591945.
    This approach combines constraint programming with multiple optimization restarts for maximum performance.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Try different rectangle dimensions to find optimal packing
    # Based on the best performing ratios from Inspiration 1
    rectangles = [(1.2, 0.8), (1.5, 0.5), (0.8, 1.2), (1.0, 1.0), (2.0, 0.2), (1.8, 0.4), (1.6, 0.6)]
    
    best_sum = 0
    best_circles = None
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    for width, height in rectangles:
        try:
            # Better initialization using hexagonal packing pattern inspired by Inspiration 1
            def initialize_hexagonal_pattern():
                circles = np.zeros((21, 3))
                
                # Create a hexagonal grid pattern
                rows = 5
                cols = 5
                
                # Calculate spacing based on area
                area = width * height
                avg_radius = np.sqrt(area / (np.pi * 21))
                
                # Hexagonal spacing
                spacing_x = avg_radius * 2.0 * 0.95  # Slight padding
                spacing_y = spacing_x * np.sqrt(3) / 2
                
                # Adjust spacing to fit within bounds
                actual_cols = int(width / spacing_x) + 2
                actual_rows = int(height / spacing_y) + 2
                
                if actual_cols * actual_rows < 21:
                    # If not enough space, reduce spacing
                    spacing_x = width / 6
                    spacing_y = height / 6
                    actual_cols = 6
                    actual_rows = 6
                
                # Place circles in hexagonal pattern
                idx = 0
                for i in range(actual_rows):
                    for j in range(actual_cols):
                        if idx >= 21:
                            break
                        x = (j + (i % 2) * 0.5) * spacing_x + spacing_x/2
                        y = i * spacing_y + spacing_y/2
                        
                        # Ensure within bounds
                        if 0 <= x <= width and 0 <= y <= height:
                            # Make sure we're not too close to boundaries
                            safe_radius = min(x, width-x, y, height-y) * 0.9
                            if safe_radius > 0:
                                circles[idx] = [x, y, min(safe_radius, avg_radius * 1.2)]
                                idx += 1
                    if idx >= 21:
                        break
                
                # Fill remaining with random positions
                for i in range(idx, 21):
                    x = np.random.uniform(0.01, width - 0.01)
                    y = np.random.uniform(0.01, height - 0.01)
                    # Estimate reasonable radius based on proximity to boundaries
                    safe_radius = min(x, width-x, y, height-y) * 0.8
                    circles[i] = [x, y, max(0.001, min(safe_radius, avg_radius * 0.8))]
                
                return circles
            
            # Objective function to maximize sum of radii
            def objective(params):
                # Reshape parameters into circles array
                circles = params.reshape(-1, 3)
                # Return negative sum of radii (since we want to maximize)
                return -np.sum(circles[:, 2])
            
            # Constraint function for non-overlapping
            def constraint_non_overlap(params):
                circles = params.reshape(-1, 3)
                # Vectorized overlap checking
                distances = cdist(circles[:, :2], circles[:, :2])
                radii_sum = circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]
                # For each pair, we want distance >= sum_radii, so we return (distance - sum_radii)
                constraints = distances - radii_sum
                # Only return the upper triangular part (avoid duplicates)
                mask = np.triu(np.ones_like(constraints), k=1).astype(bool)
                return constraints[mask]
            
            # Constraint function for boundary
            def constraint_boundary(params):
                circles = params.reshape(-1, 3)
                # For each circle, we want:
                # x - r >= 0, y - r >= 0, width - x - r >= 0, height - y - r >= 0
                constraints = np.column_stack([
                    circles[:, 0] - circles[:, 2],      # x - r >= 0
                    circles[:, 1] - circles[:, 2],      # y - r >= 0
                    width - circles[:, 0] - circles[:, 2],  # width - x - r >= 0
                    height - circles[:, 1] - circles[:, 2]   # height - y - r >= 0
                ])
                return constraints.flatten()
            
            # Generate initial configuration
            initial_circles = initialize_hexagonal_pattern()
            
            # Flatten for optimization
            initial_params = initial_circles.flatten()
            
            # Set up constraints
            cons = []
            
            # Non-overlap constraints (vectorized)
            cons.append({'type': 'ineq', 'fun': constraint_non_overlap})
            
            # Boundary constraints (vectorized)
            cons.append({'type': 'ineq', 'fun': constraint_boundary})
            
            # Bounds for optimization
            bounds = []
            for i in range(len(initial_params)):
                if i % 3 == 2:  # radius parameter
                    bounds.append((0.001, min(width, height) * 0.4))  # radius bound
                else:
                    bounds.append((0.001, width - 0.001 if i % 3 == 0 else height - 0.001))
            
            # Try multiple optimization approaches for better results (Inspiration 1 approach)
            best_method_sum = 0
            best_method_result = None
            
            # Approach 1: Differential Evolution for global optimization (Inspiration 1)
            try:
                de_result = differential_evolution(
                    objective, 
                    bounds, 
                    constraints=cons,
                    seed=42,
                    maxiter=250,
                    popsize=40,
                    tol=1e-9,
                    mutation=(0.5, 1),
                    recombination=0.7
                )
                
                if de_result.success:
                    current_sum = -de_result.fun
                    if current_sum > best_method_sum:
                        best_method_sum = current_sum
                        best_method_result = de_result
            except:
                pass
            
            # Approach 2: Trust-Constrained optimization with multiple restarts (Inspiration 1)
            if best_method_result is None:
                try:
                    # Multiple restarts with different initializations (Inspiration 1)
                    for restart in range(15):  # More restarts than before
                        # Perturb the initial solution slightly with larger variance
                        perturbed_params = initial_params + np.random.normal(0, 0.03, len(initial_params))
                        result = minimize(objective, perturbed_params, method='trust-constr', 
                                        bounds=bounds, constraints=cons, 
                                        options={'maxiter': 2500, 'ftol': 1e-10, 'gtol': 1e-10})
                        
                        if result.success:
                            optimized_circles = result.x.reshape(-1, 3)
                            current_sum = np.sum(optimized_circles[:, 2])
                            if current_sum > best_method_sum:
                                best_method_sum = current_sum
                                best_method_result = result
                except:
                    pass
            
            # Approach 3: SLSQP with multiple restarts (Inspiration 1)
            if best_method_result is None:
                try:
                    for restart in range(15):  # More restarts than before
                        # Perturb the initial solution slightly with larger variance
                        perturbed_params = initial_params + np.random.normal(0, 0.03, len(initial_params))
                        result = minimize(objective, perturbed_params, method='SLSQP', 
                                        bounds=bounds, constraints=cons, 
                                        options={'maxiter': 2500, 'ftol': 1e-10})
                        
                        if result.success:
                            optimized_circles = result.x.reshape(-1, 3)
                            current_sum = np.sum(optimized_circles[:, 2])
                            if current_sum > best_method_sum:
                                best_method_sum = current_sum
                                best_method_result = result
                except:
                    pass
            
            if best_method_result is not None and best_method_result.success:
                optimized_circles = best_method_result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
                    
        except Exception as e:
            continue
    
    # If we still haven't found a good solution, try a different approach (Inspiration 1 fallback)
    if best_circles is None:
        # Simple grid-based approach with better refinement (Inspiration 1 approach)
        width, height = 1.2, 0.8
        circles = np.zeros((21, 3))
        
        # Create a 5x5 grid with slight jittering
        rows, cols = 5, 5
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= 21:
                    break
                # Add jittering for better distribution
                x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                # Ensure within bounds
                x = np.clip(x, 0.01, width - 0.01)
                y = np.clip(y, 0.01, height - 0.01)
                # Radius based on proximity to boundaries
                safe_radius = min(x, width-x, y, height-y) * 0.7
                circles[len(circles)-21+i*cols+j] = [x, y, max(0.001, safe_radius)]
        
        # Trim to exactly 21
        circles = circles[:21]
        
        # Apply physics-based refinement with more aggressive approach (Inspiration 1)
        def physics_refinement(circles, width, height, iterations=250):
            circles = circles.copy()
            for iteration in range(iterations):
                # Simple physics-based relaxation
                forces = np.zeros_like(circles[:, :2])
                
                # Compute forces
                for i in range(len(circles)):
                    x, y, r = circles[i]
                    
                    # Boundary forces (Inspiration 1)
                    fx, fy = 0, 0
                    if x - r < 0.01:
                        fx += 500 * (r - x + 0.01)
                    if x + r > width - 0.01:
                        fx -= 500 * (x + r - (width - 0.01))
                    if y - r < 0.01:
                        fy += 500 * (r - y + 0.01)
                    if y + r > height - 0.01:
                        fy -= 500 * (y + r - (height - 0.01))
                    
                    # Circle-circle forces
                    for j in range(len(circles)):
                        if i != j:
                            x2, y2, r2 = circles[j]
                            dx = x - x2
                            dy = y - y2
                            dist = np.sqrt(dx*dx + dy*dy)
                            
                            if dist > 0 and dist < r + r2:
                                # Repulsion force - even stronger (Inspiration 1)
                                force_mag = 2000 * (r + r2 - dist) / (dist + 1e-8)
                                forces[i, 0] += dx / dist * force_mag
                                forces[i, 1] += dy / dist * force_mag
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                
                # Apply forces with damping - even higher force application rate (Inspiration 1)
                circles[:, :2] += forces * 0.04
                
                # Keep within bounds
                for i in range(len(circles)):
                    x, y, r = circles[i]
                    circles[i, 0] = np.clip(x, r, width - r)
                    circles[i, 1] = np.clip(y, r, height - r)
            
            return circles
        
        best_circles = physics_refinement(circles, width, height, 250)
    
    # Final validation and refinement (Inspiration 1 approach)
    if best_circles is not None:
        # Do one final check and minor refinement
        try:
            # Run a final optimization pass with highest precision (Inspiration 1)
            final_params = best_circles.flatten()
            bounds = []
            for i in range(len(final_params)):
                if i % 3 == 2:  # radius parameter
                    bounds.append((0.001, min(width, height) * 0.4))
                else:
                    bounds.append((0.001, width - 0.001 if i % 3 == 0 else height - 0.001))
            
            cons = []
            cons.append({'type': 'ineq', 'fun': constraint_non_overlap})
            cons.append({'type': 'ineq', 'fun': constraint_boundary})
            
            # Final trust-constr optimization with maximum precision (Inspiration 1)
            final_result = minimize(objective, final_params, method='trust-constr', 
                                  bounds=bounds, constraints=cons, 
                                  options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if final_result.success:
                validated_circles = final_result.x.reshape(-1, 3)
                validated_sum = np.sum(validated_circles[:, 2])
                if validated_sum > best_sum:
                    best_circles = validated_circles
        except:
            pass
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
