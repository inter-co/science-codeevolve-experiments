# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses advanced optimization techniques with improved initialization and constraint handling.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # More comprehensive rectangle dimension search
    best_sum = 0
    best_circles = None
    
    # Try various rectangle dimensions (aspect ratios)
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    
    # Use a more sophisticated approach with multiple restarts
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Create better initial configuration using hexagonal packing pattern
        def generate_initial_config():
            # Try to create a hexagonal-like arrangement
            circles = []
            n = 21
            
            # Determine grid layout (approximate hexagonal packing)
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            
            # Ensure we have enough space for all circles
            if rows * cols < n:
                rows += 1
                
            # Calculate spacing
            margin = 0.05
            cell_width = (width - 2*margin) / cols
            cell_height = (height - 2*margin) / rows
            
            # Adjust for hexagonal packing
            y_spacing = cell_height * np.sqrt(3) / 2
            x_spacing = cell_width * 0.75
            
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n:
                        break
                    # Offset every other row for hexagonal packing
                    offset = (i % 2) * (x_spacing / 2)
                    x = margin + j * x_spacing + offset + np.random.uniform(-0.01, 0.01)
                    y = margin + i * y_spacing + np.random.uniform(-0.01, 0.01)
                    
                    # Make sure we're within bounds
                    if x < margin or x > width - margin or y < margin or y > height - margin:
                        continue
                        
                    # Calculate maximum possible radius
                    max_radius = min(x, width - x, y, height - y)
                    # Start with a reasonable initial radius
                    radius = max(0.01, min(max_radius * 0.3, 0.3))
                    
                    circles.append([x, y, radius])
                    idx += 1
                    
                    if idx >= n:
                        break
                        
            # If we don't have enough circles, fill with random ones
            while len(circles) < n:
                x = np.random.uniform(margin, width - margin)
                y = np.random.uniform(margin, height - margin)
                max_radius = min(x, width - x, y, height - y)
                radius = max(0.01, min(max_radius * 0.3, 0.3))
                circles.append([x, y, radius])
                
            return np.array(circles[:n])
        
        # Improved constraint handling function
        def constraint_func(params):
            circles_flat = params.reshape(-1, 3)
            
            # Boundary constraints
            constraints = []
            for i in range(len(circles_flat)):
                x, y, r = circles_flat[i]
                constraints.append(x - r)  # Left boundary
                constraints.append(width - x - r)  # Right boundary
                constraints.append(y - r)  # Bottom boundary
                constraints.append(height - y - r)  # Top boundary
            
            # Overlap constraints (more efficient using vectorization)
            if len(circles_flat) > 1:
                # Vectorized distance computation
                coords = circles_flat[:, :2]
                radii = circles_flat[:, 2]
                
                # Compute pairwise distances
                dist_matrix = cdist(coords, coords)
                
                # Create overlap constraints for all pairs
                for i in range(len(circles_flat)):
                    for j in range(i+1, len(circles_flat)):
                        distance = dist_matrix[i, j]
                        radius_sum = radii[i] + radii[j]
                        constraints.append(distance - radius_sum)
            
            return np.array(constraints)
        
        # Objective function
        def objective(params):
            circles_flat = params.reshape(-1, 3)
            radii_sum = np.sum(circles_flat[:, 2])
            return -radii_sum  # Negative because we want to maximize
        
        # Generate better initial solution
        initial_circles = generate_initial_config()
        initial_params = initial_circles.flatten()
        
        # Try multiple optimization approaches
        try:
            # First, try L-BFGS-B with bounds (faster for local optimization)
            bounds = []
            for i in range(21):
                bounds.append((0.01, width - 0.01))  # x bounds
                bounds.append((0.01, height - 0.01))  # y bounds
                bounds.append((0.001, min(width, height)/2 - 0.01))  # radius bounds
            
            # Use SLSQP with constraints
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
            else:
                # Try differential evolution as backup
                try:
                    result_de = differential_evolution(
                        objective,
                        bounds,
                        constraints=[{'type': 'ineq', 'fun': constraint_func}],
                        seed=42,
                        maxiter=50,
                        popsize=10,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        atol=1e-6,
                        tol=1e-6
                    )
                    
                    if result_de.success:
                        optimized_circles = result_de.x.reshape(-1, 3)
                        current_sum = np.sum(optimized_circles[:, 2])
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_circles = optimized_circles.copy()
                except:
                    pass
                    
        except Exception as e:
            continue
    
    # If we still haven't found a good solution, use a more robust fallback
    if best_circles is None:
        # Use a completely different initialization approach - grid with randomization
        width, height = 1.2, 0.8  # Try a more balanced rectangle
        n = 21
        circles = np.zeros((n, 3))
        
        # Grid-based initialization with better spatial distribution
        rows = 5
        cols = 5
        if rows * cols < n:
            rows = 6
            cols = 4
            
        margin = 0.05
        cell_width = (width - 2*margin) / cols
        cell_height = (height - 2*margin) / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + j * cell_width + cell_width * 0.5
                y = margin + i * cell_height + cell_height * 0.5
                
                # Add structured randomness
                x += np.random.uniform(-cell_width*0.1, cell_width*0.1)
                y += np.random.uniform(-cell_height*0.1, cell_height*0.1)
                
                # Ensure we stay within bounds
                x = np.clip(x, margin, width - margin)
                y = np.clip(y, margin, height - margin)
                
                # Calculate maximum radius
                max_radius = min(x, width-x, y, height-y)
                radius = max_radius * 0.3
                
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Final refinement with optimization
        initial_params = circles.flatten()
        
        def objective_final(params):
            circles_flat = params.reshape(-1, 3)
            radii_sum = np.sum(circles_flat[:, 2])
            return -radii_sum
        
        def constraint_func_final(params):
            circles_flat = params.reshape(-1, 3)
            constraints = []
            for i in range(len(circles_flat)):
                x, y, r = circles_flat[i]
                constraints.append(x - r)
                constraints.append(width - x - r)
                constraints.append(y - r)
                constraints.append(height - y - r)
            
            # Vectorized overlap constraints
            if len(circles_flat) > 1:
                coords = circles_flat[:, :2]
                radii = circles_flat[:, 2]
                dist_matrix = cdist(coords, coords)
                for i in range(len(circles_flat)):
                    for j in range(i+1, len(circles_flat)):
                        distance = dist_matrix[i, j]
                        radius_sum = radii[i] + radii[j]
                        constraints.append(distance - radius_sum)
            
            return np.array(constraints)
        
        bounds = []
        for i in range(n):
            bounds.append((0.01, width - 0.01))
            bounds.append((0.01, height - 0.01))
            bounds.append((0.001, min(width, height)/2 - 0.01))
        
        try:
            result = minimize(
                objective_final,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func_final},
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                best_circles = result.x.reshape(-1, 3)
            else:
                best_circles = circles
        except:
            best_circles = circles
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
