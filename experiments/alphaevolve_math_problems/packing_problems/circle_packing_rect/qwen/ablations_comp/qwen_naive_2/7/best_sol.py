# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List
from deap import base, creator, tools, algorithms
import random
import time
from itertools import combinations

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining evolutionary algorithm with local optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    
    # Try different rectangle dimensions to find optimal aspect ratio
    best_result = None
    best_sum = 0
    
    # Test different width/height ratios with more focused search
    ratios = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    
    for ratio in ratios:
        width = 2.0 / (1.0 + ratio)
        height = ratio * width
        
        # Run optimization for this dimension
        result = optimize_for_dimensions(width, height, n)
        sum_radii = np.sum(result[:, 2])
        
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_result = result
            
        # Early stopping if we're getting close to benchmark
        if sum_radii > 2.36:
            break
    
    return best_result if best_result is not None else optimize_for_dimensions(1.0, 1.0, n)

def optimize_for_dimensions(width, height, n):
    """Optimize circle packing for given rectangle dimensions"""
    
    # Better initialization using hexagonal packing pattern with improved density
    def initialize_hexagonal_pattern():
        circles = []
        
        # For 21 circles, we'll use a more strategic approach
        # Try 5 rows and 4 columns for hexagonal packing
        rows = 5
        cols = 4
        
        # Calculate spacing based on rectangle dimensions
        cell_width = width / (cols + 1)
        cell_height = height / (rows + 1)
        
        # Use tighter hexagonal packing for better density
        hex_radius = min(cell_width, cell_height) * 0.35
        
        placed = 0
        for row in range(rows):
            for col in range(cols):
                if placed >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = 0 if row % 2 == 0 else hex_radius * 1.5
                x = (col + 1) * cell_width + x_offset
                y = (row + 1) * cell_height
                
                # Keep within bounds
                x = max(hex_radius, min(width - hex_radius, x))
                y = max(hex_radius, min(height - hex_radius, y))
                
                # Add more randomness to avoid perfect grid
                x += random.uniform(-hex_radius*0.15, hex_radius*0.15)
                y += random.uniform(-hex_radius*0.15, hex_radius*0.15)
                
                # Ensure within bounds after adjustment
                x = max(hex_radius, min(width - hex_radius, x))
                y = max(hex_radius, min(height - hex_radius, y))
                
                circles.append([x, y, hex_radius])
                placed += 1
                
            if placed >= n:
                break
        
        # Fill remaining slots with adaptive random positioning
        while len(circles) < n:
            # Try to place in areas that might be underutilized
            x = np.random.uniform(hex_radius, width - hex_radius)
            y = np.random.uniform(hex_radius, height - hex_radius)
            
            # Try to get a reasonable radius based on available space
            r = min(hex_radius * 0.8, (width - 2*hex_radius) / 10, (height - 2*hex_radius) / 10)
            r = np.random.uniform(0.01, r)
            
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # More efficient constraint evaluation using vectorization
    def evaluate_constraints(circles_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate boundary and overlap constraints efficiently.
        Returns:
            tuple: (boundary_constraints, overlap_constraints)
        """
        # Reshape for easier access
        x_vals = circles_flat[::3]
        y_vals = circles_flat[1::3] 
        r_vals = circles_flat[2::3]
        
        # Boundary constraints: each circle must be fully inside rectangle
        boundary_constraints = np.concatenate([
            x_vals - r_vals,                    # left boundary
            width - x_vals - r_vals,           # right boundary
            y_vals - r_vals,                    # bottom boundary
            height - y_vals - r_vals            # top boundary
        ])
        
        # Overlap constraints: distance between centers >= sum of radii
        positions = np.column_stack([x_vals, y_vals])
        
        # Vectorized overlap checking
        overlap_constraints = []
        
        # Precompute all pairs to avoid nested loops
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_radii = r_vals[i] + r_vals[j]
                overlap_constraints.append(dist - sum_radii)
        
        return boundary_constraints, np.array(overlap_constraints)
    
    # Objective function to maximize sum of radii
    def objective(circles_flat):
        # Return negative because we want to maximize (minimize negative)
        r_vals = circles_flat[2::3]
        return -np.sum(r_vals)
    
    # Constraint functions
    def boundary_constraint(circles_flat):
        bounds, _ = evaluate_constraints(circles_flat)
        return bounds
    
    def overlap_constraint(circles_flat):
        _, overlaps = evaluate_constraints(circles_flat)
        return overlaps
    
    # Initialize with better pattern
    circles = initialize_hexagonal_pattern()
    
    # First, try local optimization with more sophisticated approach
    try:
        # Flatten for optimization
        initial_guess = circles.flatten()
        
        # Bounds for each variable (x, y, r) - x and y in [0,width] and [0,height], r > 0
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, min(0.5, width/4, height/4))])
        
        # Constraints
        cons = [
            {'type': 'ineq', 'fun': boundary_constraint},
            {'type': 'ineq', 'fun': overlap_constraint}
        ]
        
        # Try multiple optimization methods
        methods = ['SLSQP', 'L-BFGS-B']
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8}
                )
                
                if result.success:
                    optimized_circles = result.x.reshape(-1, 3)
                    # Ensure all radii are positive and reasonable
                    optimized_circles[:, 2] = np.maximum(0.001, np.minimum(0.5, optimized_circles[:, 2]))
                    return optimized_circles
            except Exception:
                continue
                
    except Exception:
        pass
    
    # If local optimization fails, fall back to evolutionary approach with improved settings
    return evolutionary_optimization(width, height, n)

def evolutionary_optimization(width, height, n):
    """Use evolutionary algorithm to find good solution with improved parameters"""
    
    # Set up DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define gene ranges with better initial bounds
    def create_individual():
        individual = []
        for i in range(n):
            # x coordinate
            individual.append(random.uniform(0.001, width - 0.001))
            # y coordinate  
            individual.append(random.uniform(0.001, height - 0.001))
            # radius - use more appropriate initial values
            max_radius = min(0.5, width/8, height/8)
            individual.append(random.uniform(0.001, max_radius))
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_fitness(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Check constraints
        valid = True
        total_radius = 0
        
        # Boundary check
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r <= 0 or x + r >= width or y - r <= 0 or y + r >= height:
                valid = False
                break
            total_radius += r
        
        if not valid:
            return (-10000,)  # Penalty for invalid solutions
        
        # Overlap check - more efficient vectorized version
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute pairwise distances using vectorized operations
        # This is more efficient than nested loops
        distances = cdist(positions, positions)
        
        # Create mask for upper triangle only (avoid duplicate checks)
        mask = np.triu(np.ones((len(circles), len(circles)), dtype=bool), k=1)
        distances_masked = distances[mask]
        radii_pairs = np.array(list(combinations(radii, 2)))
        
        # Check if any overlaps exist
        sum_radii_pairs = np.sum(radii_pairs, axis=1)
        if np.any(distances_masked < sum_radii_pairs):
            return (-10000,)
        
        return (total_radius,)
    
    toolbox.register("evaluate", eval_fitness)
    toolbox.register("mate", tools.cxUniform, indpb=0.15)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    # Create initial population with better diversity
    pop = toolbox.population(n=100)
    
    # Run evolution with more generations and better parameters
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3, 
                                          ngen=100, stats=stats, halloffame=hof, 
                                          verbose=False)
    except Exception:
        # Fallback to simple approach if evolution fails
        pass
    
    # Return best solution found
    if len(hof) > 0:
        best_ind = hof[0]
        circles = np.array(best_ind).reshape(-1, 3)
        return circles
    
    # Fallback to initial grid pattern with better initialization
    return initialize_improved_grid_pattern(width, height, n)

def initialize_improved_grid_pattern(width, height, n):
    """Initialize with an improved grid pattern that's more likely to yield good results"""
    circles = []
    
    # Try to create a more balanced distribution
    # For 21 circles, try 3 rows × 7 columns or 7 rows × 3 columns
    cols = 7
    rows = 3
    
    cell_width = width / (cols + 1)
    cell_height = height / (rows + 1)
    
    # Start with a denser packing in the center
    placed = 0
    for row in range(rows):
        for col in range(cols):
            if placed >= n:
                break
                
            x = (col + 1) * cell_width
            y = (row + 1) * cell_height
            
            # Center the pattern and adjust for better utilization
            x = max(0.05, min(width - 0.05, x))
            y = max(0.05, min(height - 0.05, y))
            
            # Vary radii based on position and density
            # Circles near center can have larger radii
            center_x, center_y = width/2, height/2
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt((width/2)**2 + (height/2)**2)
            radius_factor = 1 - (dist_to_center / max_dist) * 0.5
            
            r = min(cell_width, cell_height) * 0.3 * radius_factor
            
            circles.append([x, y, r])
            placed += 1
            
        if placed >= n:
            break
    
    # Fill remaining slots with strategic random placement
    remaining = n - len(circles)
    for i in range(remaining):
        # Place in less crowded areas
        x = np.random.uniform(0.05, width - 0.05)
        y = np.random.uniform(0.05, height - 0.05)
        
        # Try to estimate a reasonable radius
        r = min(0.1, width/10, height/10)
        r = np.random.uniform(0.01, r)
        
        circles.append([x, y, r])
        
    return np.array(circles)


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
