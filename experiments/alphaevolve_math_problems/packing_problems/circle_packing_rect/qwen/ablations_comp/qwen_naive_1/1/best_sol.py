# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining advanced optimization techniques with smart initialization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle dimensions: width + height = 2, optimize this ratio
    # Start with square rectangle (good starting point)
    width, height = 1.0, 1.0
    
    def generate_hexagonal_grid(width: float, height: float, n_circles: int) -> np.ndarray:
        """Generate initial configuration using hexagonal packing"""
        # Try to create a hexagonal pattern for better packing density
        rows = int(np.sqrt(n_circles))
        cols = int(np.ceil(n_circles / rows))
        
        # Adjust dimensions to accommodate the grid
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        circles = []
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n_circles:
                    break
                # Hexagonal offset
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + 0.5 + x_offset) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Ensure within bounds
                if 0 <= x <= width and 0 <= y <= height:
                    # Initial radius based on distance to edges and spacing
                    max_r = min(x, width - x, y, height - y)
                    r = max_r * np.random.uniform(0.1, 0.3)
                    circles.append([x, y, r])
        
        # Fill remaining spots
        while len(circles) < n_circles:
            x = np.random.uniform(0.05, width - 0.05)
            y = np.random.uniform(0.05, height - 0.05)
            max_r = min(x, width - x, y, height - y)
            r = max_r * np.random.uniform(0.05, 0.15)
            circles.append([x, y, r])
        
        return np.array(circles[:n_circles])
    
    def generate_fibonacci_spiral(width: float, height: float, n_circles: int) -> np.ndarray:
        """Generate initial configuration using Fibonacci spiral for even distribution"""
        circles = []
        
        # Golden angle in radians
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n_circles):
            # Map circle index to spiral coordinates
            radius = np.sqrt(i / (n_circles - 1)) if n_circles > 1 else 0.0
            angle = i * golden_angle
            
            # Convert to Cartesian coordinates in [0,1] range
            x_norm = 0.5 + radius * np.cos(angle) * 0.4
            y_norm = 0.5 + radius * np.sin(angle) * 0.4
            
            # Scale to rectangle dimensions
            x = x_norm * width
            y = y_norm * height
            
            # Compute appropriate radius based on distance to edges
            max_r = min(x, width - x, y, height - y)
            r = max_r * np.random.uniform(0.05, 0.2)
            
            circles.append([x, y, r])
        
        return np.array(circles)
    
    def generate_better_initialization(width: float, height: float, n_circles: int) -> np.ndarray:
        """Generate a better initial configuration using a combination of strategies"""
        circles = []
        
        # Strategy 1: Place some circles at corners to utilize space efficiently
        corner_positions = [
            (0.1, 0.1), (width - 0.1, 0.1), 
            (0.1, height - 0.1), (width - 0.1, height - 0.1)
        ]
        
        for i, (x, y) in enumerate(corner_positions):
            if i < n_circles:
                max_r = min(x, width - x, y, height - y)
                r = max_r * np.random.uniform(0.15, 0.25)
                circles.append([x, y, r])
        
        # Strategy 2: Fill remaining spots with Fibonacci spiral
        remaining = n_circles - len(circles)
        if remaining > 0:
            fib_circles = generate_fibonacci_spiral(width, height, remaining)
            circles.extend(fib_circles)
        
        # Strategy 3: Add some random circles in center area for diversity
        additional = max(0, n_circles - len(circles))
        for _ in range(additional):
            x = np.random.uniform(width * 0.2, width * 0.8)
            y = np.random.uniform(height * 0.2, height * 0.8)
            max_r = min(x, width - x, y, height - y)
            r = max_r * np.random.uniform(0.05, 0.15)
            circles.append([x, y, r])
        
        # Ensure exact count
        return np.array(circles[:n_circles])
    
    def generate_concentric_circles(width: float, height: float, n_circles: int) -> np.ndarray:
        """Generate initial configuration using concentric circles for dense packing"""
        circles = []
        
        # Center of rectangle
        cx, cy = width / 2, height / 2
        
        # Start with largest possible circles near center and work outward
        max_radius = min(width, height) * 0.3
        
        # Distribute circles in concentric rings
        ring_count = max(1, int(np.sqrt(n_circles)))
        ring_spacing = max_radius / ring_count
        
        for ring in range(ring_count):
            circles_per_ring = max(1, n_circles // ring_count)
            radius = (ring + 1) * ring_spacing * 0.8  # Slightly smaller than spacing
            
            # Place circles around the ring
            for i in range(circles_per_ring):
                angle = 2 * np.pi * i / circles_per_ring
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                
                # Check bounds
                if 0 <= x <= width and 0 <= y <= height:
                    # Radius based on distance to edges
                    max_r = min(x, width - x, y, height - y)
                    actual_r = min(radius, max_r * 0.9)  # Slightly smaller for safety
                    circles.append([x, y, actual_r])
                    
                if len(circles) >= n_circles:
                    break
            if len(circles) >= n_circles:
                break
        
        # Fill remaining with random positions
        while len(circles) < n_circles:
            x = np.random.uniform(0.05, width - 0.05)
            y = np.random.uniform(0.05, height - 0.05)
            max_r = min(x, width - x, y, height - y)
            r = max_r * np.random.uniform(0.05, 0.15)
            circles.append([x, y, r])
        
        return np.array(circles[:n_circles])
    
    def calculate_overlap_penalty(circles: np.ndarray) -> float:
        """Calculate penalty for overlapping circles"""
        penalty = 0.0
        n_circles = len(circles)
        
        # Use efficient vectorized computation for overlap checking
        if n_circles < 2:
            return penalty
            
        # Create distance matrix for all pairs
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        # Vectorized calculation of distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Create matrix of sum of radii for all pairs
        radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Find overlapping pairs
        overlap_matrix = distances < radii_matrix
        np.fill_diagonal(overlap_matrix, False)  # No self-overlap
        
        # Calculate overlap penalties
        overlaps = radii_matrix[overlap_matrix] - distances[overlap_matrix]
        penalty = np.sum(overlaps) * 1000.0
        
        return penalty
    
    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness of a circle configuration"""
        # Calculate sum of radii (this is what we want to maximize)
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for violations
        penalty = 0.0
        
        # Boundary penalties - stronger penalty for boundary violations
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                penalty += 100000.0  # Much higher penalty
        
        # Overlap penalties
        penalty += calculate_overlap_penalty(circles)
        
        return total_radius - penalty
    
    def mutate_individual(individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Mutate an individual by slightly adjusting positions and radii"""
        mutated = individual.copy()
        mutation_rate = 0.3 + (0.7 * (1 - generation/max_generations))  # Decrease over time
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Randomly adjust position or radius
                if random.random() < 0.7:
                    # Adjust position with larger steps for exploration early, smaller later
                    step_size = width * 0.1 * (1 - generation/max_generations)
                    mutated[i][0] += np.random.normal(0, step_size)
                    mutated[i][1] += np.random.normal(0, step_size)
                else:
                    # Adjust radius with more aggressive scaling
                    mutated[i][2] *= np.random.normal(1.0, 0.2 * (1 - generation/max_generations))
        
        # Keep within bounds
        for i in range(len(mutated)):
            mutated[i][0] = np.clip(mutated[i][0], 0.05, width - 0.05)
            mutated[i][1] = np.clip(mutated[i][1], 0.05, height - 0.05)
            mutated[i][2] = np.clip(mutated[i][2], 0.001, min(width, height) * 0.4)
        
        return mutated
    
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring from two parents"""
        # Uniform crossover with preference for better parent traits
        offspring = parent1.copy()
        
        for i in range(len(offspring)):
            if random.random() < 0.5:
                offspring[i] = parent2[i].copy()
        
        # Add some variance
        for i in range(len(offspring)):
            if random.random() < 0.3:
                offspring[i][0] += np.random.normal(0, width * 0.03)
                offspring[i][1] += np.random.normal(0, height * 0.03)
                offspring[i][2] *= np.random.normal(1.0, 0.1)
        
        # Keep within bounds
        for i in range(len(offspring)):
            offspring[i][0] = np.clip(offspring[i][0], 0.05, width - 0.05)
            offspring[i][1] = np.clip(offspring[i][1], 0.05, height - 0.05)
            offspring[i][2] = np.clip(offspring[i][2], 0.001, min(width, height) * 0.4)
        
        return offspring
    
    def optimize_with_local_search(circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """Use local optimization to refine the solution"""
        # Flatten for optimization
        flat_initial = circles.flatten()
        
        def objective(flat_circles):
            # Convert back to circles array
            local_circles = flat_circles.reshape(-1, 3)
            # Return negative sum of radii (we minimize)
            return -np.sum(local_circles[:, 2])
        
        def boundary_constraint(flat_circles):
            local_circles = flat_circles.reshape(-1, 3)
            constraints = []
            for i in range(len(local_circles)):
                x, y, r = local_circles[i]
                # Circle must fit within rectangle
                constraints.extend([
                    x - r,           # x - r >= 0
                    width - x - r,   # width - x - r >= 0
                    y - r,           # y - r >= 0
                    height - y - r   # height - y - r >= 0
                ])
            return np.array(constraints)
        
        def overlap_constraint(flat_circles):
            local_circles = flat_circles.reshape(-1, 3)
            constraints = []
            for i in range(len(local_circles)):
                for j in range(i+1, len(local_circles)):
                    x1, y1, r1 = local_circles[i]
                    x2, y2, r2 = local_circles[j]
                    distance_sq = (x2-x1)**2 + (y2-y1)**2
                    min_distance_sq = (r1 + r2)**2
                    # We want distance^2 >= min_distance^2, so we enforce: distance^2 - min_distance^2 >= 0
                    constraints.append(distance_sq - min_distance_sq)
            return np.array(constraints)
        
        # Set up bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.01, width - 0.01), (0.01, height - 0.01), (0.001, min(width, height)/2)])
        
        try:
            result = minimize(
                objective,
                flat_initial,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: boundary_constraint(x)},
                    {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
                ],
                options={'maxiter': max_iter, 'ftol': 1e-6}
            )
            
            if result.success:
                return result.x.reshape(-1, 3)
        except:
            pass
        
        return circles
    
    def adaptive_evolution():
        """Run adaptive evolutionary optimization"""
        pop_size = 70  # Increased population size
        generations = 200  # More generations
        elite_size = 15  # More elite individuals
        
        # Initialize population with better strategies
        population = []
        for _ in range(pop_size):
            # Mix of initialization strategies
            strategy = random.choice(['fibonacci', 'hexagonal', 'hybrid', 'concentric'])
            if strategy == 'fibonacci':
                circles = generate_fibonacci_spiral(width, height, n)
            elif strategy == 'hexagonal':
                circles = generate_hexagonal_grid(width, height, n)
            elif strategy == 'concentric':
                circles = generate_concentric_circles(width, height, n)
            else:  # hybrid
                circles = generate_better_initialization(width, height, n)
            population.append(circles)
        
        population = np.array(population)
        
        best_fitness = float('-inf')
        best_solution = None
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                score = evaluate_fitness(individual)
                fitness_scores.append(score)
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = population[sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Track best solution
            if fitness_scores[0] > best_fitness:
                best_fitness = fitness_scores[0]
                best_solution = population[0].copy()
            
            # Create next generation
            new_population = []
            
            # Elite preservation
            new_population.extend(population[:elite_size])
            
            # Generate offspring through crossover and mutation
            while len(new_population) < pop_size:
                # Tournament selection with better probability for top individuals
                parent1_idx = random.randint(0, elite_size - 1)
                parent2_idx = random.randint(0, elite_size - 1)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                offspring = crossover(parent1, parent2)
                mutated_offspring = mutate_individual(offspring, generation, generations)
                
                new_population.append(mutated_offspring)
            
            population = np.array(new_population)
            
            # Print progress every 20 generations
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        return best_solution, best_fitness
    
    def improved_global_optimization():
        """Try improved global optimization approach"""
        # Try several different starting configurations and use global optimization
        best_result = None
        best_score = float('-inf')
        
        # Try different initialization approaches with varying parameters
        for attempt in range(15):  # More attempts
            # Different random seed for variety
            random.seed(attempt * 1000)
            np.random.seed(attempt * 1000)
            
            # Try with different rectangle ratios
            ratio = 0.5 + np.random.uniform(-0.5, 0.5)  # Broader range for optimization
            w = 1.0
            h = 2.0 / (1.0 + ratio)
            width, height = w, h
            
            # Generate initial solution with better initialization
            strategy = random.choice(['fibonacci', 'hexagonal', 'hybrid', 'concentric'])
            if strategy == 'fibonacci':
                circles = generate_fibonacci_spiral(width, height, n)
            elif strategy == 'hexagonal':
                circles = generate_hexagonal_grid(width, height, n)
            elif strategy == 'concentric':
                circles = generate_concentric_circles(width, height, n)
            else:  # hybrid
                circles = generate_better_initialization(width, height, n)
            
            # Refine with local search
            refined = optimize_with_local_search(circles, max_iter=500)
            score = evaluate_fitness(refined)
            
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        
        return best_result
    
    def multi_strategy_approach():
        """Use multiple strategies to find the best solution"""
        solutions = []
        scores = []
        
        # Strategy 1: Evolutionary algorithm (enhanced)
        try:
            evolved_solution, evolved_score = adaptive_evolution()
            if evolved_solution is not None:
                solutions.append(evolved_solution)
                scores.append(evolved_score)
        except Exception as e:
            pass
        
        # Strategy 2: Global optimization (enhanced)
        try:
            global_solution = improved_global_optimization()
            if global_solution is not None:
                global_score = evaluate_fitness(global_solution)
                solutions.append(global_solution)
                scores.append(global_score)
        except Exception as e:
            pass
        
        # Strategy 3: Heuristic solution with refinement (enhanced)
        try:
            heuristic_solution = generate_better_initialization(width, height, n)
            refined_heuristic = optimize_with_local_search(heuristic_solution, max_iter=1000)
            heuristic_score = evaluate_fitness(refined_heuristic)
            solutions.append(refined_heuristic)
            scores.append(heuristic_score)
        except Exception as e:
            pass
        
        # Strategy 4: Concentric circle approach
        try:
            concentric_solution = generate_concentric_circles(width, height, n)
            refined_concentric = optimize_with_local_search(concentric_solution, max_iter=500)
            concentric_score = evaluate_fitness(refined_concentric)
            solutions.append(refined_concentric)
            scores.append(concentric_score)
        except Exception as e:
            pass
        
        # Strategy 5: Multiple random starts with local search
        for i in range(8):  # More attempts
            try:
                random.seed(i * 1000)
                np.random.seed(i * 1000)
                random_solution = generate_better_initialization(width, height, n)
                refined_random = optimize_with_local_search(random_solution, max_iter=500)
                random_score = evaluate_fitness(refined_random)
                solutions.append(refined_random)
                scores.append(random_score)
            except Exception as e:
                continue
        
        # Select best solution
        if solutions:
            best_idx = np.argmax(scores)
            return solutions[best_idx]
        else:
            # Fallback to default
            return generate_better_initialization(width, height, n)
    
    # Run optimization with fallbacks
    start_time = time.time()
    
    # Try the improved multi-strategy approach first
    try:
        final_solution = multi_strategy_approach()
    except Exception as e:
        # Fallback to simple approach
        final_solution = generate_better_initialization(width, height, n)
        final_solution = optimize_with_local_search(final_solution)
    
    # Final validation and refinement
    if final_solution is not None:
        # Apply final refinement
        final_solution = optimize_with_local_search(final_solution, max_iter=1000)
        
        # Validate constraints one more time
        penalty = calculate_overlap_penalty(final_solution)
        if penalty > 1000:  # Significant overlap
            # Try to reinitialize with better approach
            circles = generate_better_initialization(width, height, n)
            final_solution = optimize_with_local_search(circles, max_iter=1000)
    
    # Final check - if still no solution, return a default
    if final_solution is None:
        final_solution = generate_better_initialization(width, height, n)
        final_solution = optimize_with_local_search(final_solution)
    
    eval_time = time.time() - start_time
    return final_solution


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
