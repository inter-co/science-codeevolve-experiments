# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses an enhanced evolutionary approach with multiple initialization strategies,
    adaptive mutation, and intensive local search.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    max_iterations = 350
    population_size = 100
    random_seed = 42
    
    # Set seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    def generate_voronoi_initialization():
        """Generate initial configuration using Voronoi diagram approach"""
        # Generate random points first
        points = np.random.rand(150, 2)
        
        # Ensure points are within bounds
        points[:, 0] = points[:, 0] * 0.8 + 0.1  # x in [0.1, 0.9]
        points[:, 1] = points[:, 1] * 0.8 + 0.1  # y in [0.1, 0.9]
        
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Get Voronoi vertices and filter valid ones
        valid_vertices = []
        for vertex in vor.vertices:
            if 0.1 <= vertex[0] <= 0.9 and 0.1 <= vertex[1] <= 0.9:
                valid_vertices.append(vertex)
        
        # Take first n points as circle centers
        centers = np.array(valid_vertices[:n]) if len(valid_vertices) >= n else np.random.rand(n, 2) * 0.8 + 0.1
        
        # Initialize radii based on Voronoi cell sizes
        radii = []
        for i in range(n):
            center = centers[i]
            min_distance = float('inf')
            
            # Find minimum distance to other centers
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(center - centers[j])
                    min_distance = min(min_distance, dist)
            
            # Set radius to half of minimum distance, but bounded
            max_radius = min(0.49, min_distance / 2.0)
            min_radius = 0.01
            radii.append(max(min_radius, min(max_radius, 0.16)))
        
        circles = np.column_stack([centers, radii])
        return circles
    
    def generate_hexagonal_initialization():
        """Generate hexagonal pattern initialization"""
        # Create a more structured hexagonal pattern
        circles = []
        
        # Create a grid pattern with hexagonal offsets
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) < n:
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y
                    
                    # Offset every other row for hexagonal packing
                    if i % 2 == 1:
                        x += spacing_x / 2
                    
                    # Add noise for better distribution
                    x += np.random.normal(0, 0.006)
                    y += np.random.normal(0, 0.006)
                    
                    # Keep within bounds
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    
                    positions.append([x, y])
        
        # Trim to exactly n positions
        positions = positions[:n]
        
        # Initialize radii with larger values for better starting point
        radii = [0.14] * n  # Start with even larger radii
        
        circles = np.array(positions)
        circles_with_radii = np.column_stack([circles, radii])
        return circles_with_radii
    
    def generate_random_initialization():
        """Generate random initialization with better constraints"""
        circles = np.random.rand(n, 3)
        circles[:, 0] = circles[:, 0] * 0.8 + 0.1  # x in [0.1, 0.9]
        circles[:, 1] = circles[:, 1] * 0.8 + 0.1  # y in [0.1, 0.9]
        circles[:, 2] = circles[:, 2] * 0.48 + 0.01  # r in [0.01, 0.49]
        return circles
    
    def compute_total_radius(circles):
        """Compute sum of all radii"""
        return np.sum(circles[:, 2])
    
    def is_valid_configuration(circles):
        """Check if configuration satisfies all constraints"""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check containment
        for i in range(len(circles)):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlaps using spatial indexing for efficiency
        distances = cdist(positions, positions)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if distances[i,j] < radii[i] + radii[j]:
                    return False
        return True
    
    def compute_overlap_penalty(circles):
        """Compute penalty based on overlap violations"""
        penalty = 0
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        distances = cdist(positions, positions)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                distance = distances[i,j]
                overlap = radii[i] + radii[j] - distance
                if overlap > 0:
                    penalty += overlap ** 2  # Quadratic penalty
        return penalty
    
    def enforce_boundaries(circles):
        """Enforce boundary constraints by adjusting positions and radii"""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        for i in range(len(circles)):
            x, y = positions[i]
            r = radii[i]
            
            # Adjust radius to fit within boundaries
            r_min = min(x, 1-x, y, 1-y)
            if r > r_min:
                r = r_min * 0.99  # Leave some margin
            
            # Adjust position if needed
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            positions[i] = [x, y]
            radii[i] = r
        
        circles[:, :2] = positions
        circles[:, 2] = radii
        return circles
    
    def mutate_individual(circles, generation, mutation_rate=0.15):
        """Apply mutation to a single individual with adaptive rate"""
        mutated = circles.copy()
        
        # Adaptive mutation rate - decrease over generations
        adaptive_rate = mutation_rate * (1.0 - generation / max_iterations)
        adaptive_rate = max(0.05, adaptive_rate)
        
        # Randomly select which circles to mutate
        indices = np.random.choice(len(circles), size=int(len(circles) * adaptive_rate), replace=False)
        
        for idx in indices:
            # Different mutation strategies based on circle index
            if idx % 3 == 0:
                # Stronger position mutation for some circles
                mutated[idx, 0] += np.random.normal(0, 0.022)
                mutated[idx, 1] += np.random.normal(0, 0.022)
                mutated[idx, 2] += np.random.normal(0, 0.013)
            elif idx % 3 == 1:
                # Moderate position mutation
                mutated[idx, 0] += np.random.normal(0, 0.016)
                mutated[idx, 1] += np.random.normal(0, 0.016)
                mutated[idx, 2] += np.random.normal(0, 0.009)
            else:
                # Lighter mutation
                mutated[idx, 0] += np.random.normal(0, 0.011)
                mutated[idx, 1] += np.random.normal(0, 0.011)
                mutated[idx, 2] += np.random.normal(0, 0.006)
            
            # Keep within bounds
            mutated[idx, 0] = max(0.01, min(0.99, mutated[idx, 0]))
            mutated[idx, 1] = max(0.01, min(0.99, mutated[idx, 1]))
            mutated[idx, 2] = max(0.001, min(0.49, mutated[idx, 2]))
        
        return mutated
    
    def evaluate_fitness(circles):
        """Evaluate fitness of a configuration"""
        if not is_valid_configuration(circles):
            # Return very low fitness for invalid configurations
            return -compute_overlap_penalty(circles)
        
        # Higher fitness = higher sum of radii
        total_radius = compute_total_radius(circles)
        return total_radius
    
    def local_improvement(circles, max_attempts=250):
        """Perform local improvement on a configuration"""
        current = circles.copy()
        best = current.copy()
        best_fitness = evaluate_fitness(best)
        
        # Track consecutive improvements to detect stagnation
        consecutive_no_improvements = 0
        max_consecutive_no_improvements = 40
        
        for attempt in range(max_attempts):
            # Make small random changes to one circle at a time
            idx = np.random.randint(0, n)
            
            # Save current state
            old_pos = current[idx, :2].copy()
            old_rad = current[idx, 2]
            
            # Try small perturbations
            new_pos = old_pos + np.random.normal(0, 0.007, 2)
            new_rad = old_rad + np.random.normal(0, 0.0025)
            
            # Clamp to valid ranges
            new_pos[0] = max(0.01, min(0.99, new_pos[0]))
            new_pos[1] = max(0.01, min(0.99, new_pos[1]))
            new_rad = max(0.001, min(0.49, new_rad))
            
            # Test if this improves the configuration
            test_solution = current.copy()
            test_solution[idx, :2] = new_pos
            test_solution[idx, 2] = new_rad
            
            if is_valid_configuration(test_solution):
                test_fitness = compute_total_radius(test_solution)
                if test_fitness > best_fitness:
                    current = test_solution
                    best = test_solution.copy()
                    best_fitness = test_fitness
                    consecutive_no_improvements = 0
                else:
                    consecutive_no_improvements += 1
                    # If no improvement for a while, try a more aggressive approach
                    if consecutive_no_improvements > max_consecutive_no_improvements:
                        # Try a more aggressive mutation
                        new_pos = old_pos + np.random.normal(0, 0.012, 2)
                        new_rad = old_rad + np.random.normal(0, 0.006, 1)
                        new_pos[0] = max(0.01, min(0.99, new_pos[0]))
                        new_pos[1] = max(0.01, min(0.99, new_pos[1]))
                        new_rad = max(0.001, min(0.49, new_rad[0]))
                        
                        test_solution = current.copy()
                        test_solution[idx, :2] = new_pos
                        test_solution[idx, 2] = new_rad
                        
                        if is_valid_configuration(test_solution):
                            test_fitness = compute_total_radius(test_solution)
                            if test_fitness > best_fitness:
                                current = test_solution
                                best = test_solution.copy()
                                best_fitness = test_fitness
                                consecutive_no_improvements = 0
            else:
                # Even if invalid, try to fix it
                current = enforce_boundaries(current)
            
            # Early stopping if we haven't improved in a while
            if consecutive_no_improvements > max_consecutive_no_improvements * 2:
                break
        
        return best
    
    # Initialize population using multiple strategies
    population = []
    
    # Generate multiple initial configurations with better diversity
    for i in range(population_size):
        if i % 4 == 0:
            # Voronoi-based initialization
            init_config = generate_voronoi_initialization()
        elif i % 4 == 1:
            # Hexagonal pattern
            init_config = generate_hexagonal_initialization()
        elif i % 4 == 2:
            # Another hexagonal pattern with more variation
            init_config = generate_hexagonal_initialization()
            # Slightly perturb it
            for j in range(n):
                init_config[j, 0] += np.random.normal(0, 0.012)
                init_config[j, 1] += np.random.normal(0, 0.012)
                init_config[j, 2] += np.random.normal(0, 0.006)
        else:
            # Random initialization with different parameters
            init_config = generate_random_initialization()
            # Make it a bit more structured
            for j in range(n):
                init_config[j, 2] = max(0.02, min(0.45, init_config[j, 2] * 1.25))
        
        # Enforce boundaries
        init_config = enforce_boundaries(init_config)
        
        population.append(init_config)
    
    # Main evolutionary loop
    best_fitness = -float('inf')
    best_solution = None
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > 55:  # Leave 5 seconds for final validation
            break
            
        # Evaluate fitness of entire population
        fitness_scores = []
        for individual in population:
            score = evaluate_fitness(individual)
            fitness_scores.append(score)
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Update best solution
        if fitness_scores[0] > best_fitness:
            best_fitness = fitness_scores[0]
            best_solution = population[0].copy()
        
        # Selection: keep top 35% and apply elitism
        selected_count = max(10, population_size // 3)
        selected_population = population[:selected_count]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        # Add mutated versions of selected individuals
        for i in range(selected_count, population_size):
            parent = selected_population[np.random.randint(0, selected_count)]
            child = mutate_individual(parent, iteration)
            new_population.append(child)
        
        # Apply local improvement to some individuals for better convergence
        if iteration % 12 == 0 and iteration > 0:
            for i in range(0, min(15, len(new_population)), 2):
                new_population[i] = local_improvement(new_population[i])
        
        population = new_population
    
    # Final validation and refinement
    if best_solution is not None:
        # Apply intensive local refinement with more aggressive search
        refined_solution = best_solution.copy()
        
        # Multiple rounds of local improvement with early stopping
        for round_num in range(6):
            improved = False
            for _ in range(250):
                # Try to improve with more aggressive local search
                idx = np.random.randint(0, n)
                
                # Save current state
                old_pos = refined_solution[idx, :2].copy()
                old_rad = refined_solution[idx, 2]
                
                # Try larger perturbations in later stages
                perturbation_scale = 0.018 if round_num < 4 else 0.025
                new_pos = old_pos + np.random.normal(0, perturbation_scale, 2)
                new_rad = old_rad + np.random.normal(0, 0.004 if round_num < 4 else 0.008, 1)
                
                # Clamp to valid ranges
                new_pos[0] = max(0.01, min(0.99, new_pos[0]))
                new_pos[1] = max(0.01, min(0.99, new_pos[1]))
                new_rad = max(0.001, min(0.49, new_rad[0]))
                
                # Test if this improves the configuration
                test_solution = refined_solution.copy()
                test_solution[idx, :2] = new_pos
                test_solution[idx, 2] = new_rad
                
                if is_valid_configuration(test_solution):
                    test_fitness = compute_total_radius(test_solution)
                    current_fitness = compute_total_radius(refined_solution)
                    
                    if test_fitness > current_fitness:
                        refined_solution = test_solution
                        improved = True
                        
            if not improved:
                break
        
        # Final validation
        if is_valid_configuration(refined_solution):
            return refined_solution
        else:
            # If validation fails, return the best valid configuration found so far
            return best_solution
    else:
        # Return the best initialization if no evolution occurred
        return population[0] if population else generate_hexagonal_initialization()


# EVOLVE-BLOCK-END
