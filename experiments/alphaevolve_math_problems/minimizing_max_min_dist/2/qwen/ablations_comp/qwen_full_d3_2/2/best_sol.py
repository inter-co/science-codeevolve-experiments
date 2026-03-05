# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining constructive geometry and advanced optimization techniques.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # More sophisticated constructive approach using hexagonal lattice
    def construct_hexagonal_lattice():
        """Construct points using a hexagonal lattice pattern - known to be efficient"""
        points = []
        # Create a hexagonal grid with 4 rows and 4 columns
        for i in range(4):
            for j in range(4):
                # Offset every other row
                x_offset = 0.25 if i % 2 == 1 else 0
                x = 0.125 + j * 0.25 + x_offset
                y = 0.125 + i * 0.25
                points.append([x, y])
        return np.array(points[:16])  # Ensure exactly 16 points
    
    # Constructive approach using golden ratio spiral
    def construct_golden_spiral():
        """Construct points using golden ratio spiral for excellent distribution"""
        n = 16
        points = []
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Generate points along a spiral
        for i in range(n):
            # Angle based on golden ratio
            angle = 2 * np.pi * i / phi
            # Radial position with logarithmic scaling
            r = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            points.append([x, y])
            
        return np.array(points)
    
    # Constructive approach using icosahedral symmetry
    def construct_icosahedral():
        """Construct points approximating icosahedral symmetry"""
        # Use vertices of regular icosahedron projected to sphere, then to square
        # Vertices of icosahedron scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        # Normalize to unit sphere and project to 2D
        points = []
        for vertex in vertices:
            # Normalize to unit sphere
            norm = np.sqrt(sum(v**2 for v in vertex))
            if norm > 0:
                vertex = [v/norm for v in vertex]
                # Project to 2D (stereographic projection or simple drop)
                x = 0.5 + 0.4 * vertex[0]
                y = 0.5 + 0.4 * vertex[1]
                points.append([x, y])
        
        # Fill to 16 points with random points near the center
        while len(points) < 16:
            x = 0.5 + 0.1 * (random.random() - 0.5)
            y = 0.5 + 0.1 * (random.random() - 0.5)
            points.append([x, y])
            
        return np.array(points[:16])
    
    # Improved evolutionary algorithm with better operators
    def improved_evolutionary_optimization():
        """Enhanced genetic algorithm with better operators and selection"""
        n = 16
        population_size = 30
        generations = 50
        mutation_rate = 0.15
        
        # Initialize with better starting points
        def create_individual():
            # Mix multiple good construction methods
            choice = random.randint(0, 2)
            if choice == 0:
                points = construct_hexagonal_lattice()
            elif choice == 1:
                points = construct_golden_spiral()
            else:
                points = construct_icosahedral()
            # Add small noise
            noise = np.random.normal(0, 0.015, points.shape)
            return np.clip(points + noise, 0, 1)
        
        # Fitness function with better numerical stability
        def fitness(individual):
            # Ensure points are within bounds
            individual = np.clip(individual, 0, 1)
            distances = pdist(individual)
            
            if len(distances) == 0 or np.max(distances) == 0:
                return -np.inf
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 1e-10:
                return -np.inf
                
            # Use log transform to stabilize the ratio calculation
            return min_dist / max_dist
        
        # Create initial population
        population = [create_individual() for _ in range(population_size)]
        
        # Evolution loop with better selection and reproduction
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness(individual) for individual in population]
            
            # Tournament selection
            def select_parent():
                tournament_size = 3
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                return population[winner_index].copy()
            
            # Create new population
            new_population = []
            
            # Elitism - keep top 3 individuals
            elite_indices = np.argsort(fitness_scores)[-3:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                parent1 = select_parent()
                parent2 = select_parent()
                
                # Blend crossover with weighted average
                alpha = random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Add mutation with adaptive strength
                if random.random() < mutation_rate:
                    # Adaptive mutation strength based on generation
                    mutation_strength = 0.02 + 0.01 * (generation / generations)
                    noise = np.random.normal(0, mutation_strength, child.shape)
                    child += noise
                    child = np.clip(child, 0, 1)
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        # Return best individual
        final_fitnesses = [fitness(individual) for individual in population]
        best_index = np.argmax(final_fitnesses)
        return population[best_index]
    
    # Enhanced simulated annealing with better cooling schedule
    def enhanced_simulated_annealing():
        """Improved simulated annealing with better cooling and neighborhood moves"""
        n = 16
        
        # Start with a good construction
        current_points = construct_hexagonal_lattice()
        current_points += np.random.normal(0, 0.01, current_points.shape)
        current_points = np.clip(current_points, 0, 1)
        
        def calculate_ratio(points):
            distances = pdist(points)
            if len(distances) == 0 or np.max(distances) == 0:
                return 0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            return min_dist / max_dist if max_dist > 0 else 0
        
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Better cooling schedule
        temperature = 1.0
        cooling_rate = 0.998  # Slower cooling for better exploration
        min_temperature = 1e-6
        steps_per_temp = 50
        
        while temperature > min_temperature:
            for _ in range(steps_per_temp):
                # Create neighbor solution with multiple strategies
                neighbor_points = current_points.copy()
                
                # Choose move type with probabilities
                move_type = random.choices(['single', 'pair', 'cluster'], weights=[0.5, 0.3, 0.2])[0]
                
                if move_type == 'single':
                    # Move single point
                    idx = random.randint(0, n - 1)
                    neighbor_points[idx] += np.random.normal(0, 0.005, 2)
                elif move_type == 'pair':
                    # Move two nearby points
                    idx1, idx2 = random.sample(range(n), 2)
                    delta = np.random.normal(0, 0.008, 2)
                    neighbor_points[idx1] += delta
                    neighbor_points[idx2] += delta
                else:  # cluster
                    # Move a group of points
                    indices = random.sample(range(n), random.randint(2, 5))
                    delta = np.random.normal(0, 0.01, 2)
                    for idx in indices:
                        neighbor_points[idx] += delta
                
                # Clip to bounds
                neighbor_points = np.clip(neighbor_points, 0, 1)
                
                neighbor_ratio = calculate_ratio(neighbor_points)
                
                # Accept or reject based on Metropolis criterion
                if neighbor_ratio > current_ratio:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
                    if current_ratio > best_ratio:
                        best_points = neighbor_points.copy()
                        best_ratio = current_ratio
                else:
                    # Accept with probability based on temperature and difference
                    delta = neighbor_ratio - current_ratio
                    if random.random() < np.exp(delta / temperature):
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio
            
            temperature *= cooling_rate
        
        return best_points
    
    # Multi-start optimization with better constraint handling
    def multi_start_optimization():
        """Run multiple local optimizations from different starting points"""
        best_points = None
        best_ratio = -np.inf
        
        # Different initialization strategies
        init_strategies = [
            lambda: construct_hexagonal_lattice(),
            lambda: construct_golden_spiral(), 
            lambda: construct_icosahedral(),
            lambda: np.random.rand(16, 2)
        ]
        
        # Run optimization from each starting point
        for i, strategy in enumerate(init_strategies):
            try:
                # Create initial points
                initial_points = strategy()
                initial_points += np.random.normal(0, 0.01, initial_points.shape)
                initial_points = np.clip(initial_points, 0, 1)
                
                # Define objective function for scipy
                def objective(x):
                    points = x.reshape(-1, 2)
                    distances = pdist(points)
                    if len(distances) == 0 or np.max(distances) == 0:
                        return 0
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    return -min_dist / max_dist if max_dist > 0 else 0
                
                # Optimize with L-BFGS-B (good for bounded problems)
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8},
                    tol=1e-8
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    distances = pdist(optimized_points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
            except Exception:
                continue
                
        return best_points if best_points is not None else np.random.rand(16, 2)
    
    # Try multiple constructive approaches first with better error handling
    strategies = [
        construct_hexagonal_lattice,
        construct_golden_spiral,
        construct_icosahedral
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Try each constructive approach
    for strategy in strategies:
        try:
            points = strategy()
            # Add small random noise to break symmetries
            points += np.random.normal(0, 0.005, points.shape)
            points = np.clip(points, 0, 1)
            
            distances = pdist(points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
        except Exception as e:
            continue
    
    # If no constructive approach worked, start with random
    if best_points is None:
        best_points = np.random.rand(16, 2)
    
    # Refine using improved evolutionary algorithm
    try:
        evolved_points = improved_evolutionary_optimization()
        distances = pdist(evolved_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_points = evolved_points
                    best_ratio = ratio
    except Exception:
        pass
    
    # Final refinement with enhanced simulated annealing
    try:
        sa_points = enhanced_simulated_annealing()
        distances = pdist(sa_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_points = sa_points
                    best_ratio = ratio
    except Exception:
        pass
    
    # Final multi-start optimization for any remaining improvement
    try:
        final_points = multi_start_optimization()
        distances = pdist(final_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_points = final_points
                    best_ratio = ratio
    except Exception:
        pass
    
    # As a last resort, use differential evolution for global search with reduced iterations
    if best_ratio < 0.08:  # If we're still not satisfied
        try:
            def objective(params):
                points = params.reshape(-1, 2)
                distances = pdist(points)
                if len(distances) == 0 or np.max(distances) == 0:
                    return 0
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                return -min_dist / max_dist if max_dist > 0 else 0
            
            bounds = [(0, 1) for _ in range(32)]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=150,  # Reduced iterations to meet time limits
                popsize=15,   # Smaller population for speed
                seed=42,
                polish=True,
                strategy='best1bin'
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                distances = pdist(refined_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_points = refined_points
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
