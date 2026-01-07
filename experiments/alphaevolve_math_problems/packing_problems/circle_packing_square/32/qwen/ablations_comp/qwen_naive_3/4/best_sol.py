# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random
from sklearn.cluster import KMeans
import time
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining evolutionary algorithms with advanced local optimization.
    """
    n = 32
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    # Define individual and fitness for DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def create_initial_individual():
        """Create a better initial individual using a more sophisticated approach"""
        # Start with a hexagonal packing pattern for better coverage
        circles = []
        grid_size = int(np.ceil(np.sqrt(n)))
        
        # Create a hexagonal lattice pattern
        spacing = 0.8 / (grid_size + 1)  # Adjust spacing to fit within unit square
        offset = spacing * 0.5
        
        placed_count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if placed_count >= n:
                    break
                # Hexagonal offset for even rows
                x = (i + 1) * spacing + (j % 2) * offset
                y = (j + 1) * spacing
                # Initial radius - based on spacing
                r = spacing * 0.3
                # Ensure within bounds
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                circles.extend([x, y, r])
                placed_count += 1
        return creator.Individual(circles)
    
    def create_random_individual():
        """Create a random individual with 32 circles"""
        individual = []
        for _ in range(n):
            # x, y, radius
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            r = random.uniform(0.01, 0.15)  # Reasonable initial radius
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (sum of radii)"""
        circles = np.array(individual).reshape(-1, 3)
        radii = circles[:, 2]
        return np.sum(radii),
    
    def is_valid(individual):
        """Check if individual satisfies all constraints efficiently"""
        circles = np.array(individual).reshape(-1, 3)
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Check boundary constraints more efficiently
        if np.any(radii > x_coords) or np.any(radii > (1 - x_coords)) or \
           np.any(radii > y_coords) or np.any(radii > (1 - y_coords)):
            return False
            
        # Use vectorized approach for overlap constraints
        try:
            # Create distance matrix for all pairs
            diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
            diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
            dist_sq = diff_x**2 + diff_y**2
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Create mask for upper triangle to avoid duplicate checks
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            overlap_distances = dist_sq[mask]
            overlap_radii = r_sum[mask]**2
            
            # Check if any pair violates overlap constraint
            if np.any(overlap_distances < overlap_radii):
                return False
        except Exception:
            return False
                
        return True
    
    def mutate_individual(individual, indpb=0.1):
        """Mutate an individual with adaptive mutation rates"""
        for i in range(len(individual)):
            if random.random() < indpb:
                if i % 3 == 0:  # x coordinate
                    individual[i] = max(0.001, min(0.999, individual[i] + random.gauss(0, 0.005)))
                elif i % 3 == 1:  # y coordinate
                    individual[i] = max(0.001, min(0.999, individual[i] + random.gauss(0, 0.005)))
                else:  # radius
                    individual[i] = max(0.001, min(0.499, individual[i] + random.gauss(0, 0.003)))
        return individual,
    
    def crossover_individual(ind1, ind2):
        """Crossover two individuals with uniform crossover"""
        size = len(ind1)
        # Uniform crossover
        for i in range(size):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    # Create initial population with better diversity
    def create_population(size=100):
        population = []
        # Add some good initial solutions
        for _ in range(size // 3):
            population.append(create_initial_individual())
        # Add random solutions
        for _ in range(2 * size // 3):
            population.append(create_random_individual())
        return population
    
    # Enhanced evolutionary algorithm with better parameters
    def evolutionary_optimize():
        # Create population
        pop = create_population(150)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution with more generations and better parameters
        try:
            pop, logbook = algorithms.eaSimple(pop, tools.ParetoSelection(), 
                                              crossover_individual, mutate_individual,
                                              cxpb=0.8, mutpb=0.3, ngen=70,
                                              stats=stats, halloffame=hof, verbose=False)
        except Exception:
            # Fallback if evolution fails
            pass
            
        return hof[0] if len(hof) > 0 else create_initial_individual()
    
    # Improved local optimization refinement with better constraint handling
    def refine_solution(individual):
        """Use advanced local optimization to improve the solution"""
        circles = np.array(individual).reshape(-1, 3)
        initial_vars = circles.flatten()
        
        # Set bounds for variables (x, y, r)
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Constraint function using more robust approach
        def constraint_func(vars):
            constraints = []
            circles = vars.reshape(-1, 3)
            x_coords = circles[:, 0]
            y_coords = circles[:, 1]
            radii = circles[:, 2]
            
            # Boundary constraints (positive when satisfied)
            left_bound = x_coords - radii
            right_bound = 1 - x_coords - radii
            bottom_bound = y_coords - radii
            top_bound = 1 - y_coords - radii
            
            constraints.extend(left_bound.tolist())
            constraints.extend(right_bound.tolist())
            constraints.extend(bottom_bound.tolist())
            constraints.extend(top_bound.tolist())
            
            # Non-overlap constraints - vectorized for efficiency
            try:
                # Create distance matrix for all pairs
                diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
                diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
                dist_sq = diff_x**2 + diff_y**2
                r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
                
                # Create mask for upper triangle to avoid duplicate checks
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                overlap_distances = dist_sq[mask]
                overlap_radii = r_sum[mask]**2
                
                # This should be positive when constraint is satisfied
                overlap_constraints = overlap_distances - overlap_radii
                constraints.extend(overlap_constraints.tolist())
            except Exception:
                # Fallback to brute force if needed
                diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
                diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
                dist_sq = diff_x**2 + diff_y**2
                r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
                
                overlap_constraints = dist_sq - r_sum**2
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                overlap_constraints = overlap_constraints[mask]
                
                constraints.extend(overlap_constraints.tolist())
            
            return np.array(constraints)
        
        # Objective function (negative sum of radii for minimization)
        def objective(vars):
            circles = vars.reshape(-1, 3)
            return -np.sum(circles[:, 2])
        
        try:
            # Try with SLSQP first
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-6}
            )
            
            if result.success:
                optimized_vars = result.x
                return optimized_vars.reshape(-1, 3).flatten()
        except Exception:
            pass
            
        return individual
    
    # Advanced local search improvement with more aggressive optimization
    def advanced_local_search(individual):
        """Perform more sophisticated local search"""
        circles = np.array(individual).reshape(-1, 3)
        
        # Multiple optimization attempts with different strategies
        best_circles = circles.copy()
        best_score = np.sum(circles[:, 2])
        
        # Strategy 1: Gradient-based optimization with better constraints
        try:
            initial_vars = circles.flatten()
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            def constraint_func(vars):
                constraints = []
                circles = vars.reshape(-1, 3)
                x_coords = circles[:, 0]
                y_coords = circles[:, 1]
                radii = circles[:, 2]
                
                # Boundary constraints
                left_bound = x_coords - radii
                right_bound = 1 - x_coords - radii
                bottom_bound = y_coords - radii
                top_bound = 1 - y_coords - radii
                
                constraints.extend(left_bound.tolist())
                constraints.extend(right_bound.tolist())
                constraints.extend(bottom_bound.tolist())
                constraints.extend(top_bound.tolist())
                
                # Non-overlap constraints
                diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
                diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
                dist_sq = diff_x**2 + diff_y**2
                r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
                
                overlap_constraints = dist_sq - r_sum**2
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                overlap_constraints = overlap_constraints[mask]
                
                constraints.extend(overlap_constraints.tolist())
                return np.array(constraints)
            
            def objective(vars):
                circles = vars.reshape(-1, 3)
                return -np.sum(circles[:, 2])
            
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 300, 'ftol': 1e-8, 'eps': 1e-6}
            )
            
            if result.success:
                new_circles = result.x.reshape(-1, 3)
                new_score = np.sum(new_circles[:, 2])
                if new_score > best_score:
                    best_circles = new_circles
                    best_score = new_score
                    
        except Exception:
            pass
        
        # Strategy 2: Local neighborhood search with more extensive exploration
        try:
            # Try to improve each circle individually
            improved = True
            iteration = 0
            max_iterations = 10
            
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                
                for i in range(n):
                    old_x, old_y, old_r = best_circles[i]
                    best_x, best_y, best_r = old_x, old_y, old_r
                    best_score = np.sum(best_circles[:, 2])
                    
                    # More extensive neighborhood search
                    step_sizes = [0.005, 0.01, 0.02]
                    for step in step_sizes:
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                if abs(dx) + abs(dy) == 0:
                                    continue
                                new_x = max(old_r, min(1-old_r, old_x + dx))
                                new_y = max(old_r, min(1-old_r, old_y + dy))
                                
                                # Test if this improves the solution
                                temp_circles = best_circles.copy()
                                temp_circles[i] = [new_x, new_y, old_r]
                                
                                # Check validity and calculate score
                                if is_valid(temp_circles.flatten()):
                                    new_score = np.sum(temp_circles[:, 2])
                                    if new_score > best_score:
                                        best_score = new_score
                                        best_x, best_y = new_x, new_y
                                        improved = True
                        
                    best_circles[i] = [best_x, best_y, old_r]
                        
        except Exception:
            pass
        
        return best_circles.flatten()
    
    # Main optimization process with improved workflow
    # Step 1: Evolutionary optimization with better initialization
    evolved_individual = evolutionary_optimize()
    
    # Step 2: Local optimization refinement
    refined_individual = refine_solution(evolved_individual)
    
    # Step 3: Advanced local search
    circles = advanced_local_search(refined_individual)
    
    # Convert back to array
    circles = np.array(circles).reshape(-1, 3)
    
    # Final validation and cleanup
    for i in range(n):
        x, y, r = circles[i]
        # Ensure valid bounds
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
        circles[i][2] = max(0.001, min(0.499, r))
    
    # Perform final aggressive local search improvement
    try:
        # Even more thorough optimization
        improved = True
        iterations = 0
        max_iterations = 15
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try to optimize each circle more aggressively
            for i in range(n):
                old_x, old_y, old_r = circles[i]
                best_x, best_y, best_r = old_x, old_y, old_r
                best_score = np.sum(circles[:, 2])
                
                # Very fine-grained neighborhood search
                search_space = np.linspace(-0.02, 0.02, 9)
                for dx in search_space:
                    for dy in search_space:
                        if abs(dx) + abs(dy) == 0:
                            continue
                        new_x = max(old_r, min(1-old_r, old_x + dx))
                        new_y = max(old_r, min(1-old_r, old_y + dy))
                        
                        # Test if this improves the solution
                        temp_circles = circles.copy()
                        temp_circles[i] = [new_x, new_y, old_r]
                        
                        # Check validity and calculate score
                        if is_valid(temp_circles.flatten()):
                            new_score = np.sum(temp_circles[:, 2])
                            if new_score > best_score:
                                best_score = new_score
                                best_x, best_y = new_x, new_y
                                improved = True
                
                circles[i] = [best_x, best_y, old_r]
                
    except Exception:
        pass
    
    return circles


# EVOLVE-BLOCK-END
