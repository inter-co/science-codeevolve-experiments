# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import math
from typing import Tuple
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, evolutionary search, and local refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 3)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0  # Avoid division by zero
            
        return -min_dist / max_dist
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def generate_geometric_initialization() -> np.ndarray:
        """Generate diverse geometric initial configurations"""
        # Strategy 1: Fibonacci spiral on sphere
        def fibonacci_spiral(n_points):
            points = []
            golden_ratio = (1 + math.sqrt(5)) / 2
            for i in range(n_points):
                theta = math.acos(-1 + 2 * i / (n_points - 1))
                phi = math.sqrt(n_points) * theta
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                points.append([x, y, z])
            return np.array(points)
        
        # Strategy 2: Octahedral arrangement
        def octahedral_arrangement():
            points = []
            # Vertices of octahedron
            for coord in [-1, 1]:
                points.append([coord, 0, 0])
                points.append([0, coord, 0])
                points.append([0, 0, coord])
            
            # Add additional points using spherical code principles
            # Add 6 more points distributed symmetrically
            for i in range(6):
                angle = 2 * math.pi * i / 6
                z = 0.5
                r = math.sqrt(1 - z*z)
                points.append([r * math.cos(angle), r * math.sin(angle), z])
                points.append([r * math.cos(angle), r * math.sin(angle), -z])
            
            return np.array(points[:14])
        
        # Strategy 3: Tetrahedral + additional points
        def tetrahedral_plus():
            # Regular tetrahedron vertices
            points = [
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1]
            ]
            
            # Normalize to unit sphere
            points = np.array(points)
            points = points / np.linalg.norm(points[0])
            
            # Add 10 more points using a spherical design
            # Use a pattern that maximizes minimum distance
            additional = []
            for i in range(10):
                # Distribute points more evenly
                phi = math.acos(-1 + 2 * i / 9)
                theta = math.sqrt(14) * phi
                x = math.sin(phi) * math.cos(theta)
                y = math.sin(phi) * math.sin(theta)
                z = math.cos(phi)
                additional.append([x, y, z])
            
            return np.vstack([points, additional])
        
        # Strategy 4: Random with constraints
        def constrained_random():
            points = []
            while len(points) < 14:
                # Generate random point in unit ball
                point = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(point) <= 1:
                    points.append(point)
            return np.array(points)
        
        # Try different strategies and pick the best
        strategies = [
            fibonacci_spiral,
            octahedral_arrangement,
            tetrahedral_plus,
            constrained_random
        ]
        
        best_initialization = None
        best_score = float('inf')
        
        for strategy in strategies:
            try:
                initial = strategy()
                if len(initial) >= 14:
                    initial = initial[:14]
                score = -objective(initial.flatten())  # Lower is better for objective
                
                if score > best_score:
                    best_score = score
                    best_initialization = initial.copy()
            except Exception:
                continue
        
        if best_initialization is None:
            # Fallback to random initialization
            points = []
            while len(points) < 14:
                point = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(point) <= 1:
                    points.append(point)
            best_initialization = np.array(points)
        
        return best_initialization
    
    def evolutionary_search() -> np.ndarray:
        """Perform evolutionary search with multiple strategies"""
        # Generate diverse initial population
        population_size = 20
        population = []
        
        # Create diverse initial solutions
        for i in range(population_size):
            # Mix different initialization strategies
            if i < 5:
                # Fibonacci spiral
                points = generate_geometric_initialization()
                points = points + np.random.normal(0, 0.05, points.shape)
            elif i < 10:
                # Perturbed octahedral
                points = generate_geometric_initialization()
                points = points + np.random.normal(0, 0.1, points.shape)
            else:
                # Random with constraints
                points = generate_geometric_initialization()
                points = points + np.random.normal(0, 0.15, points.shape)
            
            # Ensure points stay within unit sphere
            for j in range(len(points)):
                norm = np.linalg.norm(points[j])
                if norm > 1:
                    points[j] = points[j] / norm
            
            population.append(points.flatten())
        
        # Evolutionary parameters
        generations = 50
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Keep track of best solution
        best_individual = None
        best_fitness = float('-inf')
        
        for gen in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                try:
                    fitness = -objective(individual)  # Convert to maximization
                    fitness_scores.append(fitness)
                except:
                    fitness_scores.append(float('-inf'))
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_individual = population[max_idx].copy()
            
            # Selection (tournament selection)
            selected_indices = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_index)
            
            # Create next generation through crossover and mutation
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_individual.copy())
            
            while len(new_population) < population_size:
                # Select parents
                parent1_idx = selected_indices[np.random.randint(0, len(selected_indices))]
                parent2_idx = selected_indices[np.random.randint(0, len(selected_indices))]
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if random.random() < crossover_rate and len(parent1) > 1:
                    crossover_point = random.randint(1, len(parent1) - 1)
                    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    mutation_indices = np.random.choice(len(child), size=max(1, len(child) // 10), replace=False)
                    for idx in mutation_indices:
                        # Add Gaussian noise
                        child[idx] += np.random.normal(0, 0.05)
                
                # Keep within bounds
                child = np.clip(child, -1, 1)
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        # Return best solution
        if best_individual is not None:
            return best_individual.reshape(-1, 3)
        else:
            # Fallback to geometric initialization
            return generate_geometric_initialization()
    
    def local_refinement(points: np.ndarray) -> np.ndarray:
        """Apply local refinement using simulated annealing approach"""
        current_points = points.copy()
        current_objective_value = -objective(current_points.flatten())
        
        # Simulated annealing parameters
        temperature = 1.0
        cooling_rate = 0.95
        min_temperature = 1e-6
        iterations_per_temp = 100
        
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Create neighbor solution
                neighbor_points = current_points.copy()
                
                # Perturb one point
                point_idx = np.random.randint(0, len(neighbor_points))
                neighbor_points[point_idx] += np.random.normal(0, 0.01, 3)
                
                # Keep within unit sphere
                norm = np.linalg.norm(neighbor_points[point_idx])
                if norm > 1:
                    neighbor_points[point_idx] = neighbor_points[point_idx] / norm
                
                # Evaluate neighbor
                neighbor_objective_value = -objective(neighbor_points.flatten())
                
                # Accept or reject
                if neighbor_objective_value > current_objective_value:
                    current_points = neighbor_points
                    current_objective_value = neighbor_objective_value
                else:
                    # Accept with probability based on temperature
                    delta = neighbor_objective_value - current_objective_value
                    acceptance_prob = math.exp(delta / temperature)
                    if random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_objective_value = neighbor_objective_value
            
            temperature *= cooling_rate
        
        return current_points
    
    # Main hybrid approach
    # Generate initial diverse solutions using multiple geometric strategies
    initial_points = generate_geometric_initialization()
    
    # Perform evolutionary search to explore the space
    evolved_points = evolutionary_search()
    
    # Apply local refinement to the evolved solution
    refined_points = local_refinement(evolved_points)
    
    # Final evaluation and comparison
    original_score = -objective(initial_points.flatten())
    evolved_score = -objective(evolved_points.flatten())
    refined_score = -objective(refined_points.flatten())
    
    # Return the best solution found
    if refined_score >= evolved_score and refined_score >= original_score:
        return refined_points
    elif evolved_score >= original_score:
        return evolved_points
    else:
        return initial_points


# EVOLVE-BLOCK-END
