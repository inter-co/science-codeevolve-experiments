# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import math
from scipy.spatial import ConvexHull
import random
from sklearn.cluster import KMeans
import time
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import distance_matrix
import numba
from numba import jit

# Known good starting configuration for 14 points in 3D
# Based on research into optimal point distributions
def get_known_good_initialization():
    """Return a known good starting configuration for 14 points"""
    # This configuration is derived from known optimal arrangements
    # and mathematical analysis of point distributions
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0], 
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
        [0.25, 0.75, 0.25],
        [0.75, 0.25, 0.75]
    ])
    
    # Normalize to unit cube [0,1]^3
    # Find min/max along each dimension
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    
    # Avoid division by zero
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    # Normalize to [0,1]^3
    normalized = (points - mins) / ranges
    
    # Adjust to make sure we're in [0,1]^3
    normalized = np.clip(normalized, 0, 1)
    
    return normalized

@jit(nopython=True)
def compute_distance_matrix_numba(points):
    """Compute distance matrix using numba for speed"""
    n = points.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to 14x3 points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances efficiently
        distances = pdist(points)
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize ratio (minimize negative)
        # Add small epsilon to avoid division by zero
        if d_max < 1e-12:
            return -1e12
        return -d_min / d_max
    
    def objective_with_regularization(x, reg_weight=1e-6):
        """Objective with regularization to avoid degenerate solutions"""
        points = x.reshape(-1, 3)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max < 1e-12:
            return -1e12
            
        # Add regularization term to avoid very small distances
        regularization = reg_weight * np.sum((points - np.mean(points, axis=0))**2)
        return -(d_min / d_max) + regularization
    
    def constraint_bounds(x):
        """Constraint: points within unit cube [0,1]^3"""
        points = x.reshape(-1, 3)
        # Return difference from bounds (positive when out of bounds)
        return np.concatenate([
            points.min(axis=0),  # Lower bound (should be >= 0)
            1 - points.max(axis=0)  # Upper bound (should be >= 0)
        ])
    
    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n):
            theta = math.acos(-1 + (2 * i) / (n - 1))
            phi = math.sqrt(n * math.pi) * theta
            
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def generate_cube_vertices():
        """Generate vertices of a cube scaled appropriately"""
        # Cube vertices with some randomness to avoid degenerate cases
        vertices = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertices.append([i, j, k])
        return np.array(vertices)
    
    def generate_random_points(n, seed=None):
        """Generate random points in unit cube"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.rand(n, 3)
    
    def normalize_to_unit_cube(points):
        """Normalize points to fit in unit cube [0,1]^3"""
        # Translate to center around origin
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Scale to fit in [-0.5, 0.5]^3
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / (2 * max_coord)
        else:
            scaled = centered
            
        # Translate back to [0,1]^3
        final = scaled + 0.5
        return final
    
    def generate_spherical_arrangement():
        """Generate points arranged on a sphere with some randomness"""
        # Use Fibonacci sphere distribution
        points = generate_fibonacci_points(14)
        # Normalize to unit sphere and scale appropriately
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.5  # Scale to radius 0.5
        # Add some randomness to avoid perfect symmetry
        points += np.random.normal(0, 0.05, points.shape)
        # Normalize again to keep on sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.5
        # Move to unit cube
        points = points + 0.5  # Center in [0,1]^3
        return np.clip(points, 0, 1)
    
    def generate_clustered_initialization():
        """Generate points with clustering to avoid local minima"""
        # Start with a good initial configuration
        # Use a combination of sphere points and random points
        sphere_points = generate_spherical_arrangement()
        
        # Add some additional points in a structured way
        additional_points = np.random.rand(4, 3)
        
        # Combine and normalize
        combined = np.vstack([sphere_points, additional_points])
        return normalize_to_unit_cube(combined)[:14]
    
    def generate_voronoi_arrangement():
        """Generate points based on Voronoi-like arrangement"""
        # Generate points on a sphere first
        points = generate_fibonacci_points(14)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Add some perturbation to break symmetries
        perturbation = np.random.normal(0, 0.05, points.shape)
        points = points + perturbation
        
        # Normalize again to stay on sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Scale and translate to unit cube
        points = points * 0.4 + 0.5
        return np.clip(points, 0, 1)
    
    def generate_octahedral_arrangement():
        """Generate points based on octahedral symmetry"""
        # Octahedron vertices + face centers + edge midpoints
        points = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],  # Vertices
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],  # Opposite vertices
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # Face centers
            [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],  # Additional points
        ])
        
        # Normalize to unit cube
        return normalize_to_unit_cube(points[:14])
    
    def generate_tetrahedral_arrangement():
        """Generate points based on tetrahedral symmetry"""
        # Tetrahedron vertices + some interior points
        points = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],  # Tetrahedron vertices
            [0, 0, 0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5],  # Interior points
            [0, 0, 1], [0, 1, 0], [1, 0, 0],  # Face centers
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]  # Edge midpoints
        ])
        
        # Normalize to unit cube
        return normalize_to_unit_cube(points[:14])
    
    def generate_good_initialization():
        """Create multiple good initial configurations and pick the best"""
        candidates = []
        
        # Strategy 1: Known good configuration
        known_points = get_known_good_initialization()
        candidates.append(known_points)
        
        # Strategy 2: Fibonacci spiral on sphere, then project to cube
        fib_points = generate_fibonacci_points(14)
        fib_in_cube = normalize_to_unit_cube(fib_points)
        candidates.append(fib_in_cube)
        
        # Strategy 3: Random points
        rand_points = generate_random_points(14, seed=42)
        candidates.append(rand_points)
        
        # Strategy 4: Spherical arrangement
        sph_points = generate_spherical_arrangement()
        candidates.append(sph_points)
        
        # Strategy 5: Clustered initialization
        clustered_points = generate_clustered_initialization()
        candidates.append(clustered_points)
        
        # Strategy 6: Voronoi-based arrangement
        voronoi_points = generate_voronoi_arrangement()
        candidates.append(voronoi_points)
        
        # Strategy 7: Some vertices of a cube with random points
        cube_points = generate_cube_vertices()
        cube_points = np.vstack([cube_points, generate_random_points(10, seed=123)])
        candidates.append(normalize_to_unit_cube(cube_points[:14]))
        
        # Strategy 8: More diverse random initialization
        diverse_points = generate_random_points(14, seed=12345)
        candidates.append(diverse_points)
        
        # Strategy 9: Hybrid of cube vertices and random points
        cube_vertices = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
        mixed_points = np.vstack([cube_vertices, generate_random_points(8, seed=999)])
        candidates.append(normalize_to_unit_cube(mixed_points[:14]))
        
        # Strategy 10: Octahedral arrangement
        oct_points = generate_octahedral_arrangement()
        candidates.append(oct_points)
        
        # Strategy 11: Tetrahedral arrangement
        tet_points = generate_tetrahedral_arrangement()
        candidates.append(tet_points)
        
        # Strategy 12: Better initialization from literature
        # Points designed to be more evenly distributed
        literature_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.25]
        ])
        candidates.append(literature_points)
        
        # Strategy 13: Improved lattice-based arrangement
        # Create a more evenly spaced lattice pattern
        lattice_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i+j+k > 0 and i+j+k < 5:  # Avoid corners and center
                        lattice_points.append([i/2.5, j/2.5, k/2.5])
        # Add some randomness to avoid degeneracy
        if len(lattice_points) < 14:
            additional = np.random.rand(14 - len(lattice_points), 3)
            lattice_points.extend(additional)
        candidates.append(np.array(lattice_points[:14]))
        
        # Strategy 14: Golden ratio based arrangement
        golden_points = []
        phi = (1 + math.sqrt(5)) / 2
        for i in range(14):
            t = i / 13.0
            theta = math.acos(1 - 2 * t)
            phi_angle = i * 2 * math.pi / phi
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            z = math.cos(theta)
            golden_points.append([x, y, z])
        golden_points = np.array(golden_points)
        # Normalize and scale
        norms = np.linalg.norm(golden_points, axis=1)
        golden_points = golden_points / norms[:, np.newaxis] * 0.4 + 0.5
        candidates.append(np.clip(golden_points, 0, 1))
        
        # Evaluate all candidates
        best_candidate = None
        best_ratio = -np.inf
        
        for candidate in candidates:
            # Compute ratio for this candidate
            distances = pdist(candidate)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 0:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_candidate = candidate.copy()
        
        return best_candidate
    
    def advanced_optimization(initial_points, max_time=45):
        """Use advanced optimization with multiple strategies and restarts"""
        start_time = time.time()
        
        # Define constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: constraint_bounds(x)}
        ]
        
        # Bounds for optimization (points in [0,1]^3)
        bounds = [(0, 1)] * (14 * 3)
        
        best_result = None
        best_ratio = -np.inf
        
        # Multiple restarts with different optimization approaches
        restarts = 20  # More restarts for better exploration
        for restart in range(restarts):
            if time.time() - start_time > max_time * 0.9:
                break
                
            try:
                # Perturb initial points differently for each restart
                np.random.seed(restart)
                # Use adaptive perturbations
                perturbation_scale = max(0.02, 0.1 - restart * 0.005)  # Decreasing scale
                perturbed_points = initial_points + np.random.normal(0, perturbation_scale, initial_points.shape)
                # Keep in bounds
                perturbed_points = np.clip(perturbed_points, 0, 1)
                flattened_points = perturbed_points.flatten()
                
                # Try multiple optimization methods with different settings
                methods_and_options = [
                    ('L-BFGS-B', {'maxiter': 2500, 'ftol': 1e-15, 'gtol': 1e-15}),
                    ('TNC', {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}),
                    ('SLSQP', {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
                ]
                
                for method, options in methods_and_options:
                    if time.time() - start_time > max_time * 0.9:
                        break
                        
                    try:
                        result = minimize(
                            objective,
                            flattened_points,
                            method=method,
                            bounds=bounds,
                            constraints=constraints,
                            options=options
                        )
                        
                        if result.success:
                            final_points = result.x.reshape(-1, 3)
                            distances = pdist(final_points)
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            
                            if d_max > 0:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_result = final_points.copy()
                                    
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        # If no optimization worked, return the best initialization
        if best_result is None:
            return initial_points
        
        # Final refinement with regularized objective
        if time.time() - start_time < max_time * 0.95:
            try:
                flattened_points = best_result.flatten()
                result = minimize(
                    objective_with_regularization,
                    flattened_points,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}  # Even stricter tolerances
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    distances = pdist(final_points)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_result = final_points.copy()
                            
            except Exception:
                pass
        
        return best_result if best_result is not None else initial_points
    
    def improved_simulated_annealing_refinement(initial_points, max_time=20):
        """Use improved simulated annealing for global optimization with better cooling"""
        start_time = time.time()
        
        current_points = initial_points.copy()
        current_distances = pdist(current_points)
        current_d_min = np.min(current_distances)
        current_d_max = np.max(current_distances)
        current_ratio = current_d_min / current_d_max if current_d_max > 0 else 0
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Improved Simulated Annealing parameters
        temperature = 0.2  # Higher initial temp for better exploration
        cooling_rate = 0.9995  # Slower cooling for better convergence
        min_temperature = 1e-12
        max_iterations = 30000
        
        iteration = 0
        while temperature > min_temperature and iteration < max_iterations and time.time() - start_time < max_time:
            # Make a small random perturbation to one point
            point_idx = np.random.randint(0, 14)
            # Adaptive perturbation size based on temperature and iteration
            perturbation_size = max(0.005, 0.05 * temperature)  # Smaller perturbations at lower temps
            perturbation = np.random.normal(0, perturbation_size, 3)
            
            # Apply perturbation with bounds checking
            new_points = current_points.copy()
            new_points[point_idx] = new_points[point_idx] + perturbation
            new_points[point_idx] = np.clip(new_points[point_idx], 0, 1)
            
            # Calculate new ratio
            new_distances = pdist(new_points)
            new_d_min = np.min(new_distances)
            new_d_max = np.max(new_distances)
            new_ratio = new_d_min / new_d_max if new_d_max > 0 else 0
            
            # Accept or reject the move
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temperature):
                current_points = new_points
                current_ratio = new_ratio
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
            
            # Cool down
            temperature *= cooling_rate
            iteration += 1
        
        return best_points
    
    def improved_genetic_algorithm_refinement(initial_points, max_time=20):
        """Use improved genetic algorithm for global optimization"""
        start_time = time.time()
        
        population_size = 30  # Larger population for better diversity
        generations = 800   # More generations for better convergence
        mutation_rate = 0.2  # Higher mutation rate for better exploration
        elite_size = 5  # Keep top individuals
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = initial_points + np.random.normal(0, 0.03, initial_points.shape)
            individual = np.clip(individual, 0, 1)
            population.append(individual)
        
        best_individual = initial_points.copy()
        best_fitness = -np.inf
        
        for generation in range(generations):
            if time.time() - start_time > max_time:
                break
                
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                distances = pdist(individual)
                d_min = np.min(distances)
                d_max = np.max(distances)
                fitness = d_min / d_max if d_max > 0 else 0
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]
            
            # Keep elite
            selected = sorted_population[:elite_size]
            
            # Tournament selection for rest
            while len(selected) < population_size:
                tournament_size = 5
                tournament_indices = np.random.choice(len(sorted_population), tournament_size, replace=False)
                tournament_fitness = [sorted_fitness[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(sorted_population[winner_index].copy())
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
                
                # Uniform crossover with higher probability for better parents
                mask = np.random.rand(14, 3) < 0.6  # Bias towards better parents
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
                
                # Mutation with adaptive rate
                mutation_prob = max(0.05, mutation_rate * (1 - generation / generations))
                if np.random.rand() < mutation_prob:
                    mutate_idx = np.random.randint(0, 14)
                    child1[mutate_idx] += np.random.normal(0, 0.02, 3)
                    child1[mutate_idx] = np.clip(child1[mutate_idx], 0, 1)
                
                if np.random.rand() < mutation_prob:
                    mutate_idx = np.random.randint(0, 14)
                    child2[mutate_idx] += np.random.normal(0, 0.02, 3)
                    child2[mutate_idx] = np.clip(child2[mutate_idx], 0, 1)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        return best_individual
    
    # Generate initial points using the enhanced approach
    initial_points = generate_good_initialization()
    
    # First, try improved genetic algorithm refinement
    refined_points = improved_genetic_algorithm_refinement(initial_points, max_time=15)
    
    # Then try improved simulated annealing refinement
    refined_points = improved_simulated_annealing_refinement(refined_points, max_time=15)
    
    # Finally, run advanced optimization
    final_result = advanced_optimization(refined_points, max_time=30)
    
    return final_result


# EVOLVE-BLOCK-END
