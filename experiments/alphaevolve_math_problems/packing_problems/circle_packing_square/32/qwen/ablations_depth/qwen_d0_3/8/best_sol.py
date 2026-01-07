# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from sklearn.cluster import KMeans

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: enhanced evolutionary algorithm + advanced local optimization refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Enhanced initialization with better packing strategies
    def initialize_better_config():
        # Strategy 1: Clustered initialization with better spatial distribution
        def clustered_init():
            # Start with a grid pattern and then apply clustering to spread out
            positions = []
            
            # Create a coarse grid
            grid_size = 6
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= n:
                        break
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y
                    positions.append([x, y])
            
            # If we don't have enough positions, fill with random points
            if len(positions) < n:
                for _ in range(n - len(positions)):
                    positions.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
            
            positions = np.array(positions[:n])
            
            # Apply K-means clustering to improve distribution
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            kmeans.fit(positions)
            cluster_centers = kmeans.cluster_centers_
            
            # Compute initial radii based on proximity to neighbors
            radii = np.full(n, 0.05)
            tree = cKDTree(cluster_centers)
            
            for i in range(n):
                distances, indices = tree.query(cluster_centers[i], k=min(8, n), p=2)
                if len(distances) > 1:
                    min_dist = np.min(distances[1:])
                    radii[i] = min(0.1, min_dist / 2.0)
            
            return cluster_centers, radii
        
        # Strategy 2: Improved golden ratio spiral with better scaling
        def improved_golden_spiral_init():
            positions = []
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            # Better scaling for unit square
            max_radius = 0.45
            
            for i in range(n):
                angle = i * 2 * np.pi / phi
                # Use logarithmic spiral for better distribution
                radius = np.sqrt(i / (n - 1)) * max_radius if n > 1 else max_radius
                x = 0.5 + radius * np.cos(angle) * 0.9
                y = 0.5 + radius * np.sin(angle) * 0.9
                positions.append([x, y])
            
            positions = np.array(positions)
            
            # Compute initial radii
            radii = np.full(n, 0.05)
            tree = cKDTree(positions)
            
            for i in range(n):
                distances, indices = tree.query(positions[i], k=min(8, n), p=2)
                if len(distances) > 1:
                    min_dist = np.min(distances[1:])
                    radii[i] = min(0.1, min_dist / 2.0)
            
            return positions, radii
        
        # Strategy 3: Hybrid approach combining multiple patterns
        def hybrid_init():
            # Start with a regular hexagonal grid
            positions = []
            rows = 5
            cols = 7
            
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= n:
                        break
                    # Offset every other row
                    x = 0.1 + j * 0.15 + (i % 2) * 0.075
                    y = 0.1 + i * 0.13
            
                    # Keep within bounds
                    if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                        positions.append([x, y])
                if len(positions) >= n:
                    break
            
            # Fill remaining positions with random points in strategic areas
            if len(positions) < n:
                # Add some points near corners and edges for better boundary utilization
                corner_points = [
                    [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
                    [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
                ]
                for i in range(len(corner_points)):
                    if len(positions) >= n:
                        break
                    positions.append(corner_points[i])
                
                # Fill remaining with random points
                for _ in range(n - len(positions)):
                    positions.append([random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)])
            
            positions = np.array(positions[:n])
            
            # Compute initial radii based on proximity
            radii = np.full(n, 0.05)
            tree = cKDTree(positions)
            
            for i in range(n):
                distances, indices = tree.query(positions[i], k=min(8, n), p=2)
                if len(distances) > 1:
                    min_dist = np.min(distances[1:])
                    radii[i] = min(0.1, min_dist / 2.0)
            
            return positions, radii
        
        # Try initialization strategies in order of preference
        strategies = [
            (clustered_init, "clustered"),
            (hybrid_init, "hybrid"),
            (improved_golden_spiral_init, "golden")
        ]
        
        best_config = None
        best_sum = 0
        
        for init_func, name in strategies:
            try:
                positions, radii = init_func()
                # Validate initial configuration
                valid_positions = []
                valid_radii = []
                for i in range(n):
                    x, y = positions[i]
                    r = radii[i]
                    # Check containment
                    if (r <= x <= 1-r) and (r <= y <= 1-r):
                        valid_positions.append([x, y])
                        valid_radii.append(r)
                
                if len(valid_positions) == n:
                    config_sum = sum(valid_radii)
                    if config_sum > best_sum:
                        best_sum = config_sum
                        best_config = (np.array(valid_positions), np.array(valid_radii))
            except Exception as e:
                continue
        
        # Fallback to basic initialization if nothing works
        if best_config is None:
            positions = np.array([[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)] for _ in range(n)])
            radii = np.full(n, 0.05)
            # Adjust radii based on proximity
            tree = cKDTree(positions)
            for i in range(n):
                distances, indices = tree.query(positions[i], k=min(8, n), p=2)
                if len(distances) > 1:
                    min_dist = np.min(distances[1:])
                    radii[i] = min(0.1, min_dist / 2.0)
            best_config = (positions, radii)
        
        return best_config
    
    # More efficient constraint functions with better vectorization
    def containment_constraints(circles_flat):
        """Ensure all circles are fully contained within unit square"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # Vectorized containment constraints
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        r_coords = radii
        
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        constraints = np.concatenate([
            x_coords - r_coords,           # x - r >= 0
            1 - x_coords - r_coords,       # 1 - x - r >= 0
            y_coords - r_coords,           # y - r >= 0
            1 - y_coords - r_coords        # 1 - y - r >= 0
        ])
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # Vectorized non-overlap constraints using cdist for efficiency
        distances = cdist(positions, positions, 'euclidean')
        radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Non-overlap constraints: distance >= (r_i + r_j)
        # So we want: distance - (r_i + r_j) >= 0
        constraints = distances - radii_matrix
        
        # Only keep upper triangle (avoid duplicates) and diagonal zeros
        mask = np.triu(np.ones_like(constraints), k=1).astype(bool)
        return constraints[mask]
    
    # Objective function (negative because we minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat.reshape(-1, 3)[:, 2])
    
    # Constraint violation penalty with better handling
    def penalty_function(circles_flat):
        """Calculate penalty for constraint violations"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # Vectorized constraint checking
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        r_coords = radii
        
        # Check containment violations
        containment_violations = np.sum((x_coords - r_coords < 0) | 
                                       (x_coords + r_coords > 1) |
                                       (y_coords - r_coords < 0) |
                                       (y_coords + r_coords > 1))
        
        # Check overlap violations using a more efficient approach
        overlap_violations = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < radii[i] + radii[j]:
                    overlap_violations += 1
        
        # Return large penalty for violations
        return 1000 * (containment_violations + overlap_violations)
    
    # Enhanced optimization with evolutionary approach
    def evolutionary_optimization():
        # Create individual and population
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Attribute generator
        def create_individual():
            # Generate initial configuration
            positions, radii = initialize_better_config()
            # Flatten to individual representation
            individual = []
            for i in range(n):
                individual.extend([positions[i][0], positions[i][1], radii[i]])
            return creator.Individual(individual)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Fitness function with penalties
        def eval_fitness(individual):
            # Convert individual to circles array
            circles = np.array(individual).reshape(-1, 3)
            
            # Check constraints
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Sum of radii
            total_radii = np.sum(radii)
            
            # Penalty for constraint violations
            penalty = penalty_function(np.array(individual))
            
            # Sum of radii minus penalty
            return (total_radii - penalty,)
        
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution with more generations and better parameters
        pop = toolbox.population(n=75)  # Larger population
        hof = tools.HallOfFame(1)
        
        # Run evolution with early stopping
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, 
                                         ngen=75, stats=stats, halloffame=hof, verbose=False)
        except:
            # Fallback to local optimization if evolution fails
            pass
        
        if len(hof) > 0:
            return hof[0]
        return None
    
    # Advanced local optimization refinement with better strategies
    def refine_solution(initial_solution):
        """Refine solution using local optimization with multiple strategies"""
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        cons = [
            {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
            {'type': 'ineq', 'fun': lambda x: non_overlap_constraints(x)}
        ]
        
        # Try multiple optimization strategies
        best_result = None
        best_sum = -np.inf
        
        # Strategy 1: SLSQP with many restarts and better initialization
        for restart in range(25):  # More restarts for better exploration
            np.random.seed(42 + restart)
            perturbed = initial_solution.copy()
            
            # Add more substantial perturbations
            for i in range(n):
                # Add different amounts of noise to x, y, r
                perturbed[i*3] += np.random.normal(0, 0.05)  # x
                perturbed[i*3 + 1] += np.random.normal(0, 0.05)  # y
                perturbed[i*3 + 2] += np.random.normal(0, 0.02)  # r
            
            # Ensure bounds are respected
            for i in range(n):
                perturbed[i*3] = np.clip(perturbed[i*3], 0.001, 0.999)
                perturbed[i*3 + 1] = np.clip(perturbed[i*3 + 1], 0.001, 0.999)
                perturbed[i*3 + 2] = np.clip(perturbed[i*3 + 2], 0.001, 0.499)
            
            try:
                result = minimize(
                    objective,
                    perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 3000, 'ftol': 1e-12, 'eps': 1e-6},
                    callback=lambda x: None
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except:
                continue
        
        # Strategy 2: L-BFGS-B with better starting point
        if best_result is None or best_sum < 2.5:  # Only use if needed or if poor result
            try:
                # Start from a better point (maybe the initial solution itself)
                start_point = initial_solution.copy()
                # Add small random noise to avoid getting stuck in local minima
                for i in range(n):
                    start_point[i*3] += np.random.normal(0, 0.005)
                    start_point[i*3 + 1] += np.random.normal(0, 0.005)
                    start_point[i*3 + 2] += np.random.normal(0, 0.002)
                
                for i in range(n):
                    start_point[i*3] = np.clip(start_point[i*3], 0.001, 0.999)
                    start_point[i*3 + 1] = np.clip(start_point[i*3 + 1], 0.001, 0.999)
                    start_point[i*3 + 2] = np.clip(start_point[i*3 + 2], 0.001, 0.499)
                
                result = minimize(
                    objective,
                    start_point,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except:
                pass
        
        # Strategy 3: Trust-constr as backup (more robust)
        if best_result is None:
            try:
                np.random.seed(1000)
                start_point = initial_solution.copy()
                for i in range(n):
                    start_point[i*3] += np.random.normal(0, 0.01)
                    start_point[i*3 + 1] += np.random.normal(0, 0.01)
                    start_point[i*3 + 2] += np.random.normal(0, 0.005)
                
                for i in range(n):
                    start_point[i*3] = np.clip(start_point[i*3], 0.001, 0.999)
                    start_point[i*3 + 1] = np.clip(start_point[i*3 + 1], 0.001, 0.999)
                    start_point[i*3 + 2] = np.clip(start_point[i*3 + 2], 0.001, 0.499)
                
                result = minimize(
                    objective,
                    start_point,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 2500, 'ftol': 1e-12}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except:
                pass
        
        return best_result
    
    # Main optimization flow
    # Step 1: Get initial configuration
    positions, radii = initialize_better_config()
    initial_circles = np.column_stack([positions, radii]).flatten()
    
    # Step 2: Try evolutionary approach first
    evolved_solution = evolutionary_optimization()
    
    # Step 3: Refine with local optimization
    final_solution = initial_circles.copy()
    if evolved_solution is not None:
        # Try to improve the evolved solution
        refined = refine_solution(evolved_solution)
        if refined is not None and refined.success:
            final_solution = refined.x
    
    # Step 4: Final refinement with local optimization
    final_refined = refine_solution(final_solution)
    if final_refined is not None and final_refined.success:
        final_solution = final_refined.x
    
    # Convert back to circles format
    final_circles = final_solution.reshape(-1, 3)
    
    # Final validation and cleanup
    validated_circles = []
    for i in range(n):
        x = max(0.001, min(0.999, final_circles[i, 0]))
        y = max(0.001, min(0.999, final_circles[i, 1]))
        r = max(0.001, min(0.499, final_circles[i, 2]))
        validated_circles.append([x, y, r])
    
    return np.array(validated_circles)


# EVOLVE-BLOCK-END
