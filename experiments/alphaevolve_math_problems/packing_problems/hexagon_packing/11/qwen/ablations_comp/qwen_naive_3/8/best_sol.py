# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

# Precompute hexagon vertices for unit hexagon centered at origin
def get_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Get vertices of a unit regular hexagon with optional rotation"""
    angle = rotation * np.pi / 180
    # Unit hexagon vertices (radius = 1)
    hex_points = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_points.append((x + center[0], y + center[1]))
    return np.array(hex_points)

def hexagon_area(side_length):
    """Calculate area of regular hexagon with given side length"""
    return (3 * np.sqrt(3) / 2) * side_length ** 2

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_side_length(inner_hex_data):
    """Compute minimum outer hexagon side length needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_unit_hexagon_vertices((x, y), angle)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1000
    
    all_vertices = np.array(all_vertices)
    
    # For a regular hexagon, we need to compute the minimum circumscribing hexagon
    # The key insight: for a hexagon, we need to find the maximum distance from center
    # to any vertex in the 6 principal directions (including diagonals)
    
    # First get the centroid
    centroid_x = np.mean(all_vertices[:, 0])
    centroid_y = np.mean(all_vertices[:, 1])
    
    # Convert all points to polar coordinates relative to centroid
    dx = all_vertices[:, 0] - centroid_x
    dy = all_vertices[:, 1] - centroid_y
    
    # We need to find the minimum circumscribing regular hexagon
    # The maximum distance from center to any vertex in any direction
    # For a regular hexagon, we project all points onto 6 directions: 0°, 30°, 60°, 90°, 120°, 150°
    
    # Project points onto 6 directions (0°, 30°, 60°, 90°, 120°, 150°)
    directions = np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6])
    max_projections = []
    
    for direction in directions:
        # Project points onto this direction
        proj = dx * np.cos(direction) + dy * np.sin(direction)
        max_proj = np.max(proj)
        max_projections.append(max_proj)
    
    # The maximum projection gives us the distance from center to boundary in that direction
    # For a regular hexagon, the side length is equal to the distance from center to vertex
    max_distance = np.max(max_projections)
    
    # Add a small margin to ensure containment
    return max_distance * 1.02

def calculate_arrangement_penalty(hex_data):
    """Calculate penalty for overlaps and containment violations"""
    total_penalty = 0
    inner_positions = hex_data[:, :2]  # (x, y) positions
    inner_angles = hex_data[:, 2]     # angles in degrees
    
    # Check overlap between all pairs of hexagons
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            pos_i, angle_i = inner_positions[i], inner_angles[i]
            pos_j, angle_j = inner_positions[j], inner_angles[j]
            
            hex_i = get_unit_hexagon_vertices(pos_i, angle_i)
            hex_j = get_unit_hexagon_vertices(pos_j, angle_j)
            
            # Check if they intersect
            if check_hexagon_overlap(hex_i, hex_j):
                # Calculate intersection area as penalty
                poly_i = Polygon(hex_i)
                poly_j = Polygon(hex_j)
                try:
                    intersection = poly_i.intersection(poly_j)
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        total_penalty += intersection.area * 10000  # Heavier penalty
                except:
                    # Fallback if intersection fails
                    total_penalty += 100000
    
    # Check containment of all hexagons within a bounding hexagon
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    # Create a hexagon that can contain all vertices
    outer_radius = outer_side_length
    outer_hex_vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        outer_hex_vertices.append((x, y))
    outer_poly = Polygon(outer_hex_vertices)
    
    # Add penalty if any vertex of inner hexagons is outside outer hexagon
    for i in range(len(hex_data)):
        pos, angle = inner_positions[i], inner_angles[i]
        vertices = get_unit_hexagon_vertices(pos, angle)
        
        for vertex in vertices:
            point = Point(vertex)
            if not outer_poly.contains(point):
                # Add penalty proportional to how far outside it is
                try:
                    dist_to_boundary = outer_poly.distance(point)
                    total_penalty += dist_to_boundary * 100000
                except:
                    # Fallback penalty if distance calculation fails
                    total_penalty += 1000000
    
    return total_penalty

def evaluate_arrangement(hex_data):
    """Evaluate arrangement quality - returns negative of 1/outer_radius for optimization"""
    # Check for overlaps and containment violations
    penalty = calculate_arrangement_penalty(hex_data)
    
    if penalty > 0:
        return 1000000 + penalty  # Invalid arrangement
    
    # Compute outer hexagon side length
    outer_side_length = compute_outer_hexagon_side_length(hex_data)
    
    if outer_side_length > 1000 or outer_side_length <= 0:
        return 1000000  # Invalid arrangement
    
    # Return negative inverse of outer side length (to maximize 1/outer_side_length)
    return -1.0 / outer_side_length if outer_side_length > 0 else 1000000

def generate_initial_configuration():
    """Generate a better initial configuration based on known good packings"""
    # Try to create a more optimal starting configuration
    # Based on known dense packings for 11 hexagons
    
    # Better initial configuration - more symmetric and efficient
    # Start with a central hexagon and arrange others around it optimally
    
    # Pattern inspired by densest known arrangements for 11 hexagons
    # Central hexagon surrounded by a ring, plus some additional placement
    
    positions = [
        [0.0, 0.0, 0.0],           # center - fixed
        [1.0, 0.0, 0.0],           # right
        [-1.0, 0.0, 0.0],          # left
        [0.5, 0.866, 0.0],         # top-right
        [-0.5, 0.866, 0.0],        # top-left
        [0.5, -0.866, 0.0],        # bottom-right
        [-0.5, -0.866, 0.0],       # bottom-left
        [1.3, 0.75, 0.0],          # additional hexagon near top-right
        [-1.3, 0.75, 0.0],         # additional hexagon near top-left
        [1.3, -0.75, 0.0],         # additional hexagon near bottom-right
        [-1.3, -0.75, 0.0]         # additional hexagon near bottom-left
    ]
    
    # Apply more aggressive adjustments for better packing
    adjusted_positions = []
    for i, pos in enumerate(positions):
        x, y, angle = pos
        if i == 0:  # Center - keep fixed
            adjusted_positions.append([x, y, angle])
        elif i <= 6:  # First ring - compress slightly
            adjusted_positions.append([x * 0.92, y * 0.92, angle])
        else:  # Additional positions - compress even more
            adjusted_positions.append([x * 0.85, y * 0.85, angle])
    
    return np.array(adjusted_positions)

def improved_simulated_annealing_optimization(max_iterations=5000):
    """Use improved simulated annealing to optimize the hexagon arrangement"""
    # Start with a good initial configuration
    current_solution = generate_initial_configuration()
    current_energy = evaluate_arrangement(current_solution)
    
    # Parameters for simulated annealing - better tuned for this problem
    temperature = 1.0
    cooling_rate = 0.99995
    min_temperature = 0.00001
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Store history for debugging
    energy_history = []
    
    # Track recent improvements to detect stagnation
    recent_improvements = []
    
    # Use different perturbation strategies based on iteration
    for iteration in range(max_iterations):
        # Generate neighbor solution by perturbing one hexagon
        neighbor_solution = current_solution.copy()
        
        # Choose a random hexagon to perturb (exclude center for stability)
        hexagon_idx = np.random.randint(1, len(neighbor_solution))  # Skip center
        
        # Use different step sizes based on iteration and hexagon type
        # Early iterations: larger steps for exploration
        # Later iterations: smaller steps for exploitation
        
        # Determine step size based on iteration and hexagon type
        if iteration < max_iterations * 0.2:  # Early phase - explore more
            step_size_pos = 0.15
            step_size_rot = 20.0
        elif iteration < max_iterations * 0.5:  # Middle phase - balance
            step_size_pos = 0.08
            step_size_rot = 12.0
        else:  # Late phase - exploit
            step_size_pos = 0.03
            step_size_rot = 6.0
        
        # Perturb position and rotation
        neighbor_solution[hexagon_idx, 0] += np.random.normal(0, step_size_pos)  # x position
        neighbor_solution[hexagon_idx, 1] += np.random.normal(0, step_size_pos)  # y position
        neighbor_solution[hexagon_idx, 2] += np.random.normal(0, step_size_rot)   # rotation
        
        # Keep rotation in [0, 360)
        neighbor_solution[hexagon_idx, 2] = neighbor_solution[hexagon_idx, 2] % 360
        
        # Evaluate neighbor
        neighbor_energy = evaluate_arrangement(neighbor_solution)
        
        # Accept or reject based on Metropolis criterion
        if neighbor_energy < current_energy:
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        else:
            # Accept with probability based on temperature
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 10000:  # Avoid accepting very bad moves too often
                acceptance_probability = np.exp(-delta_energy / temperature)
                if np.random.random() < acceptance_probability:
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
        
        # Update best solution
        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
        
        # Cool down more aggressively in early stages
        if iteration < max_iterations * 0.2:
            temperature = max(min_temperature, temperature * (cooling_rate * 0.9))
        else:
            temperature = max(min_temperature, temperature * cooling_rate)
        
        # Track progress
        if iteration % 100 == 0:
            energy_history.append(current_energy)
            
        # Detect stagnation and apply restart strategy
        if iteration % 500 == 0 and iteration > 0:
            if len(recent_improvements) > 0:
                avg_improvement = np.mean(recent_improvements[-100:])
                if avg_improvement > -0.00001:  # Very slow improvement
                    # Restart with better solution
                    current_solution = best_solution.copy()
                    temperature = 0.1  # Higher temperature to escape local minima
        
        # Add to recent improvements
        if len(recent_improvements) >= 100:
            recent_improvements.pop(0)
        recent_improvements.append(current_energy)
    
    return best_solution, best_energy

def genetic_algorithm_optimization(max_generations=100, population_size=50):
    """Use a hybrid genetic algorithm approach for better optimization"""
    from deap import base, creator, tools, algorithms
    
    # Create types for DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define gene bounds
    def create_individual():
        # Each individual represents 11 hexagons with (x, y, angle) values
        individual = []
        for _ in range(11):
            # x, y in [-3, 3], angle in [0, 360]
            individual.extend([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                np.random.uniform(0, 360)
            ])
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_hexagon_arrangement(individual):
        # Convert individual to hex_data format
        hex_data = np.array(individual).reshape(-1, 3)
        
        # Evaluate the arrangement
        penalty = calculate_arrangement_penalty(hex_data)
        
        if penalty > 0:
            return (-1000000,)  # Invalid arrangement
        
        outer_side_length = compute_outer_hexagon_side_length(hex_data)
        
        if outer_side_length > 1000 or outer_side_length <= 0:
            return (-1000000,)
        
        # Return negative inverse of outer side length (to maximize 1/outer_side_length)
        return (-1.0 / outer_side_length if outer_side_length > 0 else -1000000,)
    
    toolbox.register("evaluate", eval_hexagon_arrangement)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run the genetic algorithm
    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    
    # Run evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.7, mutpb=0.3, ngen=max_generations,
            stats=stats, halloffame=hof, verbose=False
        )
    except:
        # Fallback to simple approach if GA fails
        return improved_simulated_annealing_optimization(max_iterations=3000)
    
    best_individual = hof[0]
    best_hex_data = np.array(best_individual).reshape(-1, 3)
    
    return best_hex_data, eval_hexagon_arrangement(best_individual)[0]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a physics-inspired simulated annealing approach for optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use improved simulated annealing optimization
    start_time = time.time()
    
    # Try both approaches and pick the better one
    sa_solution, sa_energy = improved_simulated_annealing_optimization(max_iterations=3000)
    
    # Try genetic algorithm for potential improvement
    try:
        ga_solution, ga_energy = genetic_algorithm_optimization(max_generations=50, population_size=30)
        if ga_energy > sa_energy:
            inner_hex_data = ga_solution
            best_energy = ga_energy
        else:
            inner_hex_data = sa_solution
            best_energy = sa_energy
    except:
        # If GA fails, use SA result
        inner_hex_data = sa_solution
        best_energy = sa_energy
    
    end_time = time.time()
    
    # Compute final outer hexagon size
    outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Ensure we have a valid solution
    if outer_side_length <= 0 or np.isnan(outer_side_length):
        # Fall back to a known good configuration
        inner_hex_data = generate_initial_configuration()
        outer_side_length = compute_outer_hexagon_side_length(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
