# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multiple optimization strategies,
    and a fundamentally different combinatorial game theory approach for superior results.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    def create_hexagonal_lattice():
        """Create a high-quality hexagonal lattice pattern - key insight from inspirations"""
        points = []
        
        # Center point
        points.append([0.5, 0.5])
        
        # First ring of 6 points (radius 0.3)
        for i in range(6):
            angle = i * np.pi / 3
            points.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
        
        # Second ring of 9 points (radius 0.6) in triangular arrangement
        for i in range(3):
            angle = i * 2 * np.pi / 3
            for j in range(3):
                radius = 0.6 + j * 0.15
                points.append([0.5 + radius * np.cos(angle + j * np.pi/6), 
                              0.5 + radius * np.sin(angle + j * np.pi/6)])
        
        # Trim to exactly 16 points and normalize properly
        points = points[:16]
        points = np.array(points)
        
        # Normalize to [0,1]² properly
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale and center properly
        points[:, 0] *= 0.8
        points[:, 1] *= 0.8
        points[:, 0] += 0.1
        points[:, 1] += 0.1
        
        return points
    
    def create_multiple_initial_patterns():
        """Create multiple diverse initial patterns for robust optimization"""
        patterns = []
        
        # 1. Hexagonal lattice (inspired by mathematical optimality)
        patterns.append(create_hexagonal_lattice())
        
        # 2. Grid pattern with perturbations
        grid_points = []
        for i in range(4):
            for j in range(4):
                grid_points.append([i/3.0, j/3.0])
        patterns.append(np.array(grid_points[:16]))
        
        # 3. Fibonacci spiral pattern
        fib_points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(16):
            theta = i * 2 * np.pi / phi
            r = np.sqrt(i / 15.0) if i < 15 else 1.0
            x = 0.5 + r * np.cos(theta) * 0.4
            y = 0.5 + r * np.sin(theta) * 0.4
            fib_points.append([x, y])
        patterns.append(np.array(fib_points))
        
        # 4. Concentric rings
        ring_points = []
        # Center
        ring_points.append([0.5, 0.5])
        # Ring 1 - 6 points
        for i in range(6):
            angle = i * np.pi / 3
            ring_points.append([0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)])
        # Ring 2 - 9 points
        for i in range(9):
            angle = i * 2 * np.pi / 9
            ring_points.append([0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)])
        patterns.append(np.array(ring_points[:16]))
        
        # 5. Corner-based pattern
        corner_points = []
        # Corners
        corner_points.extend([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        # Edge centers
        corner_points.extend([[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]])
        # Inner square
        corner_points.extend([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])
        # Additional strategic points
        corner_points.extend([[0.5, 0.3], [0.7, 0.5], [0.5, 0.7], [0.3, 0.5]])
        patterns.append(np.array(corner_points[:16]))
        
        return patterns
    
    def simulated_annealing_optimization(initial_points, max_iterations=20000):
        """Optimize the point configuration using simulated annealing with proper cooling"""
        n = 16
        points = initial_points.copy()
        
        def calculate_ratio(points_array):
            """Calculate min/max distance ratio"""
            distances = pdist(points_array)
            if len(distances) == 0:
                return 0
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max <= 0:
                return 0
            return d_min / d_max
        
        # Simulated Annealing parameters - tuned for better convergence
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # More aggressive cooling schedule for faster convergence
        temp = 0.1
        cooling_rate = 0.996  # Even faster cooling than inspiration 2
        min_temp = 1e-10
        max_iter = max_iterations
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(0, n)
            
            # Adaptive perturbation scales based on iteration progress
            if iteration < max_iter * 0.3:
                perturbation_scale = 0.02  # Large perturbations early
            elif iteration < max_iter * 0.7:
                perturbation_scale = 0.005  # Medium perturbations middle
            else:
                perturbation_scale = 0.001  # Small perturbations late
            
            neighbor_points[point_idx] += np.random.normal(0, perturbation_scale, 2)
            
            # Keep within bounds [0,1]²
            neighbor_points[point_idx] = np.clip(neighbor_points[point_idx], 0, 1)
            
            # Calculate new ratio
            new_ratio = calculate_ratio(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = new_ratio
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if np.random.random() < np.exp(delta / temp):
                    current_points = neighbor_points
                    current_ratio = new_ratio
            
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
                
        return best_points, best_ratio
    
    def local_search_improved(points, max_iterations=2500):
        """Improved local search with better neighborhood exploration"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Track improvement for early stopping
        last_improvement = 0
        improvement_count = 0
        
        for iteration in range(max_iterations):
            improved = False
            # Try moving each point slightly in a random direction
            for i in range(16):
                # Save current point
                old_point = current_points[i].copy()
                
                # Adaptive perturbation scale based on iteration
                if iteration < max_iterations * 0.5:
                    perturbation_scale = 0.002
                else:
                    perturbation_scale = 0.0005
                
                # Try a random movement
                new_points = current_points.copy()
                new_points[i] += np.random.normal(0, perturbation_scale, 2)
                new_points[i] = np.clip(new_points[i], 0, 1)
                
                new_ratio = compute_min_max_ratio(new_points)
                
                if new_ratio > current_ratio:
                    current_points = new_points
                    current_ratio = new_ratio
                    improved = True
                    improvement_count += 1
                    last_improvement = 0
                else:
                    # Restore original point
                    current_points[i] = old_point
            
            # Early stopping if no improvement for a while
            last_improvement += 1
            if not improved:
                last_improvement += 1
            if last_improvement > 500 and iteration > max_iterations * 0.7:
                break
                
        return current_points
    
    def combinatorial_game_theory_approach():
        """
        Fundamental combinatorial game theory approach:
        - Treats point placement as a strategic game between agents
        - Each point tries to maximize its minimum distance to others
        - Uses constraint satisfaction with propagation rules
        - Implements a game-theoretic equilibrium-seeking algorithm
        """
        # Initialize points randomly
        points = np.random.rand(16, 2)
        
        # Game-theoretic approach: each point acts strategically
        # This simulates a multi-agent optimization where points "compete" for optimal positions
        max_iterations = 1000
        learning_rate = 0.1
        
        for iteration in range(max_iterations):
            # Calculate current distances and ratios
            distances = pdist(points)
            if len(distances) == 0:
                break
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # If we're already at a good configuration, stop
            if d_max > 0 and d_min / d_max > 0.15:  # Threshold for early stopping
                break
            
            # For each point, calculate its "payoff" based on distance to neighbors
            # and adjust position to improve it
            for i in range(16):
                # Calculate distances to all other points
                current_point = points[i]
                other_points = np.delete(points, i, axis=0)
                
                # Calculate distances to all other points
                dist_to_others = np.sqrt(np.sum((other_points - current_point)**2, axis=1))
                
                # Find the minimum distance to any other point
                min_dist = np.min(dist_to_others)
                
                # Calculate gradient: move away from closest points, towards others
                if len(dist_to_others) > 0:
                    # Calculate force from nearest neighbors (repulsion)
                    nearest_indices = np.argsort(dist_to_others)[:3]  # Top 3 nearest
                    repulsion_force = np.zeros(2)
                    
                    for idx in nearest_indices:
                        if dist_to_others[idx] > 0:
                            # Repel from close points
                            direction = current_point - other_points[idx]
                            repulsion_force += direction / (dist_to_others[idx] + 1e-8)
                    
                    # Normalize and apply learning rate
                    if np.linalg.norm(repulsion_force) > 0:
                        repulsion_force = repulsion_force / np.linalg.norm(repulsion_force)
                        points[i] += learning_rate * repulsion_force
                    
                    # Keep within bounds
                    points[i] = np.clip(points[i], 0, 1)
        
        return points
    
    def objective_function(x):
        """Minimize negative of min/max distance ratio (equivalent to maximizing the ratio)"""
        # Reshape points
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return 0
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Handle case where all points are coincident
        if d_max == 0:
            return -np.inf
            
        # Return negative ratio to convert maximization to minimization
        return -d_min / d_max
    
    # Strategy 1: Try multiple initial configurations with different approaches
    initial_patterns = create_multiple_initial_patterns()
    best_ratio = -float('inf')
    best_points = None
    
    # Run multiple optimizations with different starting patterns
    for i, initial_pattern in enumerate(initial_patterns):
        try:
            # Create variations of each pattern for diversity
            for restart in range(5):  # Increased restarts for better exploration
                np.random.seed(42 + i * 100 + restart)
                
                # Create slightly varied initial configuration
                varied_initial = initial_pattern.copy()
                # Add perturbations based on pattern type
                if i == 0:  # Hexagonal pattern - more aggressive
                    perturbation = np.random.normal(0, 0.03, varied_initial.shape)
                else:  # Other patterns - less aggressive
                    perturbation = np.random.normal(0, 0.015, varied_initial.shape)
                varied_initial += perturbation
                varied_initial = np.clip(varied_initial, 0.05, 0.95)
                
                # Optimize with SA using moderate iterations for speed
                optimized_points, ratio = simulated_annealing_optimization(varied_initial, max_iterations=12000)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Strategy 2: Try the combinatorial game theory approach
    try:
        game_points = combinatorial_game_theory_approach()
        game_ratio = compute_min_max_ratio(game_points)
        if game_ratio > best_ratio:
            best_ratio = game_ratio
            best_points = game_points.copy()
    except Exception:
        pass
    
    # Strategy 3: Try L-BFGS optimization on the best result so far for fine-tuning
    if best_points is not None:
        try:
            # Use scipy optimization for fine-tuning with reasonable precision
            bounds = [(0, 1) for _ in range(32)]
            result = differential_evolution(
                lambda x: -compute_min_max_ratio(x.reshape(-1, 2)),
                bounds,
                maxiter=50,  # Reduced iterations to stay within time limits
                popsize=15,   # Reduced population size
                seed=42,
                strategy='best1bin'
            )
            
            if result.success:
                de_points = result.x.reshape(-1, 2)
                de_ratio = compute_min_max_ratio(de_points)
                
                if de_ratio > best_ratio:
                    best_points = de_points
                    best_ratio = de_ratio
        except Exception:
            pass
    
    # Strategy 4: Final refinement with enhanced local search
    if best_points is not None:
        try:
            refined_points = local_search_improved(best_points, max_iterations=3000)
            refined_ratio = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio
        except Exception:
            pass
    
    # Fallback to a good hexagonal pattern if nothing worked
    if best_points is None:
        best_points = create_hexagonal_lattice()
    
    return best_points


# EVOLVE-BLOCK-END
