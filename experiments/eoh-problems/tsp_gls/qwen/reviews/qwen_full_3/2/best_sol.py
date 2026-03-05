# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a robust, efficient penalty strategy for GLS-TSP.
    
    This implementation focuses on simplicity, efficiency, and proven effectiveness:
    - Logarithmic penalty scaling for usage-based diversification
    - Memory-aware decay to prevent over-penalization
    - Bounded growth to maintain numerical stability
    - Clean vectorized operations for performance

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    # Create a copy of the original distance matrix
    updated_edge_distance = edge_distance.copy().astype(float)
    
    # Get the number of nodes
    n = len(local_opt_tour)
    
    # Early return if no edges have been used
    if np.sum(edge_n_used) == 0:
        return updated_edge_distance
    
    # Calculate global statistics for adaptive penalty scaling
    total_penalties = np.sum(edge_n_used)
    max_usage = np.max(edge_n_used)
    avg_usage = total_penalties / (n * (n - 1) / 2) if total_penalties > 0 else 0
    
    # Define penalty parameters based on search progress
    # Higher penalty intensity for higher usage levels
    penalty_intensity = 1.0 + 0.5 * np.log(1.0 + max_usage) + 0.1 * avg_usage
    
    # Create list of tour edges
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append((a, b))
    
    # Apply penalties to tour edges
    for a, b in tour_edges:
        # Get usage count for both directions
        usage_count = max(edge_n_used[a, b], edge_n_used[b, a])
        
        # Apply logarithmic penalty with bounded growth
        penalty_multiplier = penalty_intensity * (1.0 + 0.5 * np.log(1.0 + usage_count))
        
        # Cap the penalty to prevent explosive increases
        penalty_multiplier = min(penalty_multiplier, 15.0)
        
        # Apply penalty to both directions to maintain symmetry
        updated_edge_distance[a, b] *= penalty_multiplier
        updated_edge_distance[b, a] *= penalty_multiplier
    
    # Apply memory-aware decay to prevent over-penalization
    # Gradually reduce penalties for edges that have been used less recently
    decay_factor = 0.95  # Decay rate
    for i in range(n):
        for j in range(i+1, n):
            if edge_n_used[i, j] > 0:
                # Apply decay based on usage count
                decay = decay_factor ** edge_n_used[i, j]
                updated_edge_distance[i, j] *= decay
                updated_edge_distance[j, i] *= decay
    
    # Ensure numerical stability with proper bounds
    updated_edge_distance = np.clip(updated_edge_distance, 1e-6, 1e6)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
