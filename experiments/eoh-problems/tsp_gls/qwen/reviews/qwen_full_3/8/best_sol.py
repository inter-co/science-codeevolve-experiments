# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a simple yet effective penalty strategy.
    
    This implementation uses a straightforward penalty mechanism that:
    1. Increases distances for edges that have been used frequently
    2. Applies stronger penalties to edges that are part of the current tour
    3. Uses a multiplicative penalty scheme that's computationally efficient
    
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
    updated_edge_distance = edge_distance.copy()
    
    # Get problem size
    n = len(local_opt_tour)
    
    # Convert edge_n_used to symmetric matrix
    edge_n_used = (edge_n_used + edge_n_used.T) / 2
    
    # Create a binary matrix indicating which edges are part of the current tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    # Handle the last edge connecting back to start
    a, b = local_opt_tour[-1], local_opt_tour[0]
    tour_edges[a, b] = True
    tour_edges[b, a] = True
    
    # Calculate penalty parameters
    max_used = np.max(edge_n_used)
    if max_used == 0:
        # No edges have been used before, so no penalty
        return updated_edge_distance
    
    # Normalize usage counts to [0, 1] range
    normalized_used = edge_n_used / max_used
    
    # Define penalty multipliers
    # Base penalty multiplier for frequently used edges
    base_penalty = 1.0 + normalized_used * 5.0
    
    # Increase penalty for edges in the current tour
    # These are typically more "valuable" to keep in the tour
    tour_penalty_multiplier = 1.0 + tour_edges * 3.0
    
    # Apply combined penalties
    penalty_matrix = base_penalty * tour_penalty_multiplier
    
    # Apply penalties to both directions (make sure it's symmetric)
    updated_edge_distance = updated_edge_distance * penalty_matrix
    
    # Ensure minimum distance increase to prevent numerical issues
    # and maintain reasonable exploration
    min_increase_factor = 1.05
    updated_edge_distance = np.maximum(updated_edge_distance, edge_distance * min_increase_factor)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
