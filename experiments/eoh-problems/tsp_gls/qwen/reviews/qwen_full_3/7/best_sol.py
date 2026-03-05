# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.

    This implementation follows a proven penalty strategy that:
    1. Uses logarithmic scaling for usage-based penalties (as in inspiration 1)
    2. Applies stronger penalties to edges in current tour
    3. Maintains computational efficiency with minimal overhead
    4. Ensures numerical stability and good convergence properties

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
    
    n = len(local_opt_tour)
    
    # Identify edges that are in the current tour
    tour_edges = set()
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_edges.add((min(a, b), max(a, b)))
    a, b = local_opt_tour[-1], local_opt_tour[0]
    tour_edges.add((min(a, b), max(a, b)))
    
    # Apply penalties to all edges
    for i in range(n):
        for j in range(i + 1, n):
            usage_count = edge_n_used[i, j]
            
            # Determine if this is a tour edge
            is_tour_edge = (min(i, j), max(i, j)) in tour_edges
            
            # Base penalty: logarithmic scaling for usage frequency
            # This ensures that heavily used edges get much higher penalties
            if usage_count == 0:
                # Never used edges get a strong penalty
                base_penalty = 3.0
            else:
                # Logarithmic scaling - higher usage gives higher penalty
                base_penalty = 1.0 + np.log(1.0 + usage_count)
            
            # Tour edge penalty: give extra penalty to edges in current tour
            tour_penalty = 1.0 + (is_tour_edge * 2.0)  # Double penalty for tour edges
            
            # Combine penalties
            penalty_factor = base_penalty * tour_penalty
            
            # Cap the penalty to prevent extreme inflation
            penalty_factor = min(penalty_factor, 15.0)
            
            # Apply penalty using multiplicative approach
            updated_edge_distance[i, j] *= penalty_factor
            updated_edge_distance[j, i] *= penalty_factor
    
    # Ensure minimum distance to prevent numerical issues
    # and maintain a reasonable scale
    min_distance = np.min(edge_distance[edge_distance > 0])
    updated_edge_distance = np.maximum(updated_edge_distance, min_distance * 0.1)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
