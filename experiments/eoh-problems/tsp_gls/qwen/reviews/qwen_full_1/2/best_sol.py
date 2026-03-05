# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.
    
    This implementation follows the Voudouris & Tsang approach with improvements:
    - Logarithmic penalty scaling to prevent extreme values
    - Temporal decay to avoid over-penalization
    - Proper symmetry maintenance
    - Reasonable bounds to prevent numerical issues

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    n = len(local_opt_tour)
    
    # Create a copy of the distance matrix to modify
    updated_edge_distance = edge_distance.copy()
    
    # Create penalty matrix for tour edges
    edge_penalties = np.zeros((n, n))
    
    # For each edge in the tour, compute penalty based on usage and cost
    tour_edges = []
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        tour_edges.append((a, b))
        
        # Base penalty factor: logarithmic scaling to prevent extreme penalties
        base_penalty = 1.0 + np.log(1.0 + edge_n_used[a, b])
        
        # Apply temporal decay to prevent over-penalization of recently used edges
        temporal_decay = 1.0 / (1.0 + 0.1 * edge_n_used[a, b])
        
        # Store penalty for both directions
        penalty = base_penalty * temporal_decay
        edge_penalties[a, b] = penalty
        edge_penalties[b, a] = penalty
    
    # Handle last edge of tour
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    tour_edges.append((a, b))
    
    # Base penalty factor: logarithmic scaling to prevent extreme penalties
    base_penalty = 1.0 + np.log(1.0 + edge_n_used[a, b])
    
    # Apply temporal decay to prevent over-penalization of recently used edges
    temporal_decay = 1.0 / (1.0 + 0.1 * edge_n_used[a, b])
    
    # Store penalty for both directions
    penalty = base_penalty * temporal_decay
    edge_penalties[a, b] = penalty
    edge_penalties[b, a] = penalty
    
    # Apply penalties multiplicatively to the distance matrix
    # This preserves the relative structure while increasing distances
    updated_edge_distance = updated_edge_distance * (1.0 + edge_penalties)
    
    # Ensure symmetry
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Set minimum distance to prevent numerical issues
    min_distance = np.min(edge_distance[edge_distance > 0]) if np.any(edge_distance > 0) else 1.0
    updated_edge_distance = np.maximum(updated_edge_distance, min_distance)
    
    # Cap maximum values to prevent extreme penalties
    max_distance = np.max(edge_distance) * 100.0
    updated_edge_distance = np.minimum(updated_edge_distance, max_distance)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
