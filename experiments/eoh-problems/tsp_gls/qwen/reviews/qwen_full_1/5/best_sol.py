# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a rank-based and usage-aware penalty strategy.
    
    This implementation uses:
    1. Rank-based penalties that prioritize penalizing most frequently used edges
    2. Tour-specific penalties that heavily penalize current tour edges
    3. Logarithmic scaling for smooth penalty growth
    4. Proper normalization and symmetry maintenance

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
    
    # Create a copy of the original distance matrix
    updated_edge_distance = edge_distance.copy()
    
    # Create a mask for tour edges (both directions)
    tour_mask = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_mask[a, b] = True
        tour_mask[b, a] = True
    # Handle the final edge (return to start)
    a, b = local_opt_tour[-1], local_opt_tour[0]
    tour_mask[a, b] = True
    tour_mask[b, a] = True
    
    # Flatten the usage matrix for ranking
    flat_used = edge_n_used.flatten()
    
    # Create rank-based penalty: higher rank = more frequent = more penalized
    # Use argsort to get indices that would sort the values (descending)
    sorted_indices = np.argsort(flat_used)[::-1]  # Most used first
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(sorted_indices))
    
    # Reshape back to matrix form
    rank_matrix = ranks.reshape(n, n)
    
    # Normalize rank matrix to [0, 1] range
    max_rank = np.max(rank_matrix)
    if max_rank > 0:
        normalized_rank = rank_matrix / max_rank
    else:
        normalized_rank = np.zeros_like(rank_matrix)
    
    # Apply rank-based penalty with logarithmic scaling for smoother growth
    # Base penalty of 1.0, with additional penalty based on usage rank
    rank_penalty = 1.0 + 2.0 * np.log(1.0 + normalized_rank * 5.0)
    
    # Apply tour-specific penalties (stronger penalty for tour edges)
    tour_penalty = 1.0 + 3.0 * tour_mask.astype(float)  # 4.0 for tour edges, 1.0 otherwise
    
    # Combine penalties: tour edges get both rank and tour penalties
    penalty_factor = tour_penalty * rank_penalty
    
    # Apply the combined penalty to the distance matrix
    updated_edge_distance = edge_distance * penalty_factor
    
    # Apply additional penalty to non-tour edges that have been used frequently
    # This helps prevent cycling back to previously explored areas
    non_tour_mask = ~tour_mask
    non_tour_used = edge_n_used * non_tour_mask
    
    # Apply logarithmic penalty for non-tour edges with high usage
    high_usage_mask = non_tour_used > 0
    if np.any(high_usage_mask):
        # Use log scaling to keep penalty growth manageable
        log_penalty = 1.0 + 0.5 * np.log(1.0 + non_tour_used[high_usage_mask])
        updated_edge_distance[high_usage_mask] *= log_penalty
    
    # Ensure symmetry (recommended for TSP)
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Add minimum penalty to ensure all edges remain accessible
    # This prevents any edge from becoming completely inaccessible
    min_penalty = 0.01 * np.max(edge_distance)
    updated_edge_distance += min_penalty
    
    # Clip extreme values to maintain numerical stability
    max_dist = np.max(edge_distance)
    updated_edge_distance = np.clip(updated_edge_distance, edge_distance.min(), max_dist * 5.0)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
