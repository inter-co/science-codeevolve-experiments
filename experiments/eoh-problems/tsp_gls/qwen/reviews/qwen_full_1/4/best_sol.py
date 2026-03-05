# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a novel rank-based and structure-aware GLS penalty strategy.
    
    This implementation uses a fundamentally different approach:
    1. Rank-based penalty system that penalizes high-frequency edges more aggressively
    2. Structure-aware penalties that consider tour connectivity and edge importance
    3. Stochastic exploration to encourage diversification
    4. Distance-normalized penalties that scale with problem geometry

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
    
    # Convert edge_n_used to a flattened array for ranking
    flat_used = edge_n_used.flatten()
    
    # Create rank-based penalty: higher rank = more frequent = more penalized
    # Use argsort to get indices that would sort the values
    sorted_indices = np.argsort(flat_used)[::-1]  # Descending order (most used first)
    
    # Create rank mapping (0 = most frequent, n*(n-1)/2-1 = least frequent)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(sorted_indices))
    
    # Reshape back to matrix form
    rank_matrix = ranks.reshape(n, n)
    
    # Apply rank-based penalty with exponential scaling for stronger penalties on highly used edges
    # This creates a more aggressive penalty for edges that have been used too much
    rank_penalty = 1.0 + 2.0 * (rank_matrix / np.max(rank_matrix)) ** 2
    
    # Apply tour-specific penalties (additive for tour edges)
    tour_penalty = 1.0 + 1.5 * tour_mask.astype(float)  # 2.5 for tour edges, 1.0 otherwise
    
    # Combine penalties with a hybrid approach
    # Tour edges get additional penalty + rank-based penalty
    penalty_factor = tour_penalty * rank_penalty
    
    # Add stochastic component for exploration
    # Random factor that helps escape local optima (small probability of large penalty)
    stochastic_factor = 1.0 + 0.3 * np.random.rand(n, n)
    
    # Apply the combined penalty with stochastic element
    updated_edge_distance = edge_distance * penalty_factor * stochastic_factor
    
    # Add structured penalty for edges that form cycles or are critical for tour structure
    # This helps in breaking existing tour patterns more effectively
    cycle_penalty = np.zeros((n, n))
    
    # Identify edges that are likely part of cycles or critical for maintaining tour structure
    # by examining adjacency relationships in the current tour
    for i in range(n):
        prev_node = local_opt_tour[(i - 1) % n]
        next_node = local_opt_tour[(i + 1) % n]
        
        # These adjacent edges in the tour are particularly important to penalize
        cycle_penalty[prev_node, local_opt_tour[i]] += 0.5
        cycle_penalty[local_opt_tour[i], next_node] += 0.5
    
    # Normalize cycle penalty and apply it
    if np.max(cycle_penalty) > 0:
        cycle_penalty = cycle_penalty / np.max(cycle_penalty) * 0.8
    
    updated_edge_distance = updated_edge_distance * (1.0 + cycle_penalty)
    
    # Ensure symmetry (recommended for TSP)
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Add minimum penalty to ensure accessibility
    min_penalty = 0.01 * np.max(edge_distance)
    updated_edge_distance += min_penalty
    
    # Clip extreme values to maintain numerical stability
    max_dist = np.max(edge_distance)
    updated_edge_distance = np.clip(updated_edge_distance, edge_distance.min(), max_dist * 10)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
