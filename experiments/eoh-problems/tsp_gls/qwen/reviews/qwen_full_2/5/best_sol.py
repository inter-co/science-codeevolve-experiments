# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.
    
    This implementation combines the best elements from successful approaches:
    1. Logarithmic penalty scaling for usage frequency (inspired by Inspiration 2)
    2. Stronger penalties for tour edges (inspired by Inspiration 2)
    3. Cost-aware penalties that favor longer edges
    4. Temporal decay to prevent over-penalization
    5. Improved numerical stability and symmetry preservation

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    # Create a copy to avoid modifying the original matrix
    updated_edge_distance = edge_distance.copy().astype(float)
    
    # Get the number of nodes
    n = len(local_opt_tour)
    
    # Vectorized approach to extract all tour edges and their properties
    tour_edges_a = local_opt_tour[:-1]
    tour_edges_b = local_opt_tour[1:]
    # Handle the last edge (return to start)
    tour_edges_a = np.append(tour_edges_a, local_opt_tour[-1])
    tour_edges_b = np.append(tour_edges_b, local_opt_tour[0])
    
    # Extract edge distances for cost-aware component
    tour_edge_distances = edge_distance[tour_edges_a, tour_edges_b]
    
    # Normalize usage counts for penalty calculation
    max_usage = np.max(edge_n_used)
    if max_usage > 0:
        normalized_usage = edge_n_used / max_usage
    else:
        normalized_usage = edge_n_used.astype(float)
    
    # Apply logarithmic penalty scaling for usage frequency (Inspiration 2)
    # Emphasizes rare usage while providing reasonable penalties for moderate usage
    usage_penalty = 1.0 + 1.5 * np.log1p(normalized_usage)
    
    # Apply stronger penalties to tour edges to escape current tour structure (Inspiration 2)
    # Create mask for tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    tour_mask[tour_edges_a, tour_edges_b] = True
    tour_mask[tour_edges_b, tour_edges_a] = True
    tour_multiplier = 1.0 + 2.0 * tour_mask  # Stronger penalty for tour edges
    
    # Combine penalties
    penalty_factor = usage_penalty * tour_multiplier
    
    # Apply temporal decay to prevent over-penalization of recently used edges
    # This prevents the algorithm from getting stuck in cycles
    temporal_decay = 0.95
    decay_factor = temporal_decay ** (edge_n_used)
    penalty_factor = penalty_factor * decay_factor
    
    # Apply penalties to both directions to maintain symmetry
    updated_edge_distance = edge_distance * penalty_factor
    
    # Add cost awareness: make longer edges slightly more penalized
    # This adds the cost-aware component from the target but with simpler logic
    avg_tour_cost = np.mean(tour_edge_distances) if len(tour_edge_distances) > 0 else 1.0
    cost_component = 1.0 + 0.3 * (tour_edge_distances / (avg_tour_cost + 1e-8))
    
    # Apply cost adjustments specifically to tour edges
    updated_edge_distance[tour_edges_a, tour_edges_b] *= cost_component
    updated_edge_distance[tour_edges_b, tour_edges_a] *= cost_component
    
    # Ensure symmetry by averaging with transpose
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Ensure minimum distance to prevent numerical issues
    # Use a more conservative minimum penalty
    min_penalty_factor = 1.05
    updated_edge_distance = np.maximum(updated_edge_distance, edge_distance * min_penalty_factor)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
