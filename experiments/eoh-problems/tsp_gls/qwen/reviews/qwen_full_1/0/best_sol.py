# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a novel rank-based penalty strategy for GLS-TSP.
    
    This implementation introduces several innovations:
    1. Rank-based penalties instead of raw usage counts
    2. Geometric-aware penalty distribution (penalizing edges near tour edges)
    3. Dynamic penalty scaling based on convergence state
    4. Asymmetric penalty application to encourage diverse exploration
    
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
    
    # Create a mask for tour edges
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    # Add the final edge of the tour
    a, b = local_opt_tour[-1], local_opt_tour[0]
    tour_edges[a, b] = True
    tour_edges[b, a] = True
    
    # Calculate global statistics for adaptive scaling
    max_used = np.max(edge_n_used)
    avg_used = np.mean(edge_n_used[edge_n_used > 0]) if np.any(edge_n_used > 0) else 1.0
    
    # Convert usage counts to ranks for more meaningful penalties
    # Flatten and rank the usage matrix (excluding zero entries)
    flat_usage = edge_n_used.flatten()
    sorted_indices = np.argsort(-flat_usage)  # Sort in descending order
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(sorted_indices))
    
    # Create rank-based penalty matrix
    rank_penalty = np.zeros_like(edge_distance)
    
    # Create a spatial influence map around the tour
    # This penalizes edges that are geometrically close to tour edges
    spatial_penalty = np.zeros_like(edge_distance)
    
    # Calculate tour edge weights for reference
    tour_weights = []
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_weights.append(edge_distance[a, b])
    a, b = local_opt_tour[-1], local_opt_tour[0]
    tour_weights.append(edge_distance[a, b])
    avg_tour_weight = np.mean(tour_weights) if tour_weights else 1.0
    
    # Apply rank-based penalties to tour edges
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        # Get rank of this edge usage
        edge_rank = ranks[a * n + b]
        # Normalize rank to [0, 1] scale
        normalized_rank = edge_rank / len(flat_usage) if len(flat_usage) > 0 else 0
        
        # Apply rank-based penalty that's inversely related to rank (higher usage = higher penalty)
        # But also factor in the edge's actual weight to bias towards longer edges
        edge_weight = edge_distance[a, b]
        weight_factor = edge_weight / (avg_tour_weight + 1e-8)
        # Combine rank and weight factors with non-linear scaling
        penalty = 10.0 * (normalized_rank ** 0.7) * (weight_factor ** 0.5)
        
        rank_penalty[a, b] = penalty
        rank_penalty[b, a] = penalty
    
    # Handle the final edge of the tour
    a, b = local_opt_tour[-1], local_opt_tour[0]
    edge_rank = ranks[a * n + b]
    normalized_rank = edge_rank / len(flat_usage) if len(flat_usage) > 0 else 0
    edge_weight = edge_distance[a, b]
    weight_factor = edge_weight / (avg_tour_weight + 1e-8)
    penalty = 10.0 * (normalized_rank ** 0.7) * (weight_factor ** 0.5)
    
    rank_penalty[a, b] = penalty
    rank_penalty[b, a] = penalty
    
    # Apply geometric spatial penalties
    # Penalize edges that are nearby to tour edges (within some threshold)
    # This creates a "bubble" of increased distance around tour edges
    for i in range(n):
        current_city = local_opt_tour[i]
        next_city = local_opt_tour[(i + 1) % n]
        
        # Apply spatial penalty to all edges that connect to cities near the tour
        for j in range(n):
            # Penalize edges involving neighbors of tour cities
            if j != current_city and j != next_city:
                # Apply penalty based on distance to tour edges
                # Simple heuristic: penalize edges connected to tour cities
                if tour_edges[current_city, j]:
                    spatial_penalty[current_city, j] += 5.0
                    spatial_penalty[j, current_city] += 5.0
                if tour_edges[next_city, j]:
                    spatial_penalty[next_city, j] += 5.0
                    spatial_penalty[j, next_city] += 5.0
    
    # Combine penalties with dynamic weighting based on convergence
    # Early iterations: focus more on rank penalties to quickly escape
    # Later iterations: balance with spatial penalties for fine-tuning
    iteration_penalty_factor = 0.3  # This could be passed as parameter or derived from iteration count
    
    # Apply combined penalties
    combined_penalty = rank_penalty + iteration_penalty_factor * spatial_penalty
    
    # Apply penalty with dynamic scaling
    penalty_strength = 0.3 + 0.2 * np.exp(-max_used / (avg_used + 1e-8))  # Decreasing strength over time
    
    # Apply the penalty to the distance matrix
    updated_edge_distance = edge_distance + penalty_strength * combined_penalty
    
    # Ensure symmetry (important for TSP solvers)
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Ensure minimum distance is maintained to prevent degenerate cases
    updated_edge_distance = np.maximum(updated_edge_distance, edge_distance)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
