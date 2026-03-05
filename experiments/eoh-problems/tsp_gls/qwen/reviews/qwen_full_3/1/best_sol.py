# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a robust multi-level penalty strategy for GLS-TSP.
    
    This implementation balances:
    1. Strong penalty differentiation for frequently used edges
    2. Memory-aware decay to prevent over-penalization
    3. Exploration encouragement for unused edges
    4. Numerical stability and computational efficiency

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
    
    # Calculate usage-based penalty parameters
    avg_usage = total_penalties / (n * (n - 1) / 2) if total_penalties > 0 else 0
    
    # Define adaptive penalty intensity based on search progress
    penalty_base = 1.0 + 0.5 * np.log(1.0 + max_usage) + 0.1 * avg_usage
    
    # Create adjacency matrix for tour edges
    tour_edges = set()
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(a, b), max(a, b)))
    
    # Apply penalties using a tiered approach
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        
        # Get usage counts for both directions
        usage_a_b = edge_n_used[a, b]
        usage_b_a = edge_n_used[b, a]
        usage_count = max(usage_a_b, usage_b_a)
        
        # Different penalty weights based on edge type and usage
        if (a, b) in tour_edges or (b, a) in tour_edges:
            # Tour edges - heavy penalty for frequent usage
            penalty_multiplier = penalty_base * (1.0 + 0.5 * np.log(1.0 + usage_count))
        elif usage_count > 0:
            # Frequently used non-tour edges - moderate penalty
            penalty_multiplier = penalty_base * (1.0 + 0.3 * np.log(1.0 + usage_count))
        else:
            # Rarely used edges - light penalty
            penalty_multiplier = penalty_base * (1.0 + 0.1 * np.log(1.0 + usage_count))
        
        # Apply penalty with bounded growth to prevent explosion
        penalty_multiplier = min(penalty_multiplier, 15.0)
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
    
    # Add exploration encouragement for low-usage edges
    # This helps maintain diversity in the search process
    low_usage_threshold = max(1, avg_usage * 0.5)
    low_usage_mask = (edge_n_used < low_usage_threshold) & (edge_n_used > 0)
    
    # Encourage exploration of edges that are underused
    for i in range(n):
        for j in range(i + 1, n):
            if low_usage_mask[i, j]:
                # Slightly reduce distance to encourage exploration
                updated_edge_distance[i, j] *= 0.99
                updated_edge_distance[j, i] *= 0.99
    
    # Add small randomization for diversification
    # This ensures different search paths even with same penalty structure
    np.random.seed(int(np.sum(edge_n_used) % 1000000))
    random_factor = 1.0 + 0.005 * (np.random.random() - 0.5)
    updated_edge_distance *= random_factor
    
    # Ensure numerical stability with proper clipping
    updated_edge_distance = np.clip(updated_edge_distance, 1e-6, 1e6)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
