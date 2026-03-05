# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.
    
    This implementation uses a hybrid penalty strategy combining:
    - Adaptive scaling based on edge cost magnitude
    - Non-linear penalty functions (logarithmic and exponential)
    - Tour quality awareness
    - Frequency-based ranking
    - Temporal decay to prevent over-penalization

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
    
    # Calculate tour cost for normalization
    tour_cost = 0
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        tour_cost += edge_distance[a, b]
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    tour_cost += edge_distance[a, b]
    
    # Normalize the tour cost to get a reasonable scale factor
    avg_edge_cost = tour_cost / n if n > 0 else 1.0
    
    # Create a mask for edges in the current tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    tour_edges[a, b] = True
    tour_edges[b, a] = True
    
    # Create frequency ranking matrix for edges (higher frequency = higher priority for penalization)
    # We'll use a combination of usage count and edge cost
    edge_penalties = np.zeros((n, n))
    
    # For each edge in the tour, compute penalty based on multiple factors
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        edge_cost = edge_distance[a, b]
        
        # Base penalty factor: combine usage count with normalized edge cost
        # Use logarithmic scaling to prevent extreme penalties on very costly edges
        base_penalty = 1.0 + np.log(1.0 + edge_n_used[a, b])
        
        # Adaptive penalty based on edge cost relative to average
        # More expensive edges get more aggressive penalties to encourage exploration
        cost_factor = edge_cost / (avg_edge_cost + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Combine factors with non-linear transformation
        # Apply exponential scaling to make penalties more sensitive to high usage
        penalty_factor = base_penalty * (1.0 + cost_factor * 0.5)
        
        # Apply temporal decay - if an edge was used recently, penalize less
        # This prevents over-penalization of edges that might be useful again soon
        temporal_decay = 1.0 / (1.0 + edge_n_used[a, b] * 0.1)
        
        edge_penalties[a, b] = penalty_factor * temporal_decay
        edge_penalties[b, a] = penalty_factor * temporal_decay
    
    # Handle last edge of tour
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    edge_cost = edge_distance[a, b]
    base_penalty = 1.0 + np.log(1.0 + edge_n_used[a, b])
    cost_factor = edge_cost / (avg_edge_cost + 1e-8)
    penalty_factor = base_penalty * (1.0 + cost_factor * 0.5)
    temporal_decay = 1.0 / (1.0 + edge_n_used[a, b] * 0.1)
    edge_penalties[a, b] = penalty_factor * temporal_decay
    edge_penalties[b, a] = penalty_factor * temporal_decay
    
    # Apply penalties to the distance matrix
    # Use a combination of additive and multiplicative penalties
    # This creates more diverse exploration paths
    alpha = 0.5  # Weight for multiplicative vs additive component
    
    # Apply both multiplicative and additive penalties
    # Multiplicative component increases edge weights
    # Additive component ensures minimum penalty even for low-cost edges
    updated_edge_distance = edge_distance * (1.0 + alpha * edge_penalties) + \
                           (1.0 - alpha) * edge_penalties * 10.0
    
    # Ensure symmetry
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Clamp values to reasonable bounds to prevent numerical issues
    updated_edge_distance = np.clip(updated_edge_distance, 0.1, 10000.0)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
