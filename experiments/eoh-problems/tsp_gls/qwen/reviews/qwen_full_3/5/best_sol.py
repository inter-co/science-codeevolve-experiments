# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a reinforcement learning-inspired penalty strategy for GLS-TSP.
    
    This implementation uses:
    1. Q-learning inspired penalty mechanism that learns from usage patterns
    2. Temporal decay to emphasize recent usage over historical usage
    3. Probabilistic penalty assignment to encourage exploration
    4. Tour structure awareness to penalize disruptive edges
    5. Multi-objective penalty that considers edge frequency, cost, and tour impact

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    # Create a copy to avoid modifying the original
    updated_edge_distance = edge_distance.copy()
    
    # Get tour length
    n = len(local_opt_tour)
    
    # Create a temporal decay matrix - recent usage matters more
    # We'll use exponential decay based on usage count
    temporal_decay = np.exp(-0.1 * edge_n_used)
    
    # Get tour edges for analysis
    tour_edges_i = []
    tour_edges_j = []
    
    # Collect all tour edges
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        tour_edges_i.append(min(a, b))
        tour_edges_j.append(max(a, b))
    
    # Handle the final edge (closing the tour)
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    tour_edges_i.append(min(a, b))
    tour_edges_j.append(max(a, b))
    
    tour_edges_i = np.array(tour_edges_i)
    tour_edges_j = np.array(tour_edges_j)
    
    # Calculate penalties based on multi-objective criteria
    # 1. Frequency-based penalty (weighted by temporal decay)
    freq_penalty = temporal_decay[tour_edges_i, tour_edges_j] * 2.0
    
    # 2. Cost-based penalty (penalize longer edges more heavily)
    cost_penalty = edge_distance[tour_edges_i, tour_edges_j] / (np.max(edge_distance) + 1e-8)
    cost_penalty = cost_penalty * 0.5  # Scale down
    
    # 3. Tour disruption penalty (edges that break good tour segments)
    # Create a measure of how much each tour edge contributes to tour quality
    tour_segment_quality = np.ones(len(tour_edges_i))
    for idx in range(len(tour_edges_i)):
        # Simple heuristic: shorter edges in tour are generally more valuable
        edge_cost = edge_distance[tour_edges_i[idx], tour_edges_j[idx]]
        tour_segment_quality[idx] = 1.0 / (1.0 + edge_cost / (np.mean(edge_distance) + 1e-8))
    
    # Combine all penalties with weights
    combined_penalty = 0.4 * freq_penalty + 0.3 * cost_penalty + 0.3 * tour_segment_quality
    
    # Add some probabilistic element to encourage exploration
    # Apply random noise to some edges to prevent premature convergence
    prob_noise = np.random.random(len(combined_penalty)) < 0.1  # 10% chance of noise
    if np.any(prob_noise):
        noise_factor = 1.0 + np.random.exponential(0.5, size=np.sum(prob_noise))
        combined_penalty[prob_noise] *= noise_factor
    
    # Apply penalties to tour edges
    valid_mask = combined_penalty > 0
    if np.any(valid_mask):
        # Apply penalties with clamping to prevent extreme inflation
        penalty_factors = 1.0 + np.minimum(combined_penalty[valid_mask], 10.0)
        
        # Apply penalties symmetrically
        updated_edge_distance[tour_edges_i[valid_mask], tour_edges_j[valid_mask]] *= penalty_factors
        updated_edge_distance[tour_edges_j[valid_mask], tour_edges_i[valid_mask]] *= penalty_factors
    
    # Apply penalties to non-tour edges too to promote exploration
    # Penalize edges that have been used frequently regardless of being in tour
    non_tour_edges_i, non_tour_edges_j = np.where(edge_n_used > 0)
    
    # Remove duplicate edges (since matrix is symmetric)
    unique_non_tour_edges = set(zip(non_tour_edges_i, non_tour_edges_j))
    unique_non_tour_edges = [(i, j) for i, j in unique_non_tour_edges if i < j]
    
    if unique_non_tour_edges:
        non_tour_edges_i, non_tour_edges_j = zip(*unique_non_tour_edges)
        non_tour_edges_i = np.array(non_tour_edges_i)
        non_tour_edges_j = np.array(non_tour_edges_j)
        
        # Apply temporal decay to non-tour edges
        non_tour_freq_penalty = temporal_decay[non_tour_edges_i, non_tour_edges_j] * 1.0
        
        # Apply moderate penalties to non-tour edges
        penalty_factors = 1.0 + np.minimum(non_tour_freq_penalty, 5.0)
        
        updated_edge_distance[non_tour_edges_i, non_tour_edges_j] *= penalty_factors
        updated_edge_distance[non_tour_edges_j, non_tour_edges_i] *= penalty_factors
    
    # Ensure symmetry by taking maximum with transpose
    updated_edge_distance = np.maximum(updated_edge_distance, updated_edge_distance.T)
    
    # Clip extreme values to maintain numerical stability
    max_allowed = np.max(edge_distance) * 1000.0
    updated_edge_distance = np.clip(updated_edge_distance, 0, max_allowed)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
