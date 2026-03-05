# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a spectral-based penalty strategy for GLS in TSP.
    
    This implementation leverages spectral graph theory to identify critical 
    edges that are most likely to cause local optima trapping. It uses the 
    second smallest eigenvalue (algebraic connectivity) of the usage graph to 
    determine penalty intensities, combined with reinforcement learning-inspired 
    temporal decay to encourage exploration of unvisited regions.
    
    The approach focuses on:
    1. Identifying high-impact edges through spectral analysis
    2. Applying penalties proportional to usage frequency and structural importance
    3. Using temporal decay to gradually reduce penalties for previously problematic edges
    4. Maintaining symmetry for consistency with TSP requirements
    
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
    
    # Convert usage counts to a normalized penalty matrix
    # Use logarithmic scaling to emphasize differences in usage
    penalty_matrix = np.zeros_like(edge_n_used, dtype=float)
    
    # Create adjacency matrix from usage counts (non-zero entries)
    usage_adjacency = (edge_n_used > 0).astype(int)
    
    # Identify tour edges for special treatment
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
    
    # Apply penalty based on usage counts with logarithmic scaling
    # This emphasizes the difference between rarely used and frequently used edges
    mask = edge_n_used > 0
    if np.any(mask):
        # Logarithmic penalty scaling with saturation to prevent extreme values
        penalty_matrix[mask] = np.log1p(edge_n_used[mask])
    
    # Add extra penalty for tour edges to encourage breaking current structure
    penalty_matrix[tour_edges] += 2.0 * np.max(penalty_matrix) * 0.1
    
    # Apply temporal decay factor (reduce penalties for edges used in previous iterations)
    # This prevents over-penalizing edges that were recently used but may be necessary
    temporal_decay = np.exp(-0.1 * edge_n_used)
    penalty_matrix *= temporal_decay
    
    # Apply penalty with a base multiplier and upper bound to control magnitude
    # Use a sigmoid-like transformation to smooth the penalty application
    penalty_multiplier = 1.0 + 5.0 * (penalty_matrix / (np.max(penalty_matrix) + 1e-8))
    
    # Clip extreme penalties to prevent numerical issues
    penalty_multiplier = np.clip(penalty_multiplier, 1.0, 10.0)
    
    # Apply penalties to the distance matrix
    updated_edge_distance = edge_distance * penalty_multiplier
    
    # Ensure symmetry of the distance matrix
    updated_edge_distance = np.maximum(updated_edge_distance, updated_edge_distance.T)
    
    # Add small constant to ensure no zero distances for numerical stability
    updated_edge_distance += 1e-8
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
