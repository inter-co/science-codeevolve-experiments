# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.

    Optimized scoring function that balances tight fitting, load balancing,
    fragmentation avoidance, and future packing potential.

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    if len(bins) == 0:
        return np.array([])
    
    # Calculate core metrics
    remaining = bins - item
    utilization = (100.0 - remaining) / 100.0  # Utilization ratio (0 to 1)
    
    # Base tight-fit score - prefer bins where item fits most tightly
    tight_fit_score = 1.0 / (remaining + 1.0)
    
    # Load balancing - prefer bins with moderate utilization (around 50%)
    load_balance_score = 1.0 - np.abs(utilization - 0.5) ** 2
    
    # Future packing potential - prefer bins with medium capacity for flexibility
    future_potential = 1.0 - (np.abs(remaining - 50.0) / 50.0) ** 2
    
    # Fragmentation avoidance - penalize bins with very small remaining capacity
    # This helps prevent creating many small unusable gaps
    fragmentation_penalty = np.zeros_like(remaining)
    small_capacity_mask = remaining < 10
    fragmentation_penalty[small_capacity_mask] = -0.2 * (10 - remaining[small_capacity_mask]) / 10
    
    # Penalties for extreme utilization states
    # Strong penalty for bins that would become nearly full (to prevent waste)
    near_full_penalty = np.where(remaining < 5, -0.1, 0.0)
    
    # Moderate penalty for bins that would become nearly empty (to encourage consolidation)
    near_empty_penalty = np.where(remaining > 95, -0.05, 0.0)
    
    # Combine all components with refined weights
    scores = (0.6 * tight_fit_score + 
              0.25 * load_balance_score + 
              0.1 * future_potential + 
              0.05 * fragmentation_penalty + 
              near_full_penalty + 
              near_empty_penalty)
    
    # Ensure all scores are positive and reasonable
    scores = np.maximum(scores, 1e-10)
    
    return scores

# EVOLVE-BLOCK-END

