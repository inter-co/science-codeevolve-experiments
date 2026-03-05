# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.
    
    This implementation combines proven bin packing heuristics:
    1. Tight fit scoring (favor bins that fit the item well)
    2. Utilization balancing (avoid both empty and nearly full bins)  
    3. Waste avoidance (penalize very small gaps)
    4. Future packing potential
    
    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    if len(bins) == 0:
        return np.array([])
        
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Factor 1: Tight fit scoring (inverse of remaining space)
    # Prefer bins that fit the item tightly (small remaining space)
    # Using a small epsilon to prevent division by zero
    tight_fit = 1.0 / (remaining + 1e-8)
    
    # Factor 2: Utilization balancing (quadratic function peaking at 50% utilization)
    # Encourage bins that are neither too empty nor too full
    bin_capacity = 100.0
    utilization = (bin_capacity - remaining) / bin_capacity
    utilization_balance = 4.0 * utilization * (1.0 - utilization)
    
    # Factor 3: Waste penalty for small gaps
    # Penalize bins with very small remaining capacity (less than 5 units)
    waste_penalty = np.zeros_like(remaining)
    small_gap_mask = remaining < 5.0
    if np.any(small_gap_mask):
        waste_penalty[small_gap_mask] = -0.3 * (5.0 - remaining[small_gap_mask]) / 5.0
    
    # Factor 4: Future packing potential
    # Prefer bins that leave space for items similar in size to the current item
    future_potential = np.exp(-0.1 * np.abs(remaining - item))
    
    # Combine all factors with appropriate weights
    scores = (0.4 * tight_fit + 
              0.3 * utilization_balance + 
              0.2 * waste_penalty + 
              0.1 * future_potential)
    
    # Ensure all scores are positive and numerically stable
    scores = np.maximum(scores, 1e-10)
    
    return scores

# EVOLVE-BLOCK-END

