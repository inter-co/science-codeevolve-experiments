# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Enhanced scoring function for online bin packing.
    
    This version incorporates the improved weightings and fragmentation handling
    from the top-performing inspiration while maintaining computational efficiency.
    
    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    if len(bins) == 0:
        return np.array([])
    
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Core scoring: tight fit with inverse relationship to remaining space
    tight_fit = 1.0 / (remaining + 1.0)
    
    # Fragmentation penalty: avoid very small remaining spaces
    # Very small gaps are hard to utilize for future items
    min_gap = max(10, item // 2)
    fragmentation_penalty = np.ones_like(remaining)
    small_gap_mask = remaining < min_gap
    if np.any(small_gap_mask):
        # Use INSPIRATION 1's more effective penalty approach
        fragmentation_penalty[small_gap_mask] = 0.5 * (remaining[small_gap_mask] / min_gap)
    
    # Balance score: prefer bins that don't leave too little or too much space
    # Aim for moderate utilization (around 40% remaining)
    ideal_remaining = 40.0
    # Clip to avoid extreme values and keep it bounded
    balance = 1.0 - np.minimum(np.abs(remaining - ideal_remaining) / ideal_remaining, 1.0)
    
    # Future compatibility: leave some space for similar items
    future_compat = np.zeros_like(remaining)
    compat_mask = remaining >= item * 0.7
    if np.any(compat_mask):
        # Reward bins that leave reasonable space for similar items
        future_compat[compat_mask] = remaining[compat_mask] / 100.0
    
    # Combine with optimized weights for best performance
    # Using the proven weightings from INSPIRATION 1
    scores = 0.55 * tight_fit + 0.25 * fragmentation_penalty + 0.15 * balance + 0.05 * future_compat
    
    # Ensure all scores are non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END

