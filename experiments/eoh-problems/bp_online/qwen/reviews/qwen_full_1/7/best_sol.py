# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score bins using a refined approach optimized for online bin packing.
    
    This approach follows the proven pattern from top performers:
    1. Tight fit scoring (minimal waste)
    2. Fragmentation penalty (avoiding small gaps)
    3. Balanced utilization (moderate fill levels)
    4. Simple, efficient computation
    
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
    
    # Core tight fit scoring - prefer bins with minimal remaining space
    tight_fit = 1.0 / (remaining + 1.0)
    
    # Fragmentation penalty - heavily penalize very small gaps
    # Very small gaps (less than 10 units) are hard to utilize
    min_gap = max(10, item // 2)
    # Direct vectorized computation without conditionals
    fragmentation_penalty = np.where(remaining < min_gap, 
                                   0.5 * (remaining / min_gap), 
                                   1.0)
    
    # Balance score - prefer moderate utilization (around 40% remaining)
    ideal_remaining = 40.0
    balance = 1.0 - np.minimum(np.abs(remaining - ideal_remaining) / ideal_remaining, 1.0)
    
    # Combine with optimized weights (based on proven performance)
    scores = 0.55 * tight_fit + 0.25 * fragmentation_penalty + 0.15 * balance
    
    # Ensure all scores are non-negative
    scores = np.maximum(scores, 1e-6)
    
    return scores

# EVOLVE-BLOCK-END

