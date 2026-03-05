# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.

    Optimized version matching successful heuristics approach exactly.

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Component 1: Tight fit scoring (inverse of remaining space)
    tight_fit_score = 1.0 / (remaining + 1e-8)
    
    # Component 2: Load balancing (encourage moderate fill ratios)
    fill_ratio = (100.0 - remaining) / 100.0
    load_balance_score = 1.0 - np.abs(fill_ratio - 0.5)
    
    # Component 3: Fragmentation avoidance
    fragmentation_penalty = np.ones_like(remaining)
    # Heavy penalty for very small gaps (< 5 units)
    fragmentation_penalty[remaining < 5] = 0.5
    # Moderate penalty for small gaps (5-10 units)
    fragmentation_penalty[(remaining >= 5) & (remaining < 10)] = 0.8
    
    # Component 4: Balance scoring (simple variance-based)
    if len(bins) > 1:
        balance_score = 1.0 / (np.var(remaining) + 1.0)
    else:
        balance_score = 1.0
    
    # Combine components with weights matching successful heuristics
    scores = (
        0.4 * tight_fit_score +
        0.3 * load_balance_score * fragmentation_penalty +
        0.3 * balance_score
    )
    
    # Ensure all scores are positive and reasonable
    scores = np.maximum(scores, 1e-6)
    
    return scores

# EVOLVE-BLOCK-END

