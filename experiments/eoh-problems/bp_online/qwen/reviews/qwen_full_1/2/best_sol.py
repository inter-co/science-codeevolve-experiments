# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.

    This implementation uses a hybrid scoring function that balances tight fitting
    with load balancing to avoid creating unusable small gaps.

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    bin_capacity = 100
    remaining = bins - item
    
    # Primary: tight fit scoring (inverse of remaining space)
    # Higher scores for bins that fit the item more tightly
    tight_fit = 1.0 / (remaining + 0.1)
    
    # Secondary: load balancing to avoid extreme packing states
    # Prefer bins that are neither too empty nor too full
    load_ratio = (bin_capacity - remaining) / bin_capacity
    balance = np.exp(-((load_ratio - 0.5)**2) / 0.05)
    
    # Combine tight fit and balance with fixed weights
    core_score = 0.6 * tight_fit + 0.4 * balance
    
    # Fragmentation penalty: discourage leaving very small gaps
    # Small gaps make future items harder to place
    small_gap_penalty = np.where(remaining < 5, -0.15, 0.0)
    
    # Final score
    scores = core_score + small_gap_penalty
    
    return scores

# EVOLVE-BLOCK-END

