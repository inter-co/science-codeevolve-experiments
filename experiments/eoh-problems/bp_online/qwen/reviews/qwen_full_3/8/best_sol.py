# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.
    
    Inspired by successful heuristics that balance tight fitting with load balancing
    and fragmentation avoidance to minimize the number of used bins.
    
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
    
    # Component 1: Tight fit scoring (inverse of remaining space)
    # Prefer bins that fit the item most tightly
    tight_fit_score = 1.0 / (remaining + 1e-8)
    
    # Component 2: Load balancing (encourage moderate fill ratios)
    # Target around 50% fill ratio for good packing efficiency
    fill_ratio = (100.0 - remaining) / 100.0
    ideal_fill_ratio = 0.5
    load_balance_score = 1.0 - np.abs(fill_ratio - ideal_fill_ratio)
    
    # Component 3: Fragmentation avoidance
    # Penalize bins that would leave very small remaining capacities
    fragmentation_penalty = np.ones_like(remaining)
    # Very small gaps (< 5 units) are heavily penalized to prevent fragmentation
    fragmentation_penalty[remaining < 5] = 0.5
    # Small gaps (5-10 units) are moderately penalized
    fragmentation_penalty[(remaining >= 5) & (remaining < 10)] = 0.8
    
    # Component 4: Balance scoring (simple version)
    # Encourage diversity in remaining capacities to avoid creating too many bins
    if len(bins) > 1:
        # Simple variance-based balance score (smaller variance = more balanced)
        balance_score = 1.0 / (np.var(remaining) + 1.0)
    else:
        balance_score = 1.0
    
    # Combine components with carefully chosen weights
    # Weights sum to 1.0 for normalization
    scores = (
        0.4 * tight_fit_score +
        0.3 * load_balance_score * fragmentation_penalty +
        0.3 * balance_score
    )
    
    # Ensure all scores are positive and reasonable
    scores = np.maximum(scores, 1e-6)
    
    return scores

# EVOLVE-BLOCK-END

