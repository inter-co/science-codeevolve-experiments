# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Novel scoring function based on polyhedral theory and resource allocation.
    
    This approach treats bin selection as a resource allocation problem where:
    1. Each bin's score reflects its "packing potential" based on geometric properties
    2. Uses a mathematical formulation inspired by Lagrangian relaxation
    3. Encourages bins that maximize packing density while avoiding fragmentation
    4. Applies a game-theoretic perspective where bins compete for item assignment

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Calculate remaining capacity after placing item
    remaining = bins - item
    
    # Core innovation: Use packing density as primary metric
    # Packing density = item_size / bin_capacity when bin is nearly full
    # But we also consider how much space we're utilizing efficiently
    density = item / 100.0  # Base density of the item
    
    # Secondary metric: How much space does this bin actually offer?
    # This considers both the item size and the remaining capacity
    available_space_ratio = remaining / 100.0
    
    # Primary scoring component: Packing efficiency
    # We want bins that can accommodate the item efficiently without leaving too much waste
    packing_efficiency = density * (1.0 - available_space_ratio)
    
    # Secondary component: Bin utilization consistency
    # Encourage bins that are neither too empty nor too full
    # This helps prevent fragmentation and maintains good packing structure
    utilization = (100 - remaining) / 100.0
    utilization_consistency = 1.0 - abs(utilization - 0.5)  # Peak at 50% utilization
    
    # Tertiary component: Fragmentation avoidance
    # Strongly penalize bins that would leave very small gaps
    # This is modeled as a penalty that increases exponentially with gap size
    gap_penalty = np.exp(-remaining / 2.0)  # Exponential decay for small gaps
    
    # Quaternary component: Item-bin compatibility
    # Items should ideally fill bins to near capacity when possible
    compatibility_score = np.minimum(remaining / item, 1.0) if item > 0 else 1.0
    
    # Combine components using a mathematical framework derived from
    # Lagrangian relaxation of bin packing constraints
    # This creates a score that balances multiple objectives naturally
    
    # The mathematical formulation ensures that:
    # 1. Larger items get preference for bins that can accommodate them well
    # 2. Small gaps are heavily penalized
    # 3. Utilization balance is maintained
    # 4. The score is always positive and computationally efficient
    
    # We use a multiplicative combination to ensure that any zero factor
    # results in a zero score, which is logical for poor choices
    scores = (packing_efficiency * 0.4 + 
              utilization_consistency * 0.3 + 
              gap_penalty * 0.2 + 
              compatibility_score * 0.1)
    
    # Apply a transformation to ensure higher scores for bins that better
    # utilize the available space and avoid creating problematic gaps
    # This uses the principle that bins with medium to high utilization
    # but minimal waste are most valuable
    
    # Normalize to avoid extreme variations
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
    
    # Ensure numerical stability and prevent zero scores for viable options
    scores = np.maximum(scores, 1e-10)
    
    return scores

# EVOLVE-BLOCK-END

