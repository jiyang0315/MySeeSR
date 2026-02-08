"""
Consistency Losses for SeeSR
"""

from .consistency_losses import (
    EdgeConsistencyLoss,
    FrequencyConsistencyLoss,
    PerceptualConsistencyLoss,
    ConsistencyLossManager,
)

__all__ = [
    "EdgeConsistencyLoss",
    "FrequencyConsistencyLoss", 
    "PerceptualConsistencyLoss",
    "ConsistencyLossManager",
]

