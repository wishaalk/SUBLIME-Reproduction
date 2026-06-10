"""Data augmentation for contrastive views.

See Figure 2 in the paper on what this does exactly and why this is important for the graph learning.

Feature masking: randomly zero out a fraction of feature dimensions (columns),
applied uniformly to all nodes. Used independently for both views with
potentially different rates (maskfeat_rate_anchor vs maskfeat_rate_learner).

Edge dropping is not handled here, it's done inside the GCN encoder via
dropout_adj (the encoder drops edges from whatever graph it receives).
"""
from __future__ import annotations

import numpy as np
import torch


def mask_features(features: torch.Tensor, mask_rate: float) -> torch.Tensor:
    if mask_rate == 0:
        return features.clone()

    n_features = features.shape[1]
    n_mask = int(n_features * mask_rate)
    cols = np.random.choice(n_features, size=n_mask, replace=False)

    mask = torch.ones(n_features, device=features.device)
    mask[cols] = 0.0
    return features * mask
