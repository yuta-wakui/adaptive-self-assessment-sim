# -*- coding: utf-8 -*-

"""
This module provides functions to generate reproducible random seeds for question selection
in adaptive self-assessment simulations.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/components/rng.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Random seed generation for question selection

import hashlib
from typing import Hashable

def make_selector_seed(cv_seed: int, fold: int, user_id: Hashable) -> int:
    """
    generate a reproducible random seed for question selection.

    The seed is deterministically derived from (cv_seed, fold, user_id) via SHA-256.
    Note: reproducibility assumes `user_id` has a stable string representation across runs.

    Parameters:
    ----------
    cv_seed: int
        cross-validation random seed
    fold: int
        fold number
    user_id: Hashable
        user identifier (e.g., str or int)
    
    Returns:
    -------
    int
        generated seed in the range[0, 2^32-1]
    """
    seed_bytes = f"selector|{cv_seed}_{fold}_{user_id}".encode("utf-8")
    digest = hashlib.sha256(seed_bytes).digest()
    return int.from_bytes(digest[:4], "big", signed=False)