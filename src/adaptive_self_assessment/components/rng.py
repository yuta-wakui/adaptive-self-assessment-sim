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
    generate a reproducible random seed for question selection based on cv_seed, fold, and user_id.
    Parameters:
    ----------
    cv_seed: int
        cross-validation random seed
    fold: int
        fold number
    user_id: str
        user identifier
    
    Returns:
    -------
    int
        generated random seed
    """
    seed_str = f"selector|{cv_seed}_{fold}_{user_id}".encode("utf-8")
    seed_hash = hashlib.sha256(seed_str).hexdigest()
    return int(seed_hash[:8], 16)  # convert to 32-bit integer