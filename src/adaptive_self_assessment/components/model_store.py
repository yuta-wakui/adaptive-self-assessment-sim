# -*- coding: utf-8 -*-

"""
Model store for caching trained scikit-learn models.
This module defines the ModelStore class, which provides a simple in-memory cache
for storing and retrieving trained models based on unique cache keys.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/components/model_store.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Model store for caching trained models

from dataclasses import dataclass, field
from typing import Dict, Hashable, Tuple, Optional
import logging

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

CacheKey = Tuple[Hashable, ...]

@dataclass
class ModelStore:
    """A simple in-memory cache for trained scikit-learn estimators."""
    models: Dict[CacheKey, BaseEstimator] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all cached models."""
        self.models.clear()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[CACHE][CLEAR] All cached models have been cleared.")

    def size(self) -> int:
        """Return the number of cached models."""
        return len(self.models)

    def get(self, key: CacheKey) -> Optional[BaseEstimator]:
        """Retrieve a cached model by its key."""
        model = self.models.get(key)

        if logger.isEnabledFor(logging.DEBUG):
            if model is None:
                logger.debug(f"[CACHE][MISS] key={key} (size={self.size()})")
            else:
                logger.debug(f"[CACHE][HIT] key={key} (size={self.size()})")

        return model

    def set(self, key: CacheKey, model: BaseEstimator) -> None:
        """Cache a trained model with the given key."""
        self.models[key] = model
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[CACHE][STORE] key={key} (size={self.size()})")