from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Hashable
from sklearn.pipeline import Pipeline

CacheKey = Tuple[Hashable, ...]

@dataclass
class ModelStore:
    models: Dict[CacheKey, Pipeline] = field(default_factory=dict)

    def get(self, key: CacheKey) -> Pipeline | None:
        return self.models.get(key)

    def set(self, key: CacheKey, model: Pipeline) -> None:
        self.models[key] = model