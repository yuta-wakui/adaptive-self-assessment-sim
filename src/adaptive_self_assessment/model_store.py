from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Hashable
from sklearn.pipeline import Pipeline

CacheKey = Tuple[Hashable, ...]

@dataclass
class ModelStore:
    models: Dict[CacheKey, Pipeline] = field(default_factory=dict)
    verbose: bool = True # ログON/OFF

    def get(self, key: CacheKey) -> Pipeline | None:
        model = self.models.get(key)
        if self.verbose:
            if model is None:
                print(f"[CACHE][MISS] key={key}")
            else:
                print(f"[CACHE][HIT ] key={key}")
        return model

    def set(self, key: CacheKey, model: Pipeline) -> None:
        self.models[key] = model
        if self.verbose:
            print(f"[CACHE][STORE] key={key} (size={len(self.models)})")
