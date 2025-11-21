import numpy as np
from typing import List

_rng = np.random.default_rng(42) # デフォルトseedを設定

def set_selector_seed(seed: int) -> None:
    """
    セレクタの乱数生成器のシードを設定する関数
    
    Parameters:
    ----------
    seed: int
        乱数生成器のシード値
    """
    global _rng
    _rng = np.random.default_rng(seed)

def select_question(C: List[str], strategy: str = "random") -> str:
    """
    質問項目をランダムに選択する関数
    
    Parameters:
    ----------
    C: list of str
        残りの質問項目のリスト
    strategy: str
        選択戦略（現在は"random"のみ対応）
    Returns:
        str: 選択された質問項目
    """
    if not C:
        raise ValueError("質問項目のリストが空です。")

    if strategy == "random":
        return str(_rng.choice(C))
    
    raise ValueError(f"Unknown selection strategy: {strategy}")