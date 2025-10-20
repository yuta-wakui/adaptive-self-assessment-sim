import numpy as np
from typing import List

_rng = np.random.default_rng(42) # デフォルトseedを設定

def set_selector_seed(seed: int) -> None:
    """
    セレクタの乱数シードを設定する関数

    Parameters:
    ----------
    seed : int
        乱数シードの値
    """
    global _rng
    _rng = np.random.default_rng(seed)

def select_question(C: List[str]) -> str:
    """
    質問項目をランダムに選択する関数
    
    Parameters:
    ----------
    C: list of str
        残りの質問項目のリスト
    Returns:
        str: 選択された質問項目
    """
    if not C:
        raise ValueError("質問項目のリストが空です。")
    
    idx = _rng.integers(0, len(C))
    return C[idx]