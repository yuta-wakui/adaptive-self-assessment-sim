import hashlib

def make_selector_seed(cv_seed: int, fold: int, user_id: str) -> int:
    """
    交差検証とfold番号、ユーザーIDに基づいて乱数シードを生成する関数
    Parameters:
    ----------
    cv_seed: int
        交差検証のシード値
    fold: int
        現在のfold番号
    user_id: str
        ユーザーID
    
    Returns:
    -------
    int
        生成された乱数シード
    """
    seed_str = f"{cv_seed}_{fold}_{user_id}".encode("utf-8")
    seed_hash = hashlib.sha256(seed_str).hexdigest()
    return int(seed_hash[:8], 16)  # 32ビット整数に変換