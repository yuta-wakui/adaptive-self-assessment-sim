import hashlib

def make_selector_seed(cv_seed: int, fold: int, user_id: object) -> int:
    """
    generate a reproducible random seed for question selection based on cv_seed, fold, and user_id
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