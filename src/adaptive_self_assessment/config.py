import yaml
from typing import Any, Dict

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    YAML形式の設定ファイルを読み込む関数
    
    Parameters:
    ----------
    config_path: str
        設定ファイルのパス（デフォルト； "configs/config.yaml"）
    
    Returns:
    -------
    cfg: Dict[str, Any] 
        読み込んだ設定内容を辞書形式で返す
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg