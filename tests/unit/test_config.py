import pytest
from adaptive_self_assessment.config import load_config

def test_load_config():
    cfg = load_config()
    
    print(cfg)

    # 全体がdict型であることを確認
    assert isinstance(cfg, dict)

    # 主要なキーが存在することを確認
    assert "mode" in cfg
    assert "model" in cfg
    assert "thresholds" in cfg
    assert "data" in cfg
    assert "logging" in cfg
    assert "results" in cfg

    # mode は ws1 / ws2 のどちらか
    assert cfg["mode"] in ("ws1", "ws2")

