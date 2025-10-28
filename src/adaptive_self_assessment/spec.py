from typing import Tuple, List

# 1回分の自己評価データ
SPEC_WS1 = dict(
    PCA_COL = None,
    PCA_PREFIX = None,
    CA_PREFIX = "item-",
    RA_COL = "assessment-result",
    IGNORE_COLS = ["reflection-length"],
)

# 2回分の自己評価データ
SPEC_WS2 = dict(
    PCA_COL = "w3-assessment-result",
    PCA_PREFIX = "w3-",
    CA_PREFIX = "w4-",
    RA_COL = "w4-assessment-result",
    IGNORE_COLS = ["w3-reflection-length", "w4-reflection-length"],
)

def get_spec_cols(df, spec: dict) -> Tuple[str | None , List[str] | None, List[str], str, List[str]]:
    """
    spec辞書から列名情報を抽出して返す関数

    Parameters:
    ----------
        df: pd.DataFrame
            対象データフレーム
        spec: dict
            列名仕様辞書
    Returns:
    -------
        pra_col: str | None
            過去の総合評価
        pca_cols: List[str] | None
            過去のチェックリスト
        ca_cols: List[str]
            現在のチェック項目
        ra_col: str
            現在の総合評価
        ignore_cols: Set[str]
            無視する列名の集合
    """
    pra_col = spec.get("PCA_COL")
    pca_prefix = spec.get("PCA_PREFIX")
    ca_prefix = spec["CA_PREFIX"]
    ra_col = spec["RA_COL"]
    ignore_cols = spec["IGNORE_COLS"]

    # 過去のチェック項目
    if pca_prefix is not None:
        pca_cols = [c for c in df.columns if c.startswith(pca_prefix) and c not in [pra_col] + ignore_cols]
    else:
        pca_cols = None

    # 現在のチェック項目
    ca_cols = [c for c in df.columns if c.startswith(ca_prefix) and c not in [ra_col] + ignore_cols]

    return pra_col, pca_cols, ca_cols, ra_col, ignore_cols
