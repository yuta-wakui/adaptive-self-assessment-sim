# 1回分の自己評価データ
SPEC_WS1 = dict(
    PCA_COL = None,
    PCA_PREFIX = None,
    CA_PREFIX = "item-",
    TARGET_COL = "assessment-result",
    IGNORE_COLS = {"reflection-length"},
)

# 2回分の自己評価データ
SPEC_WS2 = dict(
    PCA_COL = "w3-assessment-result",
    PCA_PREFIX = "w3-",
    CA_PREFIX = "w4-",
    TARGET_COL = "w4-assessment-result",
    IGNORE_COLS = {"w3-reflection-length", "w4-reflection-length"},
)