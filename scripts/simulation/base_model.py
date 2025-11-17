import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run_base_model(df, model_type="logistic", logs_path=None, cv_splits=5):
    pca_cols = [col for col in df.columns if col.startswith("w3-") and col != "w3-assessment-result"]
    ca_cols = [col for col in df.columns if col.startswith("w4-") and col not in ["w4-assessment-result", "w4-reflection-length"]]
    pra = ["w3-assessment-result"]
    feature_cols = pra + pca_cols + ca_cols

    X = df[feature_cols]
    y = df["w4-assessment-result"]

    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
            # class_weight={2:1.5, 3: 2, 4: 1.5}
        )
    elif model_type == "gd":
        model = GradientBoostingClassifier(random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Invalid model_type")

    # クロスバリデーションで精度確認（5分割）
    scores = cross_val_score(model, X, y, cv=cv_splits, scoring="accuracy")

    # 結果をDataFrame化
    base_results = pd.DataFrame([{
        "num_questions": len(ca_cols),
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "avg_confidence": None,
        "avg_completions": None,
        "avg_completion_accuracy": None,
        "avg_total_response_time": None,
        "avg_avg_response_time": None,
        "max_response_time": None,
        "min_response_time": None,
    }])

    # ログファイルに保存
    if logs_path:
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        base_results.to_csv(logs_path, index=False)

    return scores.mean(), scores.std()