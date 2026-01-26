from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# dataclass for thresholds
@dataclass(frozen=True)
class Thresholds:
    rc: float
    ri: float

def load_thresholds(cfg: Dict[str, Any]) -> Thresholds:
    """
    Load thresholds set from configuration dictionary
    Parameters:
    -----------
        cfg: Dict[str, Any]
            Configuration dictionary
    Returns:
    -------
        Thresholds
            Thresholds dataclass instance
    """
    t = cfg.get("thresholds", {})
    return Thresholds(
        rc=float(t.get("RC", 0.80)),
        ri=float(t.get("RI", 0.70)),
    )

def validate_columns(df: pd.DataFrame, required: List[str], df_name: str) -> None:
    """
    Validate that required columns are present in the DataFrame
    Parameters:
    -----------
        df: pd.DataFrame
            DataFrame to validate
        required: List[str]
            List of required column names
        df_name: str
            Name of the DataFrame (for error messages)
    Returns:
    -------
        None
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")

def complement_accuracy(complemented_items: List[tuple]) -> Tuple[Optional[float], List[str]]:
    """
    Calculate the accuracy of complemented items
    Parameters:
    -----------
        complemented_items: List[tuple]
            List of tuples (item, predicted, confidence, actual)
    Returns:
    -------
        accuracy: Optional[float]
            Accuracy of complemented items, or None if no items
        correct_items: List[str]
            List of correctly complemented item names
    """
    if not complemented_items:
        return None, []
    correct = [item for (item, pred, _, actual) in complemented_items if pred == actual]
    return len(correct) / len(complemented_items), correct

def summarize_metrics(
    logs_df: pd.DataFrame,
    total_questions: int,
) -> Dict[str, Any]:
    """
    Summarize simulation metrics from logs DataFrame
    Parameters:
    -----------
        logs_df: pd.DataFrame
            Logs DataFrame containing simulation results
        total_questions: int
            Total number of questions in the assessment
    Returns:
    -------
        metrics: Dict[str, Any]
            Dictionary of summarized metrics
    """

    # empty guard
    if logs_df is None or logs_df.empty:
        return {
            "accuracy_all": None,
            "f1_macro_all": None,
            "accuracy_over_threshold": None,
            "f1_macro_over_threshold": None,
            "coverage_over_threshold": None,
            "total_questions": int(total_questions),
            "avg_answered_questions": None,
            "avg_complemented_questions": None,
            "avg_complement_accuracy": None,
            "reduction_rate": None,
            "avg_response_time": None,
            "max_response_time": None,
            "min_response_time": None,
        }
    
    required_cols = [
        "actual_ra", "predicted_ra", "is_confident",
        "num_answered_questions", "num_complemented_questions",
        "complement_accuracy", "response_time",
    ]
    validate_columns(logs_df, required_cols, "logs_df")

    y_true = logs_df["actual_ra"].astype(int)
    y_pred = logs_df["predicted_ra"].astype(int)

    accuracy_all = float(accuracy_score(y_true, y_pred) * 100.0)
    f1_macro_all = float(f1_score(y_true, y_pred, average="macro") * 100.0)

    confident_df = logs_df[logs_df["is_confident"] == True]
    if not confident_df.empty:
        y_true_c = confident_df["actual_ra"].astype(int)
        y_pred_c = confident_df["predicted_ra"].astype(int)
        accuracy_over = float(accuracy_score(y_true_c, y_pred_c) * 100)
        f1_over = float(f1_score(y_true_c, y_pred_c, average="macro") * 100)
        coverage = float(len(confident_df) / len(logs_df) * 100)
    else:
        accuracy_over = f1_over = coverage = None

    avg_answered = float(logs_df["num_answered_questions"].mean())
    avg_complemented = float(logs_df["num_complemented_questions"].mean())

    if total_questions <= 0:
        reduction_rate = None
    else:
        reduction_rate = float((1.0 - avg_answered / total_questions) * 100.0)

    valid_comp = logs_df[logs_df["num_complemented_questions"] > 0]
    avg_comp_acc = float(valid_comp["complement_accuracy"].mean() * 100.0) if not valid_comp.empty else None

    avg_rt = max_rt = min_rt = None
    if "response_time" in logs_df.columns and not logs_df["response_time"].empty:
        avg_rt = float(logs_df["response_time"].mean())
        max_rt = float(logs_df["response_time"].max())
        min_rt = float(logs_df["response_time"].min())

    return {
        "accuracy_all": accuracy_all,
        "f1_macro_all": f1_macro_all,
        "accuracy_over_threshold": accuracy_over,
        "f1_macro_over_threshold": f1_over,
        "coverage_over_threshold": coverage,
        "total_questions": int(total_questions),
        "avg_answered_questions": avg_answered,
        "avg_complemented_questions": avg_complemented,
        "avg_complement_accuracy": avg_comp_acc,
        "reduction_rate": reduction_rate,
        "avg_response_time": avg_rt,
        "max_response_time": max_rt,
        "min_response_time": min_rt,
    }
