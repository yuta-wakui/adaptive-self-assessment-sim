def summarize_metrics(
    logs_df: pd.DataFrame,
    total_questions: int,
) -> Dict[str, Any]:
    """
    Summarize simulation metrics from logs DataFrame.
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
    
    validate_columns(logs_df, LOG_REQUIRED_COLS, "logs_df")

    y_true = logs_df["actual_ra"].astype(int)
    y_pred = logs_df["predicted_ra"].astype(int)

    accuracy_all = float(accuracy_score(y_true, y_pred))
    f1_macro_all = float(f1_score(y_true, y_pred, average="macro"))

    confident_df = logs_df[logs_df["is_confident"].astype(bool)]
    if not confident_df.empty:
        y_true_c = confident_df["actual_ra"].astype(int)
        y_pred_c = confident_df["predicted_ra"].astype(int)
        accuracy_over = float(accuracy_score(y_true_c, y_pred_c))
        f1_over = float(f1_score(y_true_c, y_pred_c, average="macro"))
        coverage = float(len(confident_df) / len(logs_df))
    else:
        accuracy_over = f1_over = coverage = None

    avg_answered = float(logs_df["num_answered_questions"].mean())
    avg_complemented = float(logs_df["num_complemented_questions"].mean())

    if total_questions <= 0:
        reduction_rate = None
    else:
        reduction_rate = float((1.0 - avg_answered / total_questions))

    valid_comp = logs_df[logs_df["num_complemented_questions"] > 0]
    avg_comp_acc = float(valid_comp["complement_accuracy"].mean()) if not valid_comp.empty else None

    avg_rt = max_rt = min_rt = None

    rt = logs_df["response_time"].dropna()
    if not rt.empty:
        avg_rt = float(rt.mean())
        max_rt = float(rt.max())
        min_rt = float(rt.min())

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