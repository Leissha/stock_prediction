from __future__ import annotations


def print_header(title: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(title, flush=True)
    print(f"{'='*60}", flush=True)


_last_summary_fingerprint: str | None = None


def print_summary_table(results: dict) -> None:
    """Print a compact metrics table for classification results.
    results: { model_name: { 'accuracy','precision','recall','f1_score' } }
    """
    global _last_summary_fingerprint
    # Build table once into a string, then print atomically; also dedupe identical outputs
    header = f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
    sep = "-" * 60 + "\n"
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            f"{model_name:<20} "
            f"{metrics['accuracy']:<10.4f} "
            f"{metrics['precision']:<10.4f} "
            f"{metrics['recall']:<10.4f} "
            f"{metrics['f1_score']:<10.4f}"
        )
    body = "\n".join(rows) + "\n"
    full_table = header + sep + body
    if full_table != _last_summary_fingerprint:
        print(full_table, flush=True)
        _last_summary_fingerprint = full_table


def print_sentiment_impact_table(baseline_model: str, results_sentiment: dict, baseline_results: dict) -> None:
    """Print sentiment vs baseline comparison table."""
    print(f"{baseline_model}                     Accuracy   Precision  Recall     F1-Score")
    print("-" * 60)
    print(
        f"{baseline_model} + Sentiment          {results_sentiment['accuracy']:<10.4f} "
        f"{results_sentiment['precision']:<10.4f} {results_sentiment['recall']:<10.4f} "
        f"{results_sentiment['f1_score']:<10.4f}"
    )
    print(
        f"{baseline_model} Baseline             {baseline_results['accuracy']:<10.4f} "
        f"{baseline_results['precision']:<10.4f} {baseline_results['recall']:<10.4f} "
        f"{baseline_results['f1_score']:<10.4f}"
    )


def print_sentiment_impact_line(results_lstm: dict, baseline: dict) -> None:
    """Print a short sentiment impact comparison line."""
    accuracy_improvement = results_lstm['accuracy'] - baseline['accuracy']
    f1_improvement = results_lstm['f1_score'] - baseline['f1_score']
    print("\nSentiment Impact:")
    print(f"  Accuracy improvement: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.1f}%)")
    print(f"  F1-Score improvement:  {f1_improvement:+.4f} ({f1_improvement*100:+.1f}%)")
    if abs(accuracy_improvement) < 0.001 and abs(f1_improvement) < 0.001:
        print("Sentiment features have NO IMPACT")
    else:
        print("Sentiment features have IMPACT")


def print_confusion_matrix(cm, y_true_binary) -> None:
    """Pretty-print a 2x2 confusion matrix with safe fallbacks."""
    print(f"\n  Confusion Matrix:")
    if getattr(cm, 'shape', None) == (2, 2):
        print(f"    True Neg: {cm[0,0]:4d} | False Pos: {cm[0,1]:4d}")
        print(f"    False Neg: {cm[1,0]:4d} | True Pos:  {cm[1,1]:4d}")
    elif getattr(cm, 'shape', None) == (1, 1):
        if y_true_binary[0] == 1:
            print(f"    All Positive: {cm[0,0]:4d} samples")
        else:
            print(f"    All Negative: {cm[0,0]:4d} samples")
    else:
        print(f"    Confusion Matrix Shape: {getattr(cm, 'shape', None)}")
        print(f"    Matrix: {cm}")
