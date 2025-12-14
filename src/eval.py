"""Evaluation utilities for COVID model training."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)

# ---------------------------------------------------------------------------
# Threshold helpers


def find_optimal_threshold_f1(y_true, y_proba):
    """Find optimal threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 1.0, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def find_optimal_threshold_youden(y_true, y_proba):
    """Find optimal threshold using Youden's J statistic (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx], youden_j[optimal_idx]


# ---------------------------------------------------------------------------
# Metric computation helpers


STANDARD_METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "roc_auc": "AUC (AUROC)",
    "aupr": "AUPR (Average Precision)",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Sensitivity (Recall)",
    "specificity": "Specificity",
}

OPTIMIZED_METRICS = {
    "f1": "F1 Score (Optimized)",
    "precision": "Precision (Optimized)",
    "recall": "Recall (Optimized)",
    "balanced_accuracy": "Balanced Accuracy (Optimized)",
}


def _format_mean_ci(mean: float, std: float, n: int) -> Dict[str, float | str]:
    if n <= 0 or np.isnan(std):
        return {
            "mean": mean,
            "std": std,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "formatted": f"{mean:.3f}"
        }
    se = std / np.sqrt(n)
    z = 1.96
    lower = mean - z * se
    upper = mean + z * se
    return {
        "mean": mean,
        "std": std,
        "ci_low": lower,
        "ci_high": upper,
        "formatted": f"{mean:.3f} (95% CI: {lower:.3f} - {upper:.3f})"
    }


def _build_stats(values: Sequence[float], n: int) -> Dict[str, float | str]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return _format_mean_ci(np.nan, np.nan, n)
    return _format_mean_ci(float(np.mean(arr)), float(np.std(arr)), n)


def evaluate_split(y_true, y_pred, y_proba):
    """Compute all evaluation metrics for a single split."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    aupr = average_precision_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
    else:
        specificity = 0.0

    threshold, _ = find_optimal_threshold_f1(y_true, y_proba)
    y_pred_opt = (y_proba >= threshold).astype(int)

    optimized = {
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred_opt),
        "precision": precision_score(y_true, y_pred_opt, zero_division=0),
        "recall": recall_score(y_true, y_pred_opt, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_opt),
    }

    return {
        "roc_curve": (fpr, tpr),
        "roc_auc": roc_auc,
        "aupr": aupr,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "optimized": optimized,
    }


def summarize_metrics(split_metrics: List[Mapping[str, float]], n_splits: int):
    """Summarize metric distributions across all splits for a model."""
    history = defaultdict(list)
    optimized_history = defaultdict(list)
    roc_curves = []
    aucs = []

    for metrics in split_metrics:
        for key in STANDARD_METRICS:
            history[key].append(metrics.get(key, np.nan))
        roc_curves.append(metrics["roc_curve"])
        aucs.append(metrics["roc_auc"])
        optimized = metrics.get("optimized", {})
        for key in OPTIMIZED_METRICS:
            optimized_history[key].append(optimized.get(key, np.nan))

    summary = {name: _build_stats(values, n_splits) for name, values in history.items()}
    optimized_summary = {name: _build_stats(values, n_splits) for name, values in optimized_history.items()}

    return {
        "standard": summary,
        "optimized": optimized_summary,
        "roc_curves": roc_curves,
        "aucs": aucs,
    }


def print_metric_summary(model_name: str, summary: Mapping[str, Mapping[str, float | str]]):
    print(f"\n** {model_name} **")
    for key, label in STANDARD_METRICS.items():
        stats = summary.get(key)
        if not stats:
            continue
        print(f"{label}: {stats['formatted']}")


def print_optimized_summary(model_name: str, optimized_summary: Mapping[str, Mapping[str, float | str]]):
    print(f"\n** {model_name} (F1-Optimized Threshold) **")
    for key, label in OPTIMIZED_METRICS.items():
        stats = optimized_summary.get(key)
        if not stats:
            continue
        print(f"{label}: {stats['formatted']}")


# ---------------------------------------------------------------------------
# Plotting helpers


def _mean_roc_from_split_metrics(split_metrics: Iterable[Mapping[str, object]], mean_fpr: np.ndarray):
    tprs = []
    aucs = []
    for metrics in split_metrics:
        fpr, tpr = metrics["roc_curve"]
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(metrics["roc_auc"])
    if not tprs:
        return None
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    return mean_tpr, std_tpr, mean_auc, std_auc


def plot_mean_roc_panel(split_metrics_by_model: Mapping[str, List[Mapping[str, object]]], *, data_mode: str, round_label: str, n_splits: int, n_points: int = 100, output_path: str):
    mean_fpr = np.linspace(0, 1, n_points)
    plt.figure(figsize=(10, 8))
    for model_name, split_metrics in split_metrics_by_model.items():
        roc_data = _mean_roc_from_split_metrics(split_metrics, mean_fpr)
        if roc_data is None:
            continue
        mean_tpr, std_tpr, mean_auc, std_auc = roc_data
        plt.plot(mean_fpr, mean_tpr, lw=2, label=f"{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves ({round_label}, {n_splits} Random Splits)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_comparison_by_mode(*, model_name: str, split_metrics_per_mode: Mapping[str, List[Mapping[str, object]]], labels: Mapping[str, str], colors: Mapping[str, str], n_points: int, output_path: str):
    mean_fpr = np.linspace(0, 1, n_points)
    plt.figure(figsize=(10, 8))
    for mode, split_metrics in split_metrics_per_mode.items():
        if not split_metrics:
            continue
        roc_data = _mean_roc_from_split_metrics(split_metrics, mean_fpr)
        if roc_data is None:
            continue
        mean_tpr, std_tpr, mean_auc, std_auc = roc_data
        color = colors.get(mode, None)
        label = labels.get(mode, mode.title())
        plt.plot(mean_fpr, mean_tpr, lw=2, color=color, label=f"{label} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.15)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve Comparison for {model_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_auc_differences(reference: Sequence[float], comparison: Sequence[float]):
    paired_len = min(len(reference), len(comparison))
    if paired_len == 0:
        return np.array([])
    ref = np.array(reference[:paired_len])
    comp = np.array(comparison[:paired_len])
    return ref - comp


def plot_auc_differences(differences: np.ndarray, title: str, ylabel: str, color: str, mean_line_color: str, output_path: str):
    if differences.size == 0:
        return
    mean_diff = float(np.mean(differences))
    plt.figure(figsize=(10, 6))
    splits = np.arange(1, differences.size + 1)
    plt.bar(splits, differences, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('Split Number')
    plt.ylabel(ylabel)
    plt.axhline(mean_diff, color=mean_line_color, linestyle='dashed', linewidth=2, label=f'Mean Diff: {mean_diff:.4f}')
    plt.xticks(splits)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
