"""
Shared evaluation metrics for the urban graph benchmark.

This module is deliberately model-agnostic. It provides reusable metrics for:

- count/regression prediction
- binary prediction
- ranking / top-K prioritization
- standardized metric-row formatting

It should be used by A0/A1/A2/A3 baselines and later graph models. It does not
train models, build features, create splits, or implement benchmark-specific
baseline logic.

Notes:
  - Ranking NDCG uses observed counts as graded relevance.
  - Top-K overlap compares predicted top-K and observed top-K sets of equal
    size. The reported overlap rate is overlap / K.
  - Poisson deviance clips predictions to a small positive epsilon. Models
    should still avoid producing negative count predictions when the metric is
    interpreted as a Poisson deviance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None


EPS = 1e-12


class MetricError(RuntimeError):
    """Raised when a metric cannot be computed safely."""


@dataclass(frozen=True)
class MetricResult:
    """Standardized metric result."""

    metric_name: str
    metric_value: float | None
    higher_is_better: bool
    n: int
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "higher_is_better": self.higher_is_better,
            "n_rows": self.n,
            "notes": self.notes,
        }


def require_numpy_pandas() -> None:
    """Fail clearly when numpy/pandas are unavailable."""

    if np is None:
        raise MetricError("numpy is required for evaluation metrics.") from _NUMPY_IMPORT_ERROR
    if pd is None:
        raise MetricError("pandas is required for evaluation metrics.") from _PANDAS_IMPORT_ERROR


def to_numpy_1d(values: Any, name: str = "values") -> np.ndarray:
    """Convert array-like input to a 1D float numpy array."""

    require_numpy_pandas()

    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        arr = np.ravel(arr)

    if arr.ndim != 1:
        raise MetricError(f"{name} must be one-dimensional.")

    return arr


def valid_pair_mask(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned valid numeric arrays and the valid mask."""

    yt = to_numpy_1d(y_true, "y_true")
    yp = to_numpy_1d(y_pred, "y_pred")

    if yt.shape[0] != yp.shape[0]:
        raise MetricError(
            f"Length mismatch: y_true has {yt.shape[0]} rows, y_pred has {yp.shape[0]} rows."
        )

    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask], mask


def safe_float(value: Any) -> float | None:
    """Convert scalar numeric value to float or None."""

    try:
        out = float(value)
    except Exception:
        return None

    if math.isnan(out) or math.isinf(out):
        return None

    return out


def mae(y_true: Any, y_pred: Any) -> float | None:
    """Mean absolute error."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) == 0:
        return None
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Any, y_pred: Any) -> float | None:
    """Root mean squared error."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) == 0:
        return None
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def bias(y_true: Any, y_pred: Any) -> float | None:
    """Mean prediction error: predicted minus observed."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) == 0:
        return None
    return float(np.mean(yp - yt))


def mean_absolute_percentage_error_safe(y_true: Any, y_pred: Any) -> float | None:
    """
    MAPE over strictly positive observed values.

    Rows with y_true <= 0 are excluded because the denominator is undefined.
    """

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    mask = yt > 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def mean_poisson_deviance(y_true: Any, y_pred: Any, eps: float = EPS) -> float | None:
    """
    Mean Poisson deviance for nonnegative counts and positive predictions.

    Formula per row:
        2 * (y_pred - y_true + y_true * log(y_true / y_pred))

    For y_true = 0, the log term is treated as 0.
    Predictions are clipped to a small positive epsilon.
    """

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) == 0:
        return None

    if (yt < 0).any():
        raise MetricError("Poisson deviance requires nonnegative y_true.")

    mu = np.clip(yp, eps, None)
    term = mu - yt
    positive = yt > 0
    term[positive] += yt[positive] * np.log(yt[positive] / mu[positive])
    return float(2.0 * np.mean(term))


def mean_negative_binomial_log_likelihood(
    y_true: Any,
    y_pred_mu: Any,
    alpha: float,
    eps: float = EPS,
) -> float | None:
    """
    Mean NB2 log-likelihood for count observations.

    Parameterization:
        Var(Y) = mu + alpha * mu^2
        r = 1 / alpha
        p = r / (r + mu)

    Returns the mean log-likelihood; higher is better.
    """

    if alpha <= 0:
        raise MetricError("alpha must be > 0 for negative-binomial log-likelihood.")

    yt, mu, _ = valid_pair_mask(y_true, y_pred_mu)
    if len(yt) == 0:
        return None

    if (yt < 0).any():
        raise MetricError("Negative-binomial log-likelihood requires nonnegative y_true.")

    mu = np.clip(mu, eps, None)
    r = 1.0 / alpha
    p = r / (r + mu)
    p = np.clip(p, eps, 1.0 - eps)

    # Vectorized lgamma through Python math. Fine for benchmark-size arrays.
    lgamma_y_plus_r = np.array([math.lgamma(float(y) + r) for y in yt])
    lgamma_y_plus_1 = np.array([math.lgamma(float(y) + 1.0) for y in yt])
    ll = (
        lgamma_y_plus_r
        - math.lgamma(r)
        - lgamma_y_plus_1
        + r * np.log(p)
        + yt * np.log(1.0 - p)
    )
    return float(np.mean(ll))


def pearson_corr(y_true: Any, y_pred: Any) -> float | None:
    """Pearson correlation."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) < 2 or np.std(yt) == 0 or np.std(yp) == 0:
        return None
    return float(np.corrcoef(yt, yp)[0, 1])


def spearman_corr(y_true: Any, y_pred: Any) -> float | None:
    """Spearman rank correlation."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) < 2:
        return None

    s1 = pd.Series(yt)
    s2 = pd.Series(yp)
    value = s1.corr(s2, method="spearman")
    return safe_float(value)


def kendall_corr(y_true: Any, y_pred: Any) -> float | None:
    """Kendall rank correlation."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    if len(yt) < 2:
        return None

    s1 = pd.Series(yt)
    s2 = pd.Series(yp)
    value = s1.corr(s2, method="kendall")
    return safe_float(value)


def evaluate_count_metrics(y_true: Any, y_pred: Any) -> list[MetricResult]:
    """Compute standard count/regression metrics."""

    yt, yp, _ = valid_pair_mask(y_true, y_pred)
    n = int(len(yt))

    return [
        MetricResult("mae", mae(yt, yp), higher_is_better=False, n=n),
        MetricResult("rmse", rmse(yt, yp), higher_is_better=False, n=n),
        MetricResult("bias_pred_minus_obs", bias(yt, yp), higher_is_better=False, n=n),
        MetricResult(
            "mape_positive_obs_only",
            mean_absolute_percentage_error_safe(yt, yp),
            higher_is_better=False,
            n=n,
        ),
        MetricResult(
            "mean_poisson_deviance",
            mean_poisson_deviance(yt, yp),
            higher_is_better=False,
            n=n,
            notes="Predictions clipped to EPS for deviance calculation.",
        ),
        MetricResult("pearson_corr", pearson_corr(yt, yp), higher_is_better=True, n=n),
        MetricResult("spearman_corr", spearman_corr(yt, yp), higher_is_better=True, n=n),
        MetricResult("kendall_corr", kendall_corr(yt, yp), higher_is_better=True, n=n),
    ]


def top_k_indices(values: Any, k: int) -> np.ndarray:
    """Return indices of the top-k values, with stable deterministic tie handling."""

    arr = to_numpy_1d(values)
    if k <= 0:
        raise MetricError("k must be positive.")

    n = len(arr)
    if n == 0:
        return np.array([], dtype=int)

    k_eff = min(k, n)
    # Stable sort: highest value first, original order breaks ties.
    order = np.lexsort((np.arange(n), -arr))
    return order[:k_eff]


def top_fraction_k(n: int, fraction: float) -> int:
    """Convert a fraction to a nonzero K."""

    if not (0 < fraction <= 1):
        raise MetricError("fraction must be in (0, 1].")
    return max(1, int(math.ceil(n * fraction)))


def top_k_overlap(y_true: Any, y_score: Any, k: int) -> dict[str, Any]:
    """
    Overlap between predicted top-k and observed top-k.

    This is useful when "relevance" means being among the highest-burden
    tract-months rather than exceeding a fixed threshold.

    Since predicted top-k and observed top-k have the same set size, overlap
    rate is overlap / k. The returned ``precision`` and ``recall`` aliases are
    retained for compatibility, but ``overlap_rate`` is the preferred name.
    """

    yt, ys, _ = valid_pair_mask(y_true, y_score)
    n = len(yt)
    if n == 0:
        return {
            "k": k,
            "n": 0,
            "overlap": None,
            "overlap_rate": None,
            "precision": None,
            "recall": None,
            "jaccard": None,
        }

    k_eff = min(k, n)
    observed_top = set(top_k_indices(yt, k_eff).tolist())
    predicted_top = set(top_k_indices(ys, k_eff).tolist())
    overlap = len(observed_top & predicted_top)
    union = observed_top | predicted_top
    overlap_rate = overlap / k_eff if k_eff else None

    return {
        "k": k_eff,
        "n": n,
        "overlap": overlap,
        "overlap_rate": overlap_rate,
        "precision": overlap_rate,
        "recall": overlap_rate,
        "jaccard": overlap / len(union) if union else None,
    }


def precision_at_k(
    y_true: Any,
    y_score: Any,
    k: int,
    relevance_threshold: float,
) -> float | None:
    """Precision among predicted top-k rows using a fixed relevance threshold."""

    yt, ys, _ = valid_pair_mask(y_true, y_score)
    if len(yt) == 0:
        return None

    k_eff = min(k, len(yt))
    predicted_top = top_k_indices(ys, k_eff)
    relevant = yt[predicted_top] >= relevance_threshold
    return float(np.mean(relevant)) if k_eff else None


def recall_at_k(
    y_true: Any,
    y_score: Any,
    k: int,
    relevance_threshold: float,
) -> float | None:
    """Recall among all rows above a fixed relevance threshold."""

    yt, ys, _ = valid_pair_mask(y_true, y_score)
    if len(yt) == 0:
        return None

    relevant_all = yt >= relevance_threshold
    n_relevant = int(relevant_all.sum())
    if n_relevant == 0:
        return None

    k_eff = min(k, len(yt))
    predicted_top = top_k_indices(ys, k_eff)
    return float(relevant_all[predicted_top].sum() / n_relevant)


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at K."""

    k_eff = min(k, len(relevance))
    if k_eff <= 0:
        return 0.0

    gains = relevance[:k_eff]
    discounts = np.log2(np.arange(2, k_eff + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: Any, y_score: Any, k: int) -> float | None:
    """
    Normalized discounted cumulative gain at K.

    Uses nonnegative observed counts as graded relevance. If observed values
    contain negatives, they are shifted so the smallest relevance is zero.
    """

    yt, ys, _ = valid_pair_mask(y_true, y_score)
    if len(yt) == 0:
        return None

    relevance = yt.copy()
    min_rel = np.min(relevance)
    if min_rel < 0:
        relevance = relevance - min_rel

    k_eff = min(k, len(yt))
    predicted_order = top_k_indices(ys, k_eff)
    ideal_order = top_k_indices(relevance, k_eff)

    dcg = dcg_at_k(relevance[predicted_order], k_eff)
    ideal = dcg_at_k(relevance[ideal_order], k_eff)

    if ideal <= 0:
        return None

    return float(dcg / ideal)


def top_decile_overlap(y_true: Any, y_score: Any) -> dict[str, Any]:
    """Overlap between predicted and observed top deciles."""

    yt, ys, _ = valid_pair_mask(y_true, y_score)
    if len(yt) == 0:
        return {
            "k": None,
            "n": 0,
            "overlap": None,
            "overlap_rate": None,
            "precision": None,
            "recall": None,
            "jaccard": None,
        }

    k = top_fraction_k(len(yt), 0.10)
    return top_k_overlap(yt, ys, k)


def evaluate_ranking_metrics(
    y_true: Any,
    y_score: Any,
    k_values: Sequence[int] = (10, 25, 50, 100),
    fractions: Sequence[float] = (0.05, 0.10),
) -> list[MetricResult]:
    """Compute shared ranking/prioritization metrics."""

    yt, ys, _ = valid_pair_mask(y_true, y_score)
    n = int(len(yt))

    results: list[MetricResult] = [
        MetricResult("spearman_corr", spearman_corr(yt, ys), higher_is_better=True, n=n),
        MetricResult("kendall_corr", kendall_corr(yt, ys), higher_is_better=True, n=n),
    ]

    for k in k_values:
        if n == 0:
            continue
        k_eff = min(int(k), n)
        overlap = top_k_overlap(yt, ys, k_eff)
        results.append(
            MetricResult(
                f"top{k_eff}_overlap_rate",
                safe_float(overlap["overlap_rate"]),
                higher_is_better=True,
                n=n,
                notes=f"Predicted top-{k_eff} vs observed top-{k_eff}; overlap / K.",
            )
        )
        results.append(
            MetricResult(
                f"top{k_eff}_overlap_jaccard",
                safe_float(overlap["jaccard"]),
                higher_is_better=True,
                n=n,
                notes=f"Predicted top-{k_eff} vs observed top-{k_eff}.",
            )
        )
        results.append(
            MetricResult(
                f"ndcg_at_{k_eff}",
                ndcg_at_k(yt, ys, k_eff),
                higher_is_better=True,
                n=n,
                notes="Observed counts are used as graded relevance.",
            )
        )

    for fraction in fractions:
        if n == 0:
            continue
        k_eff = top_fraction_k(n, float(fraction))
        pct_label = int(round(fraction * 100))
        overlap = top_k_overlap(yt, ys, k_eff)
        results.append(
            MetricResult(
                f"top_{pct_label}pct_overlap_rate",
                safe_float(overlap["overlap_rate"]),
                higher_is_better=True,
                n=n,
                notes=f"Predicted top {pct_label}% vs observed top {pct_label}%; overlap / K.",
            )
        )
        results.append(
            MetricResult(
                f"top_{pct_label}pct_overlap_jaccard",
                safe_float(overlap["jaccard"]),
                higher_is_better=True,
                n=n,
                notes=f"Predicted top {pct_label}% vs observed top {pct_label}%.",
            )
        )
        results.append(
            MetricResult(
                f"ndcg_at_top_{pct_label}pct",
                ndcg_at_k(yt, ys, k_eff),
                higher_is_better=True,
                n=n,
                notes="Observed counts are used as graded relevance.",
            )
        )

    return results


def auroc_score(y_true_binary: Any, y_score: Any) -> float | None:
    """
    AUROC using the rank-sum formulation.

    Returns None when only one class is present.
    """

    yt, ys, _ = valid_pair_mask(y_true_binary, y_score)
    if len(yt) == 0:
        return None

    yb = (yt > 0).astype(int)
    n_pos = int(yb.sum())
    n_neg = int(len(yb) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = pd.Series(ys).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(ranks[yb == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score(y_true_binary: Any, y_score: Any) -> float | None:
    """
    Average precision / area under precision-recall step curve.

    Returns None when no positive labels are present.
    """

    yt, ys, _ = valid_pair_mask(y_true_binary, y_score)
    if len(yt) == 0:
        return None

    yb = (yt > 0).astype(int)
    n_pos = int(yb.sum())
    if n_pos == 0:
        return None

    order = top_k_indices(ys, len(ys))
    y_sorted = yb[order]

    tp_cumsum = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision = tp_cumsum / ranks

    ap = float(np.sum(precision[y_sorted == 1]) / n_pos)
    return ap


def binary_confusion_counts(
    y_true_binary: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Compute binary confusion counts from probabilities/scores."""

    yt, ys, _ = valid_pair_mask(y_true_binary, y_score)
    yb = (yt > 0).astype(int)
    pred = (ys >= threshold).astype(int)

    tp = int(((pred == 1) & (yb == 1)).sum())
    tn = int(((pred == 0) & (yb == 0)).sum())
    fp = int(((pred == 1) & (yb == 0)).sum())
    fn = int(((pred == 0) & (yb == 1)).sum())

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def f1_score_binary(y_true_binary: Any, y_score: Any, threshold: float = 0.5) -> float | None:
    """Binary F1 score at threshold."""

    counts = binary_confusion_counts(y_true_binary, y_score, threshold=threshold)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    denom = 2 * tp + fp + fn
    if denom == 0:
        return None

    return float(2 * tp / denom)


def balanced_accuracy(y_true_binary: Any, y_score: Any, threshold: float = 0.5) -> float | None:
    """Balanced accuracy at threshold."""

    counts = binary_confusion_counts(y_true_binary, y_score, threshold=threshold)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

    tpr = tp / (tp + fn) if (tp + fn) else None
    tnr = tn / (tn + fp) if (tn + fp) else None

    if tpr is None or tnr is None:
        return None

    return float((tpr + tnr) / 2)


def brier_score(y_true_binary: Any, y_prob: Any) -> float | None:
    """Brier score for binary probabilities."""

    yt, yp, _ = valid_pair_mask(y_true_binary, y_prob)
    if len(yt) == 0:
        return None

    yb = (yt > 0).astype(float)
    prob = np.clip(yp, 0.0, 1.0)
    return float(np.mean((prob - yb) ** 2))


def evaluate_binary_metrics(
    y_true_binary: Any,
    y_score_or_prob: Any,
    threshold: float = 0.5,
) -> list[MetricResult]:
    """Compute shared binary diagnostic metrics."""

    yt, ys, _ = valid_pair_mask(y_true_binary, y_score_or_prob)
    n = int(len(yt))

    return [
        MetricResult("auroc", auroc_score(yt, ys), higher_is_better=True, n=n),
        MetricResult(
            "auprc_average_precision",
            average_precision_score(yt, ys),
            higher_is_better=True,
            n=n,
        ),
        MetricResult(
            "f1_at_threshold",
            f1_score_binary(yt, ys, threshold=threshold),
            higher_is_better=True,
            n=n,
            notes=f"threshold={threshold}",
        ),
        MetricResult(
            "balanced_accuracy_at_threshold",
            balanced_accuracy(yt, ys, threshold=threshold),
            higher_is_better=True,
            n=n,
            notes=f"threshold={threshold}",
        ),
        MetricResult("brier_score", brier_score(yt, ys), higher_is_better=False, n=n),
    ]


def make_metric_row(
    *,
    metric: MetricResult,
    benchmark_id: str,
    dataset_version: str,
    split_name: str,
    split_type: str,
    prediction_setting: str,
    model_stage: str,
    model_name: str,
    target_name: str,
    target_type: str,
    feature_set_name: str,
    n_train: int | None = None,
    n_validation: int | None = None,
    n_test: int | None = None,
    extra_notes: str | None = None,
) -> dict[str, Any]:
    """Create a standardized benchmark metric table row."""

    notes = metric.notes
    if extra_notes:
        notes = f"{notes}; {extra_notes}" if notes else extra_notes

    return {
        "benchmark_id": benchmark_id,
        "dataset_version": dataset_version,
        "split_name": split_name,
        "split_type": split_type,
        "prediction_setting": prediction_setting,
        "model_stage": model_stage,
        "model_name": model_name,
        "target_name": target_name,
        "target_type": target_type,
        "feature_set_name": feature_set_name,
        "metric_name": metric.metric_name,
        "metric_value": metric.metric_value,
        "higher_is_better": metric.higher_is_better,
        "n_rows": metric.n,
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
        "notes": notes,
    }


def make_metrics_dataframe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a metrics DataFrame with stable column order."""

    require_numpy_pandas()

    columns = [
        "benchmark_id",
        "dataset_version",
        "split_name",
        "split_type",
        "prediction_setting",
        "model_stage",
        "model_name",
        "target_name",
        "target_type",
        "feature_set_name",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
        "n_train",
        "n_validation",
        "n_test",
        "notes",
    ]

    df = pd.DataFrame(list(rows))
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df[columns]


def metric_results_to_rows(
    metrics: Sequence[MetricResult],
    **row_kwargs: Any,
) -> list[dict[str, Any]]:
    """Convert a list of MetricResult objects to standardized metric rows."""

    return [
        make_metric_row(metric=metric, **row_kwargs)
        for metric in metrics
    ]


def evaluate_predictions_bundle(
    y_true_count: Any,
    y_pred_count: Any,
    *,
    y_true_binary: Any | None = None,
    y_pred_binary_score: Any | None = None,
    ranking_k_values: Sequence[int] = (10, 25, 50, 100),
    ranking_fractions: Sequence[float] = (0.05, 0.10),
) -> dict[str, list[MetricResult]]:
    """
    Convenience wrapper returning count, ranking, and optional binary metrics.

    This does not know about splits or models; callers should convert results
    to standardized rows using ``metric_results_to_rows``.
    """

    out = {
        "count": evaluate_count_metrics(y_true_count, y_pred_count),
        "ranking": evaluate_ranking_metrics(
            y_true_count,
            y_pred_count,
            k_values=ranking_k_values,
            fractions=ranking_fractions,
        ),
    }

    if y_true_binary is not None and y_pred_binary_score is not None:
        out["binary"] = evaluate_binary_metrics(y_true_binary, y_pred_binary_score)

    return out


__all__ = [
    "EPS",
    "MetricError",
    "MetricResult",
    "average_precision_score",
    "auroc_score",
    "balanced_accuracy",
    "bias",
    "binary_confusion_counts",
    "brier_score",
    "dcg_at_k",
    "evaluate_binary_metrics",
    "evaluate_count_metrics",
    "evaluate_predictions_bundle",
    "evaluate_ranking_metrics",
    "f1_score_binary",
    "kendall_corr",
    "mae",
    "make_metric_row",
    "make_metrics_dataframe",
    "mean_absolute_percentage_error_safe",
    "mean_negative_binomial_log_likelihood",
    "mean_poisson_deviance",
    "metric_results_to_rows",
    "ndcg_at_k",
    "pearson_corr",
    "precision_at_k",
    "recall_at_k",
    "rmse",
    "spearman_corr",
    "top_decile_overlap",
    "top_fraction_k",
    "top_k_indices",
    "top_k_overlap",
]