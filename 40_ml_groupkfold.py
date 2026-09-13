from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis_core import filter_analysis_sample, json_ready, regression_metrics
from config import SEED


ALPHA_GRID = (0.01, 0.1, 1.0, 2.0, 10.0, 100.0)


@dataclass(frozen=True)
class BinAssignment:
    labels: np.ndarray
    thresholds: Tuple[float, float]


def load_subgroup_summary(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "subgroup_summary.csv"
    if not path.exists():
        raise SystemExit("Missing out/subgroup_summary.csv. Run 30_join_and_plot.py first.")
    return pd.read_csv(path)


def _parse_interval_midpoint(label: str) -> float:
    clean = str(label).replace("km", "").strip()
    if clean.startswith(">="):
        return float(clean.replace(">=", ""))
    low, high = clean.split("-", 1)
    return 0.5 * (float(low) + float(high))


def make_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, pd.DataFrame]:
    frame = filter_analysis_sample(df)
    frame["log10_pop_subgroup"] = np.log10(frame["pop_sum_subgroup"].to_numpy(float))
    frame["log_n_events"] = np.log(np.maximum(frame["n_events"].to_numpy(float), 1.0))
    frame["ring_mid_km"] = 0.0
    frame["depth_mid_km"] = 0.0

    for index, row in frame.iterrows():
        try:
            if row["subgroup_type"] == "ring":
                frame.at[index, "ring_mid_km"] = _parse_interval_midpoint(row["subgroup"])
            elif row["subgroup_type"] == "depth":
                frame.at[index, "depth_mid_km"] = _parse_interval_midpoint(row["subgroup"])
        except (ValueError, TypeError):
            continue

    names = [
        "log10_pop_subgroup",
        "mainshock_mag",
        "log_n_events",
        "ring_mid_km",
        "depth_mid_km",
    ]
    features = [frame[name].to_numpy(float) for name in names]

    subgroup_types = frame["subgroup_type"].astype(str)
    for value in sorted(subgroup_types.unique()):
        features.append((subgroup_types == value).to_numpy(float))
        names.append(f"type_{value}")

    region_mask = subgroup_types == "region"
    region_labels = sorted(frame.loc[region_mask, "subgroup"].astype(str).unique())
    for value in region_labels:
        features.append((region_mask & (frame["subgroup"].astype(str) == value)).to_numpy(float))
        names.append(f"region_{value}")

    matrix = np.column_stack(features)
    outcome = frame["p"].to_numpy(float)
    groups = frame["mainshock_id"].astype(str).to_numpy()
    valid = np.isfinite(outcome) & np.all(np.isfinite(matrix), axis=1)
    return matrix[valid], names, outcome[valid], groups[valid], frame.loc[valid].reset_index(drop=True)


def leave_one_group_out(groups: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=str)
    splits = []
    for group in sorted(np.unique(groups)):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        if len(train) and len(test):
            splits.append((str(group), train, test))
    return splits


def _standardization(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def ridge_train(
    matrix: np.ndarray,
    outcome: np.ndarray,
    *,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = _standardization(matrix)
    standardized = (matrix - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + alpha * penalty) @ (design.T @ outcome)
    return beta, mean, scale


def ridge_predict(
    beta: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    standardized = (matrix - mean) / scale
    return np.column_stack([np.ones(len(standardized)), standardized]) @ beta


def ols_train(matrix: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.column_stack([np.ones(len(matrix)), matrix])) @ outcome


def ols_predict(beta: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(matrix)), matrix]) @ beta


def training_quantile_bins(y_train: np.ndarray, values: np.ndarray) -> BinAssignment:
    thresholds_array = np.quantile(np.asarray(y_train, dtype=float), [0.33, 0.66])
    thresholds = (float(thresholds_array[0]), float(thresholds_array[1]))
    labels = np.digitize(np.asarray(values, dtype=float), bins=thresholds_array)
    return BinAssignment(labels=labels.astype(int), thresholds=thresholds)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    n_classes: int = 3,
) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    accuracy = float(np.mean(true == pred))
    f1_values = []
    for value in range(n_classes):
        true_positive = int(np.sum((true == value) & (pred == value)))
        false_positive = int(np.sum((true != value) & (pred == value)))
        false_negative = int(np.sum((true == value) & (pred != value)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_values))}


def _macro_metrics(fold_metrics: List[Dict[str, object]]) -> Dict[str, float | None]:
    result: Dict[str, float | None] = {}
    for metric in ("mae", "rmse"):
        values = [float(row[metric]) for row in fold_metrics if row.get(metric) is not None]
        result[metric] = float(np.mean(values)) if values else None
    r2_values = [float(row["r2"]) for row in fold_metrics if row.get("r2") is not None]
    result["r2"] = float(np.mean(r2_values)) if r2_values else None
    return result


def _select_alpha(
    matrix: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    alpha_grid: Iterable[float],
) -> float:
    grid = tuple(float(value) for value in alpha_grid)
    splits = leave_one_group_out(groups)
    if len(splits) < 2:
        return grid[0]
    scores = []
    for alpha in grid:
        fold_rmse = []
        for _, train, test in splits:
            beta, mean, scale = ridge_train(matrix[train], outcome[train], alpha=alpha)
            metrics = regression_metrics(
                outcome[test], ridge_predict(beta, mean, scale, matrix[test])
            )
            if metrics["rmse"] is not None:
                fold_rmse.append(float(metrics["rmse"]))
        scores.append((float(np.mean(fold_rmse)), alpha))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def permute_group_level_feature(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.array(sorted(np.unique(groups)))
    group_values = np.array([values[groups == group][0] for group in unique_groups])
    shuffled = np.random.default_rng(seed).permutation(group_values)
    mapping = dict(zip(unique_groups, shuffled))
    return np.array([mapping[group] for group in groups], dtype=float)


def _permute_within_groups(values: np.ndarray, groups: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.asarray(values, dtype=float).copy()
    groups = np.asarray(groups, dtype=str)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        result[index] = rng.permutation(result[index])
    return result


def _is_group_level(values: np.ndarray, groups: np.ndarray) -> bool:
    return all(np.unique(values[groups == group]).size == 1 for group in np.unique(groups))


def _group_aware_importance(
    matrix: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    fold_models: List[Dict[str, object]],
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    baseline = np.empty(len(outcome), dtype=float)
    for fold in fold_models:
        test = fold["test_index"]
        baseline[test] = ridge_predict(fold["beta"], fold["mean"], fold["scale"], matrix[test])
    base_rmse = regression_metrics(outcome, baseline)["rmse"]
    importance = np.zeros(matrix.shape[1], dtype=float)
    for feature in range(matrix.shape[1]):
        deltas = []
        for repeat in range(repeats):
            permuted = matrix.copy()
            values = matrix[:, feature]
            draw_seed = seed + 1000 * feature + repeat
            if _is_group_level(values, groups):
                permuted[:, feature] = permute_group_level_feature(values, groups, seed=draw_seed)
            else:
                permuted[:, feature] = _permute_within_groups(values, groups, seed=draw_seed)
            predictions = np.empty(len(outcome), dtype=float)
            for fold in fold_models:
                test = fold["test_index"]
                predictions[test] = ridge_predict(
                    fold["beta"], fold["mean"], fold["scale"], permuted[test]
                )
            rmse = regression_metrics(outcome, predictions)["rmse"]
            deltas.append(float(rmse) - float(base_rmse))
        importance[feature] = float(np.mean(deltas))
    return importance


def run_logo_comparison(
    df: pd.DataFrame,
    *,
    alpha_grid: Iterable[float] = ALPHA_GRID,
    importance_repeats: int = 20,
    seed: int = SEED,
) -> Dict[str, object]:
    matrix, feature_names, outcome, groups, _ = make_features(df)
    splits = leave_one_group_out(groups)
    if len(splits) < 3:
        raise ValueError("At least three mainshocks are required for nested LOGO evaluation.")

    predictions = {
        name: np.full(len(outcome), np.nan, dtype=float)
        for name in ("mean", "ols", "ridge")
    }
    true_classes = np.full(len(outcome), -1, dtype=int)
    predicted_classes = np.full(len(outcome), -1, dtype=int)
    folds: List[Dict[str, object]] = []
    fold_models: List[Dict[str, object]] = []

    for fold_number, (test_group, train, test) in enumerate(splits, start=1):
        selected_alpha = _select_alpha(
            matrix[train], outcome[train], groups[train], tuple(alpha_grid)
        )
        ridge_beta, mean, scale = ridge_train(
            matrix[train], outcome[train], alpha=selected_alpha
        )
        ols_beta = ols_train(matrix[train], outcome[train])
        predictions["mean"][test] = float(np.mean(outcome[train]))
        predictions["ols"][test] = ols_predict(ols_beta, matrix[test])
        predictions["ridge"][test] = ridge_predict(ridge_beta, mean, scale, matrix[test])

        true_bin = training_quantile_bins(outcome[train], outcome[test])
        pred_bin = training_quantile_bins(outcome[train], predictions["ridge"][test])
        true_classes[test] = true_bin.labels
        predicted_classes[test] = pred_bin.labels

        model_metrics = {
            name: regression_metrics(outcome[test], values[test])
            for name, values in predictions.items()
        }
        folds.append(
            {
                "fold": fold_number,
                "group_test": test_group,
                "n_test": int(len(test)),
                "target_sd": float(np.std(outcome[test])),
                "alpha": selected_alpha,
                "class_thresholds": list(true_bin.thresholds),
                "models": model_metrics,
            }
        )
        fold_models.append(
            {"test_index": test, "beta": ridge_beta, "mean": mean, "scale": scale}
        )

    models = {}
    for name, values in predictions.items():
        models[name] = {
            "pooled": regression_metrics(outcome, values),
            "macro": _macro_metrics([fold["models"][name] for fold in folds]),
        }

    importance = _group_aware_importance(
        matrix,
        outcome,
        groups,
        fold_models,
        repeats=importance_repeats,
        seed=seed,
    )
    return {
        "mode": "filtered_subgroup_nested_logo",
        "n_samples": int(len(outcome)),
        "n_groups": int(len(np.unique(groups))),
        "feature_names": feature_names,
        "group_level_features": [
            feature_names[index]
            for index in range(matrix.shape[1])
            if _is_group_level(matrix[:, index], groups)
        ],
        "models": models,
        "classification": classification_metrics(true_classes, predicted_classes),
        "folds": folds,
        "permutation_importance": {
            feature_names[index]: float(importance[index])
            for index in range(len(feature_names))
        },
        "predictions": {
            "mainshock_id": groups.tolist(),
            "true_p": outcome.tolist(),
            **{f"pred_{name}": values.tolist() for name, values in predictions.items()},
            "true_class": true_classes.tolist(),
            "predicted_class": predicted_classes.tolist(),
        },
    }


def _plot_predictions(path: Path, true: np.ndarray, predictions: Dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 5.3))
    colors = {"mean": "#9ca3af", "ols": "#f59e0b", "ridge": "#2563eb"}
    for name in ("mean", "ols", "ridge"):
        plt.scatter(true, predictions[name], s=34, alpha=0.65, label=name.upper(), color=colors[name])
    minimum = float(min(true.min(), *(value.min() for value in predictions.values())))
    maximum = float(max(true.max(), *(value.max() for value in predictions.values())))
    plt.plot([minimum, maximum], [minimum, maximum], "--", color="0.3", lw=1)
    plt.xlabel("True p")
    plt.ylabel("Predicted p")
    plt.title("Nested LOGO predictions: baselines and Ridge")
    plt.legend()
    plt.grid(True, ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_importance(path: Path, importance: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(importance.items(), key=lambda item: item[1])
    plt.figure(figsize=(9.2, 5.2))
    plt.barh([item[0] for item in ordered], [item[1] for item in ordered], color="#2563eb")
    plt.axvline(0.0, color="0.3", lw=1)
    plt.xlabel("Group-aware OOF permutation importance (delta RMSE)")
    plt.title("Ridge feature importance under nested LOGO")
    plt.grid(True, axis="x", ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_confusion(path: Path, true: np.ndarray, predicted: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    matrix = np.zeros((3, 3), dtype=int)
    for actual, estimate in zip(true, predicted):
        matrix[int(actual), int(estimate)] += 1
    plt.figure(figsize=(5.7, 4.9))
    plt.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.xticks(range(3), ["low", "mid", "high"])
    plt.yticks(range(3), ["low", "mid", "high"])
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Leakage-free p-bin classification")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_logo_comparison(load_subgroup_summary(out_dir))
    serializable = json_ready(result)
    for name in ("ml_logo_model_comparison.json", "ml_groupkfold_results.json"):
        (out_dir / name).write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    rows = []
    for model, metrics in result["models"].items():
        rows.append(
            {
                "model": model,
                **{f"pooled_{key}": value for key, value in metrics["pooled"].items()},
                **{f"macro_{key}": value for key, value in metrics["macro"].items()},
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "ml_logo_model_comparison.csv", index=False)

    predictions = result["predictions"]
    true = np.asarray(predictions["true_p"], dtype=float)
    model_predictions = {
        name: np.asarray(predictions[f"pred_{name}"], dtype=float)
        for name in ("mean", "ols", "ridge")
    }
    _plot_predictions(out_dir / "fig_ml_pred_vs_true.png", true, model_predictions)
    _plot_importance(out_dir / "fig_ml_perm_importance.png", result["permutation_importance"])
    _plot_confusion(
        out_dir / "fig_ml_confusion.png",
        np.asarray(predictions["true_class"], dtype=int),
        np.asarray(predictions["predicted_class"], dtype=int),
    )
    print("Saved corrected nested-LOGO outputs to:", out_dir)


if __name__ == "__main__":
    main()
