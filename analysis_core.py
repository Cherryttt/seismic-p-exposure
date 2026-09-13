from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


EXPOSURE_TERM = "log10_pop_sum_subgroup"
ANALYSIS_SPEC = (
    "p ~ log10(pop_sum_subgroup) + log(n_events) + "
    "C(subgroup_type) + FE(mainshock_id)"
)
FALLBACK_SUBGROUPS = {"all_depth", "all_relaxed"}


@dataclass(frozen=True)
class WithinDesign:
    matrix: np.ndarray
    outcome: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    rank: int
    frame: pd.DataFrame


def _is_full_radius_ring(label: object, full_radius_km: float) -> bool:
    match = re.fullmatch(
        r"\s*0(?:\.0+)?\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*km\s*",
        str(label),
        flags=re.IGNORECASE,
    )
    return bool(match and math.isclose(float(match.group(1)), full_radius_km))


def filter_analysis_sample(
    df: pd.DataFrame,
    *,
    full_radius_km: float = 400.0,
    min_events: int = 25,
) -> pd.DataFrame:
    """Return the non-aggregate subgroup rows used for inference and ML."""
    required = {
        "mainshock_id",
        "subgroup_type",
        "subgroup",
        "n_events",
        "pop_sum_subgroup",
        "p",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing analysis columns: {sorted(missing)}")

    out = df.copy()
    labels = out["subgroup"].astype(str)
    fallback = labels.isin(FALLBACK_SUBGROUPS)
    full_ring = (out["subgroup_type"].astype(str) == "ring") & labels.map(
        lambda value: _is_full_radius_ring(value, full_radius_km)
    )
    numeric_ok = (
        pd.to_numeric(out["n_events"], errors="coerce").ge(min_events)
        & pd.to_numeric(out["pop_sum_subgroup"], errors="coerce").gt(0)
        & np.isfinite(pd.to_numeric(out["p"], errors="coerce"))
    )
    out = out.loc[~fallback & ~full_ring & numeric_ok].copy()
    out["analysis_included"] = True
    return out.reset_index(drop=True)


def _append_if_independent(
    columns: List[np.ndarray],
    names: List[str],
    candidate: np.ndarray,
    name: str,
) -> None:
    candidate = np.asarray(candidate, dtype=float)
    if not np.all(np.isfinite(candidate)):
        return
    before = np.column_stack(columns) if columns else np.empty((len(candidate), 0))
    after = np.column_stack([*columns, candidate])
    before_rank = np.linalg.matrix_rank(before) if columns else 0
    if np.linalg.matrix_rank(after) > before_rank:
        columns.append(candidate)
        names.append(name)


def build_within_design(df: pd.DataFrame, *, outcome: str = "p") -> WithinDesign:
    sample = filter_analysis_sample(df) if "analysis_included" not in df.columns else df.copy()
    needed = [
        outcome,
        "pop_sum_subgroup",
        "n_events",
        "mainshock_id",
        "subgroup_type",
    ]
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    if len(sample) < 5 or sample["mainshock_id"].nunique() < 2:
        raise ValueError("At least five rows and two mainshocks are required.")

    n = len(sample)
    columns: List[np.ndarray] = []
    names: List[str] = []
    _append_if_independent(columns, names, np.ones(n), "const")
    _append_if_independent(
        columns,
        names,
        np.log10(sample["pop_sum_subgroup"].to_numpy(float)),
        EXPOSURE_TERM,
    )
    _append_if_independent(
        columns,
        names,
        np.log(np.maximum(sample["n_events"].to_numpy(float), 1.0)),
        "log_n_events",
    )

    subgroup_types = sample["subgroup_type"].astype(str)
    for value in sorted(subgroup_types.unique())[1:]:
        _append_if_independent(
            columns,
            names,
            (subgroup_types == value).to_numpy(float),
            f"type_{value}",
        )

    groups = sample["mainshock_id"].astype(str)
    for value in sorted(groups.unique())[1:]:
        _append_if_independent(
            columns,
            names,
            (groups == value).to_numpy(float),
            f"fe_{value}",
        )

    matrix = np.column_stack(columns)
    return WithinDesign(
        matrix=matrix,
        outcome=sample[outcome].to_numpy(float),
        groups=groups.to_numpy(str),
        feature_names=names,
        rank=int(np.linalg.matrix_rank(matrix)),
        frame=sample.reset_index(drop=True),
    )


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def fit_within_fe(df: pd.DataFrame, *, outcome: str = "p") -> Dict[str, object]:
    design = build_within_design(df, outcome=outcome)
    x = design.matrix
    y = design.outcome
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    n = len(y)
    k = x.shape[1]
    dof = n - k

    naive_ci: Dict[str, List[float | None]] = {}
    naive_pvalues: Dict[str, float | None] = {}
    if dof > 0:
        from scipy.stats import t as t_dist

        sigma2 = float(residual @ residual) / dof
        covariance = np.linalg.pinv(x.T @ x) * sigma2
        se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        critical = float(t_dist.ppf(0.975, dof))
        for idx, name in enumerate(design.feature_names):
            naive_ci[name] = [
                _finite_or_none(beta[idx] - critical * se[idx]),
                _finite_or_none(beta[idx] + critical * se[idx]),
            ]
            if se[idx] > 0:
                pvalue = 2.0 * (1.0 - t_dist.cdf(abs(beta[idx] / se[idx]), dof))
                naive_pvalues[name] = _finite_or_none(pvalue)
            else:
                naive_pvalues[name] = None

    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_residual = float(residual @ residual)
    r2 = 1.0 - ss_residual / ss_total if ss_total > 1e-15 else None
    adjusted_r2 = (
        1.0 - (1.0 - r2) * (n - 1) / dof
        if r2 is not None and dof > 0
        else None
    )
    return {
        "analysis_spec": ANALYSIS_SPEC,
        "n": n,
        "n_groups": int(pd.Series(design.groups).nunique()),
        "n_parameters": k,
        "rank": design.rank,
        "dof": max(dof, 0),
        "r2": _finite_or_none(r2) if r2 is not None else None,
        "adj_r2": _finite_or_none(adjusted_r2) if adjusted_r2 is not None else None,
        "feature_names": design.feature_names,
        "coef": {name: float(beta[idx]) for idx, name in enumerate(design.feature_names)},
        "naive_iid_ci95": naive_ci,
        "naive_iid_pvalues": naive_pvalues,
    }


def permute_exposure_within_groups(
    df: pd.DataFrame,
    *,
    seed: int,
    exposure: str = "pop_sum_subgroup",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _, index in out.groupby("mainshock_id", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        values = out.loc[positions, exposure].to_numpy(copy=True)
        out.loc[positions, exposure] = rng.permutation(values)
    return out


def within_group_permutation(
    df: pd.DataFrame,
    *,
    n_perm: int = 5000,
    seed: int = 7,
    outcome: str = "p",
) -> Dict[str, object]:
    sample = filter_analysis_sample(df) if "analysis_included" not in df.columns else df.copy()
    observed = fit_within_fe(sample, outcome=outcome)["coef"][EXPOSURE_TERM]
    null = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        permuted = permute_exposure_within_groups(sample, seed=seed + idx)
        null[idx] = fit_within_fe(permuted, outcome=outcome)["coef"][EXPOSURE_TERM]
    pvalue = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return {
        "analysis_spec": ANALYSIS_SPEC,
        "n": int(len(sample)),
        "n_groups": int(sample["mainshock_id"].nunique()),
        "n_perm": int(n_perm),
        "observed_slope": float(observed),
        "p_value_two_sided": float(pvalue),
        "null_mean": float(np.mean(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "null": null,
    }


def cluster_bootstrap_fe(
    df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 7,
    outcome: str = "p",
) -> Dict[str, object]:
    sample = filter_analysis_sample(df) if "analysis_included" not in df.columns else df.copy()
    group_values = np.array(sorted(sample["mainshock_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    slopes: List[float] = []
    for boot_idx in range(n_boot):
        selected = rng.choice(group_values, size=len(group_values), replace=True)
        blocks = []
        for copy_idx, group in enumerate(selected):
            block = sample.loc[sample["mainshock_id"].astype(str) == group].copy()
            block["mainshock_id"] = f"boot_{copy_idx}_{group}"
            blocks.append(block)
        try:
            fitted = fit_within_fe(pd.concat(blocks, ignore_index=True), outcome=outcome)
            slope = fitted["coef"].get(EXPOSURE_TERM)
            if slope is not None and math.isfinite(float(slope)):
                slopes.append(float(slope))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not slopes:
        raise RuntimeError("No valid cluster bootstrap draws were produced.")
    draws = np.asarray(slopes, dtype=float)
    return {
        "analysis_spec": ANALYSIS_SPEC,
        "n_boot_requested": int(n_boot),
        "n_boot_valid": int(len(draws)),
        "median": float(np.median(draws)),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "probability_negative": float(np.mean(draws < 0)),
        "draws": draws,
    }


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float | None]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if len(true) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_total = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / ss_total if ss_total > 1e-12 else None
    return {"mae": mae, "rmse": rmse, "r2": _finite_or_none(r2) if r2 is not None else None}


def json_ready(value: object) -> object:
    """Convert NumPy objects and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items() if key != "null"}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value
