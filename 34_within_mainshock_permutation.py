from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from analysis_core import filter_analysis_sample, json_ready, within_group_permutation

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def run_permutation(
    df: pd.DataFrame,
    *,
    n_perm: int = 5000,
    seed: int = 7,
) -> Dict[str, object]:
    sample = filter_analysis_sample(df)
    return within_group_permutation(sample, n_perm=n_perm, seed=seed)


def main() -> None:
    sg = pd.read_csv(OUT_DIR / "subgroup_summary.csv")
    result = run_permutation(sg, n_perm=5000, seed=7)
    null = np.asarray(result.pop("null"), dtype=float)
    out = json_ready(result)
    (OUT_DIR / "within_mainshock_permutation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame({"perm": np.arange(len(null)), "null_slope": null}).to_csv(OUT_DIR / "within_mainshock_permutation_null.csv", index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9.0, 4.6))
    plt.hist(null, bins=40, alpha=0.85, color="#f97316")
    plt.axvline(result["observed_slope"], color="black", lw=2, label=f"observed={result['observed_slope']:.4f}")
    plt.title(f"Within-mainshock permutation test (n_perm={result['n_perm']}): slope null distribution")
    plt.xlabel("Slope (within-mainshock / FE)")
    plt.ylabel("Count")
    plt.grid(True, ls=":", alpha=0.35)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_within_mainshock_permutation.png", dpi=180)
    plt.close()

    print("Saved permutation test outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
