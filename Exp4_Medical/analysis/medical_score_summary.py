#!/usr/bin/env python3
"""Utility to summarise CTMS vs medical score relationships.

Reads the per-subject metrics CSV produced by ``run_exp4_medical.py`` and
computes richer statistics for paper-ready reporting:

* Pearson correlation (r, p) for each CTMS dimension and selected
  non-negative convex combinations.
* Bootstrap confidence intervals and permutation-based p-values for the
  strongest configuration per medical score.
* Cross-validated R^2 from a linear model using all four CTMS dimensions.
* Optional binary classification AUC (MoCA < threshold).

Outputs a CSV/JSON summary alongside a Markdown table for direct inclusion in
reports.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

DIMS = ["circadian_score", "task_score", "movement_score", "social_score"]


@dataclass
class CorrelationResult:
    score: str
    n: int
    feature: str
    weights: Dict[str, float]
    pearson_r: float
    pearson_p: float
    bootstrap_ci: Tuple[float, float]
    perm_p: float
    cv_r2: float
    auc: float | None

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "n": self.n,
            "feature": self.feature,
            "weights": self.weights,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "ci_low": self.bootstrap_ci[0],
            "ci_high": self.bootstrap_ci[1],
            "perm_p": self.perm_p,
            "cv_r2": self.cv_r2,
            "auc": self.auc,
        }


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [col for col in df.columns if col.endswith("_score") or col in {"moca", "zbi", "dss", "fas", "age"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[i] = stats.pearsonr(x[idx], y[idx])[0]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = stats.pearsonr(x, y)[0]
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if abs(stats.pearsonr(x, y_perm)[0]) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def generate_weight_grid(step: float) -> Iterable[Tuple[float, float, float, float]]:
    values = np.arange(0.0, 1.0 + 1e-6, step)
    for a, b, c in itertools.product(values, repeat=3):
        d = 1.0 - (a + b + c)
        if d < -1e-6:
            continue
        d = max(0.0, d)
        total = a + b + c + d
        if total == 0:
            continue
        yield (a / total, b / total, c / total, d / total)


def best_combination(df: pd.DataFrame, score: str, weight_step: float = 0.1) -> Tuple[str, Dict[str, float], float, float]:
    sub = df.dropna(subset=[score])
    if sub.empty:
        return "", {}, np.nan, np.nan

    target = sub[score].to_numpy(dtype=float)
    best_feature = ""
    best_weights: Dict[str, float] = {}
    best_r = 0.0
    best_p = np.nan

    # Single dimensions
    for dim in DIMS:
        r, p = pearson(sub[dim].to_numpy(dtype=float), target)
        if np.isnan(r):
            continue
        if abs(r) > abs(best_r):
            best_r = r
            best_p = p
            best_feature = dim
            best_weights = {dim: 1.0}

    # Convex combinations
    dim_values = sub[DIMS].to_numpy(dtype=float)
    for weights in generate_weight_grid(weight_step):
        combo = np.dot(dim_values, np.asarray(weights))
        r, p = pearson(combo, target)
        if np.isnan(r):
            continue
        if abs(r) > abs(best_r):
            best_r = r
            best_p = p
            best_feature = "weighted"
            best_weights = {dim: float(w) for dim, w in zip(DIMS, weights)}

    return best_feature, best_weights, best_r, best_p


def cv_r2(df: pd.DataFrame, score: str, seed: int = 0) -> float:
    sub = df.dropna(subset=[score])
    if len(sub) < 5:
        return float("nan")
    X = sub[DIMS].to_numpy(dtype=float)
    y = sub[score].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression()
    cv = KFold(n_splits=min(5, len(sub)), shuffle=True, random_state=seed)
    scores = cross_val_score(model, Xs, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def maybe_auc(sub: pd.DataFrame, score: str) -> float | None:
    if score != "moca":
        return None
    if "combined_score" not in sub:
        return None
    threshold = 26
    y = (sub[score] < threshold).astype(int)
    if y.nunique() < 2:
        return None
    preds = sub["combined_score"].to_numpy(dtype=float)
    return float(roc_auc_score(y, preds))


def summarise(df: pd.DataFrame, weight_step: float, seed: int) -> List[CorrelationResult]:
    results: List[CorrelationResult] = []
    for score in ["moca", "zbi", "dss", "fas"]:
        feature, weights, r, p = best_combination(df, score, weight_step)
        sub = df.dropna(subset=[score])
        n = len(sub)
        if n < 3 or not weights:
            continue
        dim_matrix = sub[DIMS].to_numpy(dtype=float)
        w_vec = np.zeros(len(DIMS), dtype=float)
        for idx, dim in enumerate(DIMS):
            w_vec[idx] = weights.get(dim, 0.0)
        combined = np.dot(dim_matrix, w_vec)

        ci = bootstrap_ci(combined, sub[score].to_numpy(dtype=float), seed=seed)
        perm = permutation_p(combined, sub[score].to_numpy(dtype=float), seed=seed)
        cv = cv_r2(sub, score, seed=seed)
        auc = maybe_auc(sub.assign(combined_score=combined), score)

        results.append(CorrelationResult(
            score=score,
            n=n,
            feature=feature,
            weights=weights,
            pearson_r=r,
            pearson_p=p,
            bootstrap_ci=ci,
            perm_p=perm,
            cv_r2=cv,
            auc=auc,
        ))
    return results


def to_markdown(results: List[CorrelationResult]) -> str:
    header = "| Score | n | Feature | Weights | r | 95% CI | p_perm | CV R^2 | AUC |\n"
    header += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    rows = []
    for res in results:
        weight_str = ", ".join(f"{dim[0].upper()}={w:.2f}" for dim, w in res.weights.items())
        ci_str = f"[{res.bootstrap_ci[0]:.3f}, {res.bootstrap_ci[1]:.3f}]"
        auc_str = f"{res.auc:.3f}" if res.auc is not None else "—"
        rows.append(
            f"| {res.score.upper()} | {res.n} | {res.feature} | {weight_str} | "
            f"{res.pearson_r:.3f} | {ci_str} | {res.perm_p:.3f} | {res.cv_r2:.3f} | {auc_str} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("../outputs/exp4_medical_subject_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("../outputs/exp4_medical_correlation_summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("../outputs/exp4_medical_correlation_table.md"))
    parser.add_argument("--weight-step", type=float, default=0.1, help="Grid step for convex combination weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap/permutation")
    args = parser.parse_args()

    df = load_metrics(args.metrics)
    results = summarise(df, weight_step=args.weight_step, seed=args.seed)

    args.output.write_text(json.dumps([res.to_dict() for res in results], indent=2), encoding="utf-8")
    args.markdown.write_text(to_markdown(results), encoding="utf-8")

    print(f"Summaries written to {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
