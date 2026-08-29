"""
segment_analysis.py
=====================
Segmented A/B analysis for the RetainIQ project.

The overall gate_30 vs gate_40 comparison can hide effects that only show up
(or reverse) within a specific slice of players — this is the classic
"Simpson's paradox" risk of only ever looking at an aggregate effect. This
module re-runs the frequentist + Bayesian comparisons within two kinds of
player segments:

1. ENGAGEMENT segments — casual / regular / hardcore, based on how many
   game rounds a player played in their first 14 days (`sum_gamerounds`).
2. GATE-REACHED segments — an approximation of whether a player played
   enough rounds to plausibly have reached the gate at all.

IMPORTANT CAVEAT — read before trusting the gate-reached segment
-------------------------------------------------------------------
The Cookie Cats dataset does NOT contain the player's actual in-game level —
only `sum_gamerounds` (total rounds played across all levels in 14 days).
There is no fixed, known mapping of "1 round = 1 level", so we CANNOT know
for certain whether a given player actually reached level 30/40. The
`reached_gate_segment()` function below uses `sum_gamerounds >= gate_threshold`
as a rough proxy (assuming, generously, that reaching level N takes at least
N rounds). This will systematically UNDER-count "reached the gate" (some
levels take many rounds to clear) — treat this segment as an approximate,
directional cut, not a precise causal population. Say this explicitly in
your report/README so it doesn't come across as more rigorous than it is.

A second important warning: BOTH segmentation schemes here are built from
`sum_gamerounds`, which is a POST-treatment variable (it's measured after
the gate could have already changed a player's behavior). This means
segment membership itself can be affected by which arm a player was in —
so segment-level comparisons should be read as "how does the effect look
within different engagement bands" (a data description), not as if
engagement were a pre-treatment, randomly-assigned covariate. This is a
form of what's sometimes called "post-treatment bias" or "conditioning on a
collider" — flag it in your write-up as a known limitation, not something
this module can fix for you.

Functions
---------
- add_engagement_segment   : add a casual/regular/hardcore column via quantiles
- add_reached_gate_segment : add an approximate "reached the gate" boolean column
- run_segmented_comparison : run frequentist + Bayesian tests within each segment
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from frequentist_tests import two_proportion_z_test, ZTestResult
from bayesian_tests import (
    beta_binomial_posterior,
    prob_b_better_than_a,
    credible_interval,
    BayesianComparisonResult,
)


# ---------------------------------------------------------------------------
# 1. Engagement segmentation
# ---------------------------------------------------------------------------

def add_engagement_segment(
    df: pd.DataFrame,
    rounds_col: str = "sum_gamerounds",
    labels: tuple[str, str, str] = ("casual", "regular", "hardcore"),
) -> pd.DataFrame:
    """
    Add an `engagement_segment` column by splitting players into tertiles of
    `sum_gamerounds` (roughly equal-sized groups: bottom third = casual,
    middle third = regular, top third = hardcore).

    Why tertiles (quantile-based) instead of fixed thresholds (e.g. <10,
    10-50, >50 rounds)?
    - Quantile-based cuts adapt to this dataset's actual distribution and
      guarantee reasonably sized groups in each segment (important for the
      z-test's sample-size assumptions to hold within each segment).
    - Fixed thresholds are more interpretable ("hardcore = 50+ rounds") but
      risk producing a tiny, underpowered segment if the distribution is
      skewed — which `sum_gamerounds` typically is (heavy right skew, most
      players churn early). Feel free to swap in fixed thresholds if you
      want more interpretable segment definitions for your write-up; just
      sanity-check segment sizes if you do (see the printed counts in
      `run_segmented_comparison`).

    Returns a COPY of df with the new column added (does not mutate in place).
    """
    out = df.copy()
    out["engagement_segment"] = pd.qcut(
        out[rounds_col], q=3, labels=labels, duplicates="drop"
    )
    return out


# ---------------------------------------------------------------------------
# 2. Gate-reached segmentation (approximate — see module docstring caveat)
# ---------------------------------------------------------------------------

def add_reached_gate_segment(
    df: pd.DataFrame,
    version_col: str = "version",
    rounds_col: str = "sum_gamerounds",
) -> pd.DataFrame:
    """
    Add a `reached_gate` boolean column: an APPROXIMATE flag for whether a
    player plausibly played enough rounds to reach their assigned gate
    (30 rounds for the `gate_30` arm, 40 rounds for the `gate_40` arm).

    See the module docstring for why this is a rough proxy, not a precise
    "did this player reach level N" flag — there is no level data in this
    dataset, only total rounds played.

    Returns a COPY of df with the new column added.
    """
    out = df.copy()
    gate_threshold = out[version_col].map({"gate_30": 30, "gate_40": 40})
    out["reached_gate"] = out[rounds_col] >= gate_threshold
    return out


# ---------------------------------------------------------------------------
# 3. Run frequentist + Bayesian comparisons within each segment
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    segment_name: str
    n_a: int
    n_b: int
    rate_a: float
    rate_b: float
    frequentist: ZTestResult
    bayesian: BayesianComparisonResult


def run_segmented_comparison(
    df: pd.DataFrame,
    segment_col: str,
    outcome_col: str = "retention_1",
    version_col: str = "version",
    group_a_label: str = "gate_30",
    group_b_label: str = "gate_40",
    min_segment_size: int = 100,
) -> list[SegmentResult]:
    """
    For each unique value of `segment_col`, run BOTH the frequentist
    two-proportion z-test and the Bayesian Beta-Binomial comparison of
    `outcome_col` between `group_a_label` and `group_b_label`.

    Segments smaller than `min_segment_size` (per arm) are skipped with a
    printed warning — small segments both violate the z-test's normal-
    approximation assumption and produce wide, uninformative Bayesian
    credible intervals, so there's little value reporting them either way.

    Returns a list of SegmentResult, one per (non-skipped) segment — iterate
    over this list to build a summary table or bar chart in your notebook.
    """
    results: list[SegmentResult] = []

    for segment_value in df[segment_col].dropna().unique():
        segment_df = df[df[segment_col] == segment_value]

        group_a = segment_df[segment_df[version_col] == group_a_label]
        group_b = segment_df[segment_df[version_col] == group_b_label]

        n_a, n_b = len(group_a), len(group_b)
        if n_a < min_segment_size or n_b < min_segment_size:
            print(
                f"[skipped] segment '{segment_value}': n_a={n_a}, n_b={n_b} "
                f"(below min_segment_size={min_segment_size})"
            )
            continue

        successes_a = int(group_a[outcome_col].sum())
        successes_b = int(group_b[outcome_col].sum())

        freq_result = two_proportion_z_test(successes_a, n_a, successes_b, n_b)

        post_a = beta_binomial_posterior(successes_a, n_a)
        post_b = beta_binomial_posterior(successes_b, n_b)
        bayes_result = BayesianComparisonResult(
            prob_b_better=prob_b_better_than_a(post_a, post_b),
            diff_mean=post_b.mean - post_a.mean,
            credible_interval=credible_interval(post_a, post_b),
            credible_level=0.95,
            expected_loss_choosing_a=None,  # not computed here to keep this fast; use bayesian_tests.expected_loss() directly if needed
            expected_loss_choosing_b=None,
        )

        results.append(
            SegmentResult(
                segment_name=str(segment_value),
                n_a=n_a,
                n_b=n_b,
                rate_a=successes_a / n_a,
                rate_b=successes_b / n_b,
                frequentist=freq_result,
                bayesian=bayes_result,
            )
        )

    return results


def print_segment_results(results: list[SegmentResult]) -> None:
    """Pretty-print a list of SegmentResult as a quick console table."""
    print(f"{'Segment':<12} {'n_A':>7} {'n_B':>7} {'rate_A':>8} {'rate_B':>8} "
          f"{'p-value':>9} {'sig?':>6} {'P(B>A)':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r.segment_name:<12} {r.n_a:>7} {r.n_b:>7} "
            f"{r.rate_a:>8.2%} {r.rate_b:>8.2%} "
            f"{r.frequentist.p_value:>9.4f} {str(r.frequentist.significant):>6} "
            f"{r.bayesian.prob_b_better:>8.2%}"
        )


if __name__ == "__main__":
    # Smoke test with synthetic data so this file can be run standalone
    # (`python src/segment_analysis.py`) before you have the real CSV wired in.
    import numpy as np

    rng = np.random.default_rng(0)
    n = 50_000
    fake_df = pd.DataFrame({
        "version": rng.choice(["gate_30", "gate_40"], size=n),
        "sum_gamerounds": rng.exponential(scale=20, size=n).astype(int),
    })
    # Simulate retention correlated with engagement, plus a small gate effect
    base_rate = 0.1 + 0.4 * (1 - np.exp(-fake_df["sum_gamerounds"] / 30))
    gate_effect = np.where(fake_df["version"] == "gate_40", -0.01, 0.0)
    fake_df["retention_1"] = rng.binomial(1, np.clip(base_rate + gate_effect, 0, 1))

    fake_df = add_engagement_segment(fake_df)
    fake_df = add_reached_gate_segment(fake_df)

    print("=== By engagement segment ===")
    results = run_segmented_comparison(fake_df, segment_col="engagement_segment")
    print_segment_results(results)

    print("\n=== By reached_gate segment ===")
    results = run_segmented_comparison(fake_df, segment_col="reached_gate")
    print_segment_results(results)