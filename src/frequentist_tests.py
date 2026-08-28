"""
frequentist_tests.py
=====================
Reusable frequentist hypothesis-testing utilities for the RetainIQ project.

This module answers the core question: "Did moving the gate from level 30 to
level 40 change retention?" using classical (frequentist) statistics —
specifically the two-proportion z-test, since Day-1 and Day-7 retention are
binary outcomes (a player either came back or didn't).

Functions
---------
- two_proportion_z_test        : hypothesis test for a difference in two proportions
- confidence_interval_diff     : CI on the difference in retention rates (Wald + Wilson)
- required_sample_size         : power analysis / sample size calculator
- summarize_ab_test            : convenience wrapper that runs everything at once

Statistical notes are inlined as comments throughout — read them, since a few
of these choices matter for whether your conclusions are trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
    confint_proportions_2indep,
)
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


# ---------------------------------------------------------------------------
# Result containers (dataclasses instead of plain dicts/tuples so that the
# calling code and the reader both get self-documenting field names)
# ---------------------------------------------------------------------------

@dataclass
class ZTestResult:
    group_a_rate: float
    group_b_rate: float
    diff: float                # group_b_rate - group_a_rate
    z_stat: float
    p_value: float
    alpha: float
    significant: bool
    alternative: str


@dataclass
class ConfidenceIntervalResult:
    diff: float
    wald_ci: tuple[float, float]
    wilson_ci: tuple[float, float]
    confidence_level: float


@dataclass
class PowerAnalysisResult:
    required_n_per_group: int
    baseline_rate: float
    minimum_detectable_effect: float
    power: float
    alpha: float


# ---------------------------------------------------------------------------
# 1. Two-proportion z-test
# ---------------------------------------------------------------------------

def two_proportion_z_test(
    count_a: int,
    nobs_a: int,
    count_b: int,
    nobs_b: int,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> ZTestResult:
    """
    Test whether two independent proportions differ.

    In RetainIQ's context: count_a/nobs_a = retained_1 / total for gate_30,
    count_b/nobs_b = retained_1 / total for gate_40 (or the Day-7 equivalents).

    Parameters
    ----------
    count_a, count_b : number of "successes" (e.g. players retained) in each group
    nobs_a, nobs_b   : total number of observations (players) in each group
    alternative      : 'two-sided', 'larger', or 'smaller'
                        - 'two-sided' : is there ANY difference (this is the
                          right default for an exploratory causal question
                          like "did moving the gate change retention?")
                        - 'larger'/'smaller' : only use these if you had a
                          directional hypothesis BEFORE looking at the data
                          (e.g. "we expect gate_40 to increase retention
                          because players see more content before the wall").
                          Picking a one-sided test after peeking at the data
                          to get a smaller p-value is p-hacking — don't do it.
    alpha            : significance threshold (0.05 is conventional; a lower
                        alpha like 0.01 is more conservative)

    Returns
    -------
    ZTestResult

    Statistical assumptions to double-check
    ----------------------------------------
    1. Independence: each player's outcome must be independent of every other
       player's outcome. This is very likely fine for Cookie Cats since users
       don't interact with each other's retention, but always worth stating.
    2. Sample size / normal approximation: the z-test relies on the sampling
       distribution of the sample proportion being approximately normal. The
       standard rule of thumb is n*p >= 5 AND n*(1-p) >= 5 for BOTH groups.
       With the Cookie Cats dataset (~90k users per group, retention ~19-45%)
       this holds comfortably, but the function will warn you if it doesn't.
    3. Random assignment: the z-test tells you whether the OBSERVED difference
       is unlikely under the null of "no difference" — it does NOT by itself
       prove causality. Causality here relies on the A/B assignment actually
       being random, which is why we sanity-check group balance in the EDA
       notebook before trusting this test's causal interpretation.
    """
    p_a = count_a / nobs_a
    p_b = count_b / nobs_b

    # Rule-of-thumb normal-approximation check (n*p and n*(1-p) >= 5)
    for label, p, n in [("A", p_a, nobs_a), ("B", p_b, nobs_b)]:
        if n * p < 5 or n * (1 - p) < 5:
            print(
                f"[warning] Group {label}: n*p or n*(1-p) < 5 "
                f"(n={n}, p={p:.4f}). The normal approximation underlying "
                "the z-test may be unreliable here — consider Fisher's "
                "exact test instead."
            )

    counts = np.array([count_a, count_b])
    nobs = np.array([nobs_a, nobs_b])

    # statsmodels' alternative is phrased as "prop[0] vs prop[1]"
    z_stat, p_value = proportions_ztest(
        count=counts, nobs=nobs, alternative=alternative
    )

    return ZTestResult(
        group_a_rate=p_a,
        group_b_rate=p_b,
        diff=p_b - p_a,
        z_stat=z_stat,
        p_value=p_value,
        alpha=alpha,
        significant=p_value < alpha,
        alternative=alternative,
    )


# ---------------------------------------------------------------------------
# 2. Confidence interval on the difference in proportions
# ---------------------------------------------------------------------------

def confidence_interval_diff(
    count_a: int,
    nobs_a: int,
    count_b: int,
    nobs_b: int,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """
    Compute a confidence interval on (p_b - p_a).

    Why two intervals?
    ------------------
    - Wald CI: the "textbook" formula (diff +/- z * SE). It's simple but is
      KNOWN to perform poorly (can even produce bounds outside [-1, 1], or
      have coverage well below the nominal 95%) when proportions are close to
      0 or 1, or sample sizes are small/unbalanced.
    - Wilson/Newcombe CI: built from Wilson score intervals for each group and
      combined (Newcombe's method), which has better coverage properties in
      practice than the plain Wald interval. This is the interval
      statisticians generally recommend for proportions.

    Recommendation: report the Wilson/Newcombe interval as your primary
    number; use Wald only as a sanity-check / because it's the one most
    readers expect to see. If the two disagree meaningfully, trust Wilson/Newcombe.
    """
    p_a = count_a / nobs_a
    p_b = count_b / nobs_b
    diff = p_b - p_a

    # --- Wald CI (manual, standard formula) ---
    z = _z_score_for_confidence(confidence_level)
    se_wald = np.sqrt(p_a * (1 - p_a) / nobs_a + p_b * (1 - p_b) / nobs_b)
    wald_lo, wald_hi = diff - z * se_wald, diff + z * se_wald

    # --- Wilson/Newcombe-adjusted CI via statsmodels ---
    # "newcomb" combines per-group Wilson score intervals into a CI on the
    # difference — this is what's usually meant by "the Wilson CI" for a
    # difference of two proportions (statsmodels' "wilson" method name is
    # reserved for the ratio/odds-ratio comparisons, not "diff").
    wilson_lo, wilson_hi = confint_proportions_2indep(
        count1=count_b,
        nobs1=nobs_b,
        count2=count_a,
        nobs2=nobs_a,
        method="newcomb",
        compare="diff",
        alpha=1 - confidence_level,
    )

    return ConfidenceIntervalResult(
        diff=diff,
        wald_ci=(wald_lo, wald_hi),
        wilson_ci=(wilson_lo, wilson_hi),
        confidence_level=confidence_level,
    )


def _z_score_for_confidence(confidence_level: float) -> float:
    """Two-sided critical z-value for a given confidence level (e.g. 0.95 -> 1.96)."""
    from scipy.stats import norm

    alpha = 1 - confidence_level
    return norm.ppf(1 - alpha / 2)


# ---------------------------------------------------------------------------
# 3. Sample size / power calculator
# ---------------------------------------------------------------------------

def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> PowerAnalysisResult:
    """
    Compute the required sample size PER GROUP to detect a given effect size
    with a given statistical power.

    This is normally run BEFORE collecting data (to plan an experiment), but
    it's just as useful run RETROSPECTIVELY on the Cookie Cats data to answer:
    "given our actual sample sizes (~45k per group), what's the smallest
    true effect we had good power to detect?" That reframes a non-significant
    p-value as "we couldn't detect an effect smaller than X%" rather than
    "there is no effect."

    Parameters
    ----------
    baseline_rate               : expected/observed retention rate in the
                                   control group (e.g. gate_30's Day-1 rate)
    minimum_detectable_effect   : the absolute difference in rate you care
                                   about detecting (e.g. 0.01 for a 1
                                   percentage-point change). Smaller = needs
                                   a much bigger sample.
    power                       : probability of detecting the effect if it's
                                   real (0.8 = 80% is the conventional
                                   minimum; 0.9 is more rigorous)
    alpha                       : significance threshold used in the eventual
                                   test
    alternative                 : should match whatever you plan to use in
                                   two_proportion_z_test

    Statistical note
    -----------------
    This uses Cohen's h (an arcsine-transformed effect size for proportions)
    via statsmodels' NormalIndPower, which is standard for two-proportion
    power analysis and behaves better than a raw-difference approximation
    across the full range of baseline rates.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect

    effect_size = proportion_effectsize(p1, p2)  # Cohen's h

    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=abs(effect_size),
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative,
    )

    return PowerAnalysisResult(
        required_n_per_group=int(np.ceil(n_per_group)),
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        power=power,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# 4. Convenience wrapper
# ---------------------------------------------------------------------------

def summarize_ab_test(
    count_a: int,
    nobs_a: int,
    count_b: int,
    nobs_b: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> None:
    """
    Pretty-print a full frequentist summary: z-test + both confidence
    intervals. Handy for quickly checking Day-1 and Day-7 retention in a
    notebook cell without re-typing the same three calls each time.
    """
    test = two_proportion_z_test(count_a, nobs_a, count_b, nobs_b, alternative, alpha)
    ci = confidence_interval_diff(count_a, nobs_a, count_b, nobs_b, 1 - alpha)

    print(f"Group A (e.g. gate_30) rate : {test.group_a_rate:.4%}  (n={nobs_a})")
    print(f"Group B (e.g. gate_40) rate : {test.group_b_rate:.4%}  (n={nobs_b})")
    print(f"Difference (B - A)          : {test.diff:+.4%}")
    print(f"z-statistic                 : {test.z_stat:.4f}")
    print(f"p-value                     : {test.p_value:.4f}")
    print(f"Significant at alpha={alpha}? : {test.significant}")
    print(f"Wald  {int((1-alpha)*100)}% CI on diff  : "
          f"({ci.wald_ci[0]:+.4%}, {ci.wald_ci[1]:+.4%})")
    print(f"Wilson {int((1-alpha)*100)}% CI on diff  : "
          f"({ci.wilson_ci[0]:+.4%}, {ci.wilson_ci[1]:+.4%})")


if __name__ == "__main__":
    # Small smoke test with made-up numbers so you can run
    # `python src/frequentist_tests.py` and confirm the module imports and
    # runs cleanly before wiring it into the real notebook.
    summarize_ab_test(count_a=3200, nobs_a=20000, count_b=3050, nobs_b=20000)