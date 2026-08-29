

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PosteriorResult:
    alpha: float
    beta: float
    mean: float
    mode: float | None
    std: float


@dataclass
class BayesianComparisonResult:
    prob_b_better: float
    diff_mean: float
    credible_interval: tuple[float, float]
    credible_level: float
    expected_loss_choosing_a: float
    expected_loss_choosing_b: float


# ---------------------------------------------------------------------------
# 1. Posterior for a single group
# ---------------------------------------------------------------------------

def beta_binomial_posterior(
    successes: int,
    trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> PosteriorResult:
    """
    Compute the Beta posterior for a single group's true retention rate.

    Parameters
    ----------
    successes, trials : observed retained players / total players
    prior_alpha, prior_beta : Beta prior parameters.
        Default is Beta(1, 1) — the UNIFORM prior over [0, 1], i.e. "before
        seeing data, any retention rate from 0% to 100% is equally likely."
        This is a deliberately weak/uninformative choice so the posterior is
        driven almost entirely by the data (appropriate here since you have
        tens of thousands of observations per group and no strong prior
        belief about Cookie Cats retention specifically).

        A common alternative is Beta(2, 2) — still weak, but places a little
        more mass near 0.5 than at the extremes. With this sample size the
        choice barely matters; it would matter much more on a small segment
        (see segment_analysis.py) where the prior has more influence and is
        worth thinking about explicitly.

    Returns
    -------
    PosteriorResult with the updated (alpha, beta) and summary stats.
    """
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)

    dist = stats.beta(post_alpha, post_beta)
    mean = dist.mean()
    std = dist.std()

    # Mode is only well-defined for alpha, beta > 1
    mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else None

    return PosteriorResult(alpha=post_alpha, beta=post_beta, mean=mean, mode=mode, std=std)


# ---------------------------------------------------------------------------
# 2. Comparing two groups
# ---------------------------------------------------------------------------

def prob_b_better_than_a(
    posterior_a: PosteriorResult,
    posterior_b: PosteriorResult,
    n_samples: int = 200_000,
    random_state: int | None = 42,
) -> float:
    """
    Estimate P(true_rate_B > true_rate_A) via Monte Carlo sampling from the
    two posteriors.

    Note on method: there IS a closed-form expression for this probability
    when comparing two Beta distributions, but it involves a sum that's
    numerically fiddly for large alpha/beta (which you'll have, given tens
    of thousands of observations). Monte Carlo sampling is simpler, robust
    at any sample size, and 200k draws gives plenty of precision (~0.1%
    Monte Carlo error) for a portfolio project. Set `random_state` for
    reproducibility.
    """
    rng = np.random.default_rng(random_state)
    samples_a = stats.beta(posterior_a.alpha, posterior_a.beta).rvs(n_samples, random_state=rng)
    samples_b = stats.beta(posterior_b.alpha, posterior_b.beta).rvs(n_samples, random_state=rng)

    return float(np.mean(samples_b > samples_a))


def credible_interval(
    posterior_a: PosteriorResult,
    posterior_b: PosteriorResult,
    credible_level: float = 0.95,
    n_samples: int = 200_000,
    random_state: int | None = 42,
) -> tuple[float, float]:
    """
    Equal-tailed credible interval on (true_rate_B - true_rate_A), computed
    from the same Monte Carlo draws used above.

    Frequentist CI vs Bayesian credible interval — know the difference:
    - A 95% CONFIDENCE interval (frequentist) means: "if we repeated this
      experiment many times, 95% of such intervals would contain the true
      difference." It does NOT mean "95% probability the true value is in
      this specific interval."
    - A 95% CREDIBLE interval (Bayesian) means exactly what people often
      WRONGLY assume the frequentist CI means: "given the data and the
      prior, there's a 95% probability the true difference lies in this
      interval." This more intuitive interpretation is one of the main
      selling points of the Bayesian approach for a portfolio project —
      it's worth stating explicitly in your README/report.
    """
    rng = np.random.default_rng(random_state)
    samples_a = stats.beta(posterior_a.alpha, posterior_a.beta).rvs(n_samples, random_state=rng)
    samples_b = stats.beta(posterior_b.alpha, posterior_b.beta).rvs(n_samples, random_state=rng)
    diff_samples = samples_b - samples_a

    alpha = 1 - credible_level
    lo, hi = np.quantile(diff_samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def expected_loss(
    posterior_a: PosteriorResult,
    posterior_b: PosteriorResult,
    n_samples: int = 200_000,
    random_state: int | None = 42,
) -> tuple[float, float]:
    """
    Compute the expected loss of choosing A when B is actually better, and
    vice versa. This answers a more decision-relevant question than a bare
    p-value: "if I roll out gate_40 and I'm wrong, how much retention do I
    expect to lose on average?"

    expected_loss_choosing_a = E[max(rate_B - rate_A, 0)]
    expected_loss_choosing_b = E[max(rate_A - rate_B, 0)]

    This is the standard "expected loss" criterion used in Bayesian A/B
    testing frameworks (e.g. as popularized by VWO/Optimizely-style
    Bayesian testing tools) to decide when it's safe to stop a test: pick a
    loss threshold you're willing to tolerate (e.g. 0.1 percentage points)
    and stop once the expected loss of your preferred choice drops below it.
    """
    rng = np.random.default_rng(random_state)
    samples_a = stats.beta(posterior_a.alpha, posterior_a.beta).rvs(n_samples, random_state=rng)
    samples_b = stats.beta(posterior_b.alpha, posterior_b.beta).rvs(n_samples, random_state=rng)

    loss_choosing_a = np.mean(np.maximum(samples_b - samples_a, 0))
    loss_choosing_b = np.mean(np.maximum(samples_a - samples_b, 0))

    return float(loss_choosing_a), float(loss_choosing_b)


# ---------------------------------------------------------------------------
# 3. Convenience wrapper
# ---------------------------------------------------------------------------

def summarize_bayesian_ab_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
) -> BayesianComparisonResult:
    """
    Run the full Bayesian comparison and print a readable summary. Mirrors
    `summarize_ab_test()` in frequentist_tests.py so you can run both side
    by side on the same counts and compare conclusions directly.
    """
    post_a = beta_binomial_posterior(successes_a, trials_a, prior_alpha, prior_beta)
    post_b = beta_binomial_posterior(successes_b, trials_b, prior_alpha, prior_beta)

    p_b_better = prob_b_better_than_a(post_a, post_b)
    ci = credible_interval(post_a, post_b, credible_level)
    loss_a, loss_b = expected_loss(post_a, post_b)

    result = BayesianComparisonResult(
        prob_b_better=p_b_better,
        diff_mean=post_b.mean - post_a.mean,
        credible_interval=ci,
        credible_level=credible_level,
        expected_loss_choosing_a=loss_a,
        expected_loss_choosing_b=loss_b,
    )

    print(f"Posterior A: Beta({post_a.alpha:.1f}, {post_a.beta:.1f}) "
          f"-> mean={post_a.mean:.4%}, std={post_a.std:.4%}")
    print(f"Posterior B: Beta({post_b.alpha:.1f}, {post_b.beta:.1f}) "
          f"-> mean={post_b.mean:.4%}, std={post_b.std:.4%}")
    print(f"P(B > A)                    : {p_b_better:.4%}")
    print(f"Mean difference (B - A)     : {result.diff_mean:+.4%}")
    print(f"{int(credible_level*100)}% credible interval on diff : "
          f"({ci[0]:+.4%}, {ci[1]:+.4%})")
    print(f"Expected loss if choosing A : {loss_a:.5%}")
    print(f"Expected loss if choosing B : {loss_b:.5%}")

    return result


if __name__ == "__main__":
    # Smoke test with the same made-up numbers used in frequentist_tests.py,
    # so you can compare the two approaches' conclusions directly.
    summarize_bayesian_ab_test(
        successes_a=3200, trials_a=20000,
        successes_b=3050, trials_b=20000,
    )