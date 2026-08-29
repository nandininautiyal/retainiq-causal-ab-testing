# RetainIQ: Causal A/B Testing on Mobile Game Retention

Does moving a progression gate from **level 30** to **level 40** actually change
player retention in *Cookie Cats*? This project answers that question using
both **frequentist** and **Bayesian** hypothesis testing, on the
[Cookie Cats A/B test dataset](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)
from Kaggle.

Rather than reporting a single p-value and calling it a day, this project is
built to show the full reasoning behind a causal claim: checking the
randomization held, comparing what frequentist and Bayesian methods each say,
and testing whether the effect is consistent across player segments — while
being explicit about where the data can and can't support strong conclusions.

## Project Structure

```
retainiq-causal-ab-testing/
├── data/                  # cookie_cats.csv (download from Kaggle, not committed)
├── notebooks/
│   └── eda.ipynb          # data loading, randomization sanity checks, retention rates
├── src/
│   ├── frequentist_tests.py   # two-proportion z-test, confidence intervals, power analysis
│   ├── bayesian_tests.py      # Beta-Binomial posterior, P(B>A), credible intervals, expected loss
│   └── segment_analysis.py    # re-runs both tests within engagement / gate-reached segments
├── reports/               # saved plots and result tables
├── requirements.txt
└── README.md
```

## The Question

*Cookie Cats* originally gated further play behind level 30. The A/B test
moved this gate to level 40 for one arm (`gate_40`) while the other arm kept
the original gate (`gate_30`). We ask: **did this change Day-1 and Day-7
retention?**

## Methodology

### 1. Exploratory Data Analysis (`notebooks/eda.ipynb`)
Before trusting any causal claim, we check that the ~50/50 `gate_30`/`gate_40`
split is consistent with random assignment (chi-square goodness-of-fit test),
and screen for data-quality issues (duplicate users, an extreme outlier user
with an implausibly high round count).

### 2. Frequentist testing (`src/frequentist_tests.py`)
A two-proportion z-test on Day-1 and Day-7 retention, reported alongside:
- A p-value against the null of "no difference between arms"
- Two confidence intervals on the difference (Wald and Wilson/Newcombe — the
  latter is more reliable near the extremes and is the recommended one)
- A retrospective power calculation: what's the smallest effect size we
  actually had good power to detect, given the sample sizes we have?

### 3. Bayesian testing (`src/bayesian_tests.py`)
A Beta-Binomial model (conjugate prior — exact closed-form posterior, no
MCMC needed for a binary outcome like this) reporting:
- P(gate_40's true retention rate > gate_30's)
- A 95% credible interval on the difference (interpretable directly as "95%
  probability the true difference is in this range," unlike a frequentist CI)
- Expected loss of choosing each variant — a decision-oriented metric: how
  much retention we'd expect to lose, on average, if we picked wrong

### 4. Segment analysis (`src/segment_analysis.py`)
Re-runs both approaches within:
- **Engagement tertiles** (casual / regular / hardcore, by rounds played)
- **Approximate "reached the gate" status** — since the dataset has no actual
  level data, this is a proxy (`sum_gamerounds >= gate threshold`), not a
  precise flag

This checks whether the aggregate effect (or lack thereof) holds up across
different kinds of players, or whether it's masking something that only
shows up in a subgroup.

## Key Limitations (read before trusting the causal story)

- **No pre-treatment covariates.** The dataset has no age/device/country/etc.,
  so the only real randomization check available is group-size balance —
  we can't verify balance on any other dimension.
- **Segments are built from a post-treatment variable.** `sum_gamerounds` is
  measured *after* the gate could have already affected behavior, so
  segment-level results describe how the effect varies by engagement level —
  they aren't a clean, pre-assigned subgroup analysis.
- **"Reached the gate" is approximate**, not ground truth — see the
  docstring in `segment_analysis.py` for why.

## Results

Run on the full dataset (n=90,189: 44,700 in `gate_30`, 45,489 in `gate_40`).

| Metric | gate_30 | gate_40 | Diff (B−A) | z-test p-value | Bayesian P(gate_40 > gate_30) |
|---|---|---|---|---|---|
| Day-1 retention | 44.82% | 44.23% | −0.59 pp | 0.074 (not significant) | 3.8% |
| Day-7 retention | 19.02% | 18.20% | −0.82 pp | 0.0016 (significant) | 0.08% |

**Bottom line:** moving the gate from level 30 to level 40 did **not** improve
retention — if anything, it modestly *hurt* Day-7 retention, and the
frequentist and Bayesian analyses agree closely on this. Day-1 retention
shows the same direction of effect but doesn't clear the conventional
significance threshold on its own. Given this, recommendation would be to
**keep the gate at level 30** rather than move it to 40.

A methodological side-note: the chi-square balance check in the EDA notebook
returns p≈0.0086 (i.e. "significant" at α=0.05) for the 44.96%/50.44% split.
With n=90k, even a sub-1-point deviation from an exact 50/50 split clears
that bar — this is a known quirk of this specific dataset, not evidence that
randomization was broken.

### Segment analysis findings

- **By engagement level** (casual/regular/hardcore): the Day-7 effect is
  concentrated in **regular and hardcore** players (both significant, gate_40
  worse); the **casual** segment shows no significant difference. This makes
  intuitive sense — casual players who barely play any rounds are unlikely to
  be affected by a gate at level 30 *or* 40.
- **By approximate "reached the gate" status**: this segment shows gate_40
  *outperforming* gate_30 in both sub-groups — the opposite direction of the
  overall effect. This is expected and is exactly the post-treatment
  conditioning bias flagged in `segment_analysis.py`'s docstring: since
  `sum_gamerounds` is measured after the gate could already have shaped
  behavior, conditioning on it changes who ends up in each bucket
  differently for the two arms. **Don't read this segment's numbers as
  contradicting the overall (more trustworthy) result** — it's a
  demonstration of why that caveat matters, not a real finding.

## How to Run

```powershell
# 1. Set up environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Download cookie_cats.csv from Kaggle into ./data/

# 3. Run the EDA + testing pipeline
jupyter nbconvert --to notebook --execute notebooks\eda.ipynb --output eda.ipynb

# 4. Run individual modules directly
python src\frequentist_tests.py
python src\bayesian_tests.py
python src\segment_analysis.py
```

## Tech Stack

Python · pandas · NumPy · SciPy · statsmodels · matplotlib/seaborn · Jupyter


