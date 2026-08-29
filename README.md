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

*(Fill in after running the notebook + modules on the real dataset —
see below for how.)*

| Metric | gate_30 | gate_40 | Diff | p-value | P(B>A) |
|---|---|---|---|---|---|
| Day-1 retention | — | — | — | — | — |
| Day-7 retention | — | — | — | — | — |

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

