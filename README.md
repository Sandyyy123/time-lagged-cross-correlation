# Time-Lagged Cross-Correlation for Dyadic Tracking (HCI)

A compact, runnable pipeline for **continuous time-series cross-correlation**,
**lag/peak extraction**, and **mixed-effects group comparison** — the exact
method stack used in observer/target tracking studies.

It mirrors a real dyadic design: **28 observers x 6 targets = 168 dyadic
analyses**, sampled at **1 Hz (1000 ms)**, with time-lagged cross-correlation
computed over a bounded **−5 s … +10 s** window (**16 discrete lags, 16
coefficients per trial**).

## What it does

1. **Baseline zero-lag Pearson r** for every observer–target dyad.
2. **Time-lagged cross-correlation (TLCC)** — Pearson r at each 1000 ms shift
   across −5 s…+10 s (16 coefficients), `tlcc.py::tlcc`.
3. **Peak extraction** — peak r and the lag at which it occurs = *personalized
   temporal cognitive lag*, `tlcc.py::peak_of`.
4. **Group-level inference** — linear mixed model
   `peak_r ~ group + (1|observer) + (1|target)` (crossed random effects),
   plus an **APA-style results summary**.

## Run

```bash
pip install -r requirements.txt
python run_analysis.py
```

Output (seeded synthetic data, runs anywhere without the real dataset):

```
N = 168 dyadic analyses (28 observers x 6 targets).
Test    : M = 0.681, SD = 0.075, n = 84
Control : M = 0.392, SD = 0.084, n = 84
LMM  Test vs Control: b = 0.290, SE = 0.010, z = 29.73, p < .001
```

Per-dyad peaks are written to `dyadic_peaks.csv`.

## Files

| File | Purpose |
|------|---------|
| `tlcc.py` | TLCC core: `lagged_pearson`, `tlcc`, `peak_of` |
| `run_analysis.py` | Full 168-dyad pipeline + LMM + APA summary |

## Using real data

Replace `simulate_dyad()` with a CSV loader that returns two equal-length
1 Hz series (observer, target) per dyad; everything downstream is unchanged.
Swap the LMM for a mixed-design ANOVA (`pingouin.mixed_anova`) if the design
calls for it.

---
*Demo by Dr. Sandeep Grover — PhD (Data Science), clinical/statistical research.*
