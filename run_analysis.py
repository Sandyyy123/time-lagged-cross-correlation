"""
End-to-end mirror of the HCI dyadic tracking design:

    28 Observers  (14 Test group, 14 Control group)
    x 6 Targets
    = 168 dyadic time-series analyses

For each dyad:
    1. baseline zero-lag Pearson r
    2. full TLCC over -5s ... +10s at 1 Hz  (16 coefficients)
    3. peak r + peak lag (personalized temporal cognitive lag)

Then group-level inference:
    LMM  peak_r ~ group  with random intercepts for observer and target
    (crossed random effects; falls back to observer-only if singular).

Synthetic data ONLY (seeded) so the pipeline runs anywhere without the
client's real dataset. Swap `simulate_dyad` for a CSV loader on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tlcc import tlcc, peak_of

RNG = np.random.default_rng(7)

N_PER_GROUP = 14
N_TARGETS = 6
SERIES_LEN = 300           # 300 s trial at 1 Hz
FS = 1.0


def simulate_dyad(true_lag_s: int, coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """Target = smooth random walk; Observer = lagged, noisy copy.

    coupling in [0,1] sets tracking fidelity; true_lag_s injects a ground-truth
    temporal lag we should recover as the TLCC peak.
    """
    innov = RNG.normal(0, 1, SERIES_LEN + 20)
    target = np.convolve(innov, np.ones(8) / 8, mode="same")[:SERIES_LEN]
    lag = int(true_lag_s * FS)
    shifted = np.roll(target, lag)
    noise = RNG.normal(0, 1, SERIES_LEN)
    observer = coupling * shifted + (1 - coupling) * noise
    return observer, target


def build_dataset() -> pd.DataFrame:
    records = []
    for group, base_coupling, base_lag in [
        ("Test", 0.72, 2),        # test group: tighter tracking, ~2 s lag
        ("Control", 0.55, 4),     # control: looser tracking, ~4 s lag
    ]:
        for obs_i in range(N_PER_GROUP):
            obs_id = f"{group[:1]}{obs_i:02d}"
            coupling = np.clip(base_coupling + RNG.normal(0, 0.06), 0.1, 0.95)
            lag_bias = base_lag + RNG.integers(-1, 2)
            for tgt in range(1, N_TARGETS + 1):
                observer, target = simulate_dyad(lag_bias, coupling)
                frame = tlcc(observer, target, fs_hz=FS, lo_s=-5, hi_s=10)
                assert len(frame) == 16, f"expected 16 lags, got {len(frame)}"
                pk = peak_of(frame)
                records.append(
                    {
                        "observer": obs_id,
                        "group": group,
                        "target": f"T{tgt}",
                        **pk,
                    }
                )
    return pd.DataFrame(records)


def fit_lmm(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    df = df.copy()
    df["group"] = pd.Categorical(df["group"], categories=["Control", "Test"])
    # Crossed random effects: observer (groups) + target (variance component)
    try:
        vc = {"target": "0 + C(target)"}
        model = smf.mixedlm(
            "peak_r ~ group", df, groups=df["observer"], vc_formula=vc
        )
        res = model.fit(reml=True, method="lbfgs")
    except Exception:
        model = smf.mixedlm("peak_r ~ group", df, groups=df["observer"])
        res = model.fit(reml=True)
    return res


def apa_report(df: pd.DataFrame, res) -> str:
    g = df.groupby("group")["peak_r"].agg(["mean", "std", "count"])
    lag = df.groupby("group")["peak_lag_s"].agg(["mean", "std"])
    beta = res.fe_params.get("group[T.Test]", float("nan"))
    se = res.bse.get("group[T.Test]", float("nan"))
    z = res.tvalues.get("group[T.Test]", float("nan"))
    p = res.pvalues.get("group[T.Test]", float("nan"))
    lines = [
        "APA-STYLE RESULTS SUMMARY",
        "=" * 60,
        f"N = {len(df)} dyadic analyses "
        f"({df['observer'].nunique()} observers x {df['target'].nunique()} targets).",
        "",
        "Descriptives (peak cross-correlation r):",
        f"  Test    : M = {g.loc['Test','mean']:.3f}, SD = {g.loc['Test','std']:.3f}, n = {int(g.loc['Test','count'])}",
        f"  Control : M = {g.loc['Control','mean']:.3f}, SD = {g.loc['Control','std']:.3f}, n = {int(g.loc['Control','count'])}",
        "",
        "Temporal cognitive lag (peak lag, s):",
        f"  Test    : M = {lag.loc['Test','mean']:.2f}, SD = {lag.loc['Test','std']:.2f}",
        f"  Control : M = {lag.loc['Control','mean']:.2f}, SD = {lag.loc['Control','std']:.2f}",
        "",
        "Linear mixed model  peak_r ~ group + (1|observer) + (1|target):",
        f"  Test vs Control: b = {beta:.3f}, SE = {se:.3f}, z = {z:.2f}, p = {p:.4f}",
        "=" * 60,
    ]
    return "\n".join(lines)


def main():
    print("Building 168 dyadic TLCC analyses (28 observers x 6 targets)...")
    df = build_dataset()
    print(f"  -> {len(df)} dyads, each with 16 lag coefficients.\n")
    print(df.head(8).to_string(index=False))
    print("\nFitting linear mixed model...\n")
    res = fit_lmm(df)
    print(apa_report(df, res))
    df.to_csv("dyadic_peaks.csv", index=False)
    print("\nSaved per-dyad peaks -> dyadic_peaks.csv")


if __name__ == "__main__":
    main()
