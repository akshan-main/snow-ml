# pylint: disable=C0103
"""
Locally trained vs multi-HUC (PNW-trained) model on the 22 Tuolumne HUC12s,
scored against UA SWE. Findings 4 and 5 in docs/Evaluation_Findings.md.

The Tuolumne basins are outside every multi-HUC training split, so the
multi-HUC predictions there are an out-of-region test. Both models are scored
over the water years both prediction sets cover (WY2011-2022), with KGE, MSE
and the per-water-year signatures from LSTM_signatures, then compared with the
paired chain in LSTM_compare.

Also reports how often a basin's yearly errors share the sign of its median
error (finding 4): near 1.0 is a consistent bias, near 0.5 means errors
alternate and a signed median will look falsely unbiased.

Caveat: the 22 basins sit in one HUC08 and are not independent, so the
p-values are optimistic.

Run standalone:
    python3 -m snowML.analysis.tuolumne_compare --pred-dir mlflow_data/predictions
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from snowML.LSTM import LSTM_compare as cmp
from snowML.LSTM import LSTM_signatures as sig

COMMON_WY = list(range(2011, 2023))
METRICS = ["test_kge", "test_mse", "test_peak_swe_bias", "test_peak_swe_bias_abs",
           "test_peak_doy_err_abs", "test_meltout_doy_err_abs",
           "test_melt_rate_bias_abs"]


def kge(obs, sim):
    """Kling-Gupta efficiency. Reproduces the stored test_kge values in
    mlflow_data/predictions/metric_dict_*.json to 1e-7."""
    obs, sim = np.ravel(obs), np.ravel(sim)
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(sim) / np.std(obs) - 1) ** 2
                       + (np.mean(sim) / np.mean(obs) - 1) ** 2)


def load_predictions(pred_dir):
    """Yield (huc, kind, frame) for every predictions_*.csv; kind is
    'local' or 'multi'."""
    for f in sorted(glob.glob(os.path.join(pred_dir, "predictions_*.csv"))):
        base = os.path.basename(f)
        huc = base.replace("predictions_", "").replace("_multi", "").replace(".csv", "")
        kind = "multi" if "_multi" in base else "local"
        yield huc, kind, pd.read_csv(f, parse_dates=["day"])


def score_basins(pred_dir, water_years=COMMON_WY):
    """Per-basin scores and per-water-year signatures for both models.

    Returns (scores, per_year): scores has one row per (model, unit) with
    KGE, MSE and the summarized signatures; per_year has one row per
    (kind, huc, water year).
    """
    rows, per = [], []
    for huc, kind, d in load_predictions(pred_dir):
        d = d[np.isin(sig.water_year(d.day), water_years)]
        s = sig.signatures_by_water_year(d)
        per.append(s.assign(huc=huc, kind=kind).reset_index())
        y, p = d.mean_swe.to_numpy(), d.mean_swe_predict.to_numpy()
        r = {"model": kind, "unit": huc, "test_kge": kge(y, p),
             "test_mse": float(np.mean((y - p) ** 2))}
        r.update(sig.summarize(s))
        rows.append(r)
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True)


def sign_consistency(per_year, col, min_years=4):
    """Median, across basins, of the fraction of water years whose error has
    the same sign as that basin's median error. One value per model kind."""
    out = {}
    for kind, g in per_year.groupby("kind"):
        fracs = []
        for _, b in g.groupby("huc"):
            v = b[col].dropna().to_numpy()
            if len(v) >= min_years:
                fracs.append(np.mean(np.sign(v) == np.sign(np.median(v))))
        out[kind] = float(np.median(fracs)) if fracs else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", default="mlflow_data/predictions")
    ap.add_argument("--plot", default=None, help="optional path for the comparison matrix PNG")
    args = ap.parse_args()

    scores, per_year = score_basins(args.pred_dir)
    print(f"water years blanked by min_peak_swe: {int(per_year.below_min_peak.sum())}"
          f" of {len(per_year)}\n")

    _, pw = cmp.compare_models(scores, METRICS)
    print(pw[["metric", "median_a", "median_b", "n_pairs", "p_adj", "significant",
              "winner"]].round(4).to_string(index=False))

    print("\nsign consistency (1.0 = consistent bias, ~0.5 = alternating):")
    for col in ["peak_swe_bias", "peak_doy_err", "meltout_doy_err"]:
        print(f"  {col:18s} {sign_consistency(per_year, col)}")

    print("\nsigned vs absolute median peak SWE bias (%):")
    for kind, g in per_year.groupby("kind"):
        b = g.groupby("huc").peak_swe_bias
        print(f"  {kind:6s} signed {b.median().median():7.2f}"
              f"   abs {b.apply(lambda s: s.abs().median()).median():7.2f}")

    if args.plot:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use("Agg")
        cmp.plot_comparison_matrix(
            pw, METRICS, path=args.plot,
            title="Locally trained vs PNW-trained LSTM | 22 Tuolumne HUC12s, WY2011-2022\n"
                  "Wilcoxon signed-rank + Bonferroni (basins nested in one HUC08: not independent)")
        print(f"\nmatrix saved to {args.plot}")


# ----------------------------------------------------------------------------
# self-check

def _self_check():
    rng = np.random.default_rng(0)
    obs = rng.gamma(2.0, 0.2, 500)
    assert abs(kge(obs, obs) - 1) < 1e-12, "perfect simulation must score 1"
    # doubling the simulation doubles both the mean and the spread: r stays 1,
    # each ratio term is off by 1, so KGE = 1 - sqrt(2)
    assert abs(kge(obs, 2 * obs) - (1 - np.sqrt(2))) < 1e-12

    per_year = pd.DataFrame({
        "kind": ["local"] * 6 + ["multi"] * 6,
        "huc": ["a"] * 6 + ["b"] * 6,
        "e": [5, 6, 4, 7, 5, 6,            # always high: consistent
              10, -10, 10, -10, 10, -9]})  # alternating
    c = sign_consistency(per_year, "e")
    assert c["local"] == 1.0, c
    assert 0.4 <= c["multi"] <= 0.6, c

    print("tuolumne_compare self-check passed (3 cases)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
