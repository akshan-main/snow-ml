# pylint: disable=C0103
"""
UA SWE (the training target used throughout this repo, built on PRISM
precipitation) vs ERA5-Land SWE (a second SWE product, independent of PRISM).

They agree closely on shape (daily correlation ~0.97 median across basins) but
disagree on peak magnitude in a meaningful subset of basins -- and a basin's
direction of disagreement is stable from year to year, which is the key check:
random noise would not be reproducible across independent water years.

This matters because GRIDMET, this repo's only precipitation forcing, is also
PRISM-derived. If the model is swapped onto a different precipitation product
later, a change in skill could reflect either a real precipitation-quality
difference or simply reduced agreement with a PRISM-descended target. ERA5
SWE, being independent of PRISM, is the natural reference for separating
those two explanations -- see docs/Evaluation_Findings.md, finding 1.

Needs the model-ready UA series (mean_swe column) and the ERA5-Land snow CSVs
(snow_cover_era5_land/ERA5_Snow_{WY-1}.csv, where the filename year is the
water year minus one -- e.g. ERA5_Snow_2016.csv covers WY2017).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

PEAK_MIN_SWE = 0.01  # m; basins with a smaller UA peak are excluded as
                     # effectively snow-free, where a ratio is meaningless


def load_era5_year(path):
    """Load one ERA5_Snow_{YYYY}.csv file into {huc_id: Series(day -> swe)}."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"date": "day",
                            "snow_depth_water_equivalent": "swe_era5"})
    df["huc_id"] = df.huc_id.astype(str)
    return df


def compare_one_basin(ua_df, era5_df, ua_col="mean_swe"):
    """Daily correlation and peak-SWE ratio (ERA5/UA) for one basin-year.

    Returns None if there isn't enough overlapping data or the UA peak is too
    small to divide by meaningfully.
    """
    j = ua_df[["day", ua_col]].merge(era5_df[["day", "swe_era5"]], on="day")
    if len(j) < 300:
        return None
    ua, era5 = j[ua_col].to_numpy(), j.swe_era5.to_numpy()
    if ua.max() <= PEAK_MIN_SWE:
        return None
    return {"n_days": len(j), "corr": float(np.corrcoef(ua, era5)[0, 1]),
           "ua_peak": float(ua.max()), "era5_peak": float(era5.max()),
           "peak_ratio": float(era5.max() / ua.max())}


def compare_year(model_ready_dir, era5_csv, huc_ids=None):
    """Run compare_one_basin across every basin with a model-ready file."""
    era5 = load_era5_year(era5_csv)
    rows = []
    for f in sorted(glob.glob(os.path.join(model_ready_dir, "model_ready_huc*.csv"))):
        base = os.path.basename(f)
        if "_ucla" in base:
            continue
        huc = base.replace("model_ready_huc", "").replace(".csv", "")
        if huc_ids is not None and huc not in huc_ids:
            continue
        ua = pd.read_csv(f, parse_dates=["day"])
        ua = ua.loc[:, ~ua.columns.str.startswith("Unnamed")]
        e = era5[era5.huc_id == huc]
        if e.empty:
            continue
        result = compare_one_basin(ua, e)
        if result is not None:
            rows.append({"huc": huc, **result})
    return pd.DataFrame(rows)


def cross_year_stability(ratios_by_year):
    """
    Given {year_label: DataFrame with huc, peak_ratio}, checks whether a
    basin's peak_ratio is correlated across years (evidence the magnitude
    disagreement is a real, stable property of the basin, not yearly noise),
    and how many basins fall consistently on the same side of 2x / 0.5x.

    Args:
        ratios_by_year: dict of at least two {label: df[huc, peak_ratio]}.

    Returns a dict with the pairwise Spearman correlation(s) and consistent
    extreme-ratio basin counts for each pair of years supplied.
    """
    labels = list(ratios_by_year)
    out = {}
    for i in range(len(labels)):
        for k in range(i + 1, len(labels)):
            a, b = labels[i], labels[k]
            ja = ratios_by_year[a].set_index("huc").peak_ratio
            jb = ratios_by_year[b].set_index("huc").peak_ratio
            common = ja.index.intersection(jb.index)
            if len(common) < 5:
                continue
            x, y = ja.loc[common], jb.loc[common]
            rho, p = stats.spearmanr(x, y)
            out[f"{a}_vs_{b}"] = {
                "n_common_basins": len(common),
                "spearman_rho": rho, "p": p,
                "both_over_2x": int(((x > 2) & (y > 2)).sum()),
                "both_under_half": int(((x < 0.5) & (y < 0.5)).sum()),
            }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-ready-dir", required=True)
    ap.add_argument("--era5-csv", action="append", required=True,
                    help="one or more ERA5_Snow_{YYYY}.csv paths; repeat for"
                         " multiple water years to check cross-year stability")
    ap.add_argument("--huc-list", default=None,
                    help="optional file of one HUC id per line to restrict to")
    args = ap.parse_args()

    huc_ids = None
    if args.huc_list:
        huc_ids = {l.strip() for l in open(args.huc_list) if l.strip()}

    by_year = {}
    for path in args.era5_csv:
        label = os.path.basename(path).replace(".csv", "")
        t = compare_year(args.model_ready_dir, path, huc_ids=huc_ids)
        by_year[label] = t
        print(f"\n{label}: n={len(t)} basins")
        print(t[["corr", "peak_ratio"]].describe().loc[["25%", "50%", "75%"]]
             .round(3).to_string())
        print(f"  ERA5 peak > 2x UA: {(t.peak_ratio > 2).sum()}"
              f"  |  ERA5 peak < 0.5x UA: {(t.peak_ratio < 0.5).sum()}")

    if len(by_year) >= 2:
        print("\ncross-year stability:")
        for pair, r in cross_year_stability(by_year).items():
            print(f"  {pair}: {r}")


# ----------------------------------------------------------------------------
# self-check: synthetic UA/ERA5 series where each basin's bias is known by
# construction, so compare_one_basin and cross_year_stability are verified
# against a ground truth rather than only against each other.

def _self_check():
    days = pd.date_range("2020-10-01", periods=365, freq="D")
    doy = np.arange(1, 366)
    triangle = lambda peak: np.maximum(0.0, np.where(
        doy <= 180, peak * doy / 180, peak * (280 - doy) / 100))

    # basin A: ERA5 consistently overestimates peak by 2.2x
    ua_a = triangle(1.0)
    era5_a = triangle(2.2)
    r = compare_one_basin(pd.DataFrame({"day": days, "mean_swe": ua_a}),
                          pd.DataFrame({"day": days, "swe_era5": era5_a}))
    assert abs(r["peak_ratio"] - 2.2) < 1e-6, r["peak_ratio"]
    assert r["corr"] > 0.99, r["corr"]  # same shape, so correlation stays high

    # basin B: too little overlap should return None, not a bogus ratio
    short = compare_one_basin(
        pd.DataFrame({"day": days[:100], "mean_swe": ua_a[:100]}),
        pd.DataFrame({"day": days[:100], "swe_era5": era5_a[:100]}))
    assert short is None

    # basin C: a basin with negligible UA snow is excluded, not divided by ~0
    flat = compare_one_basin(
        pd.DataFrame({"day": days, "mean_swe": np.zeros(365)}),
        pd.DataFrame({"day": days, "swe_era5": triangle(0.5)}))
    assert flat is None

    # cross_year_stability: a basin that is ALWAYS biased high should be
    # distinguished from one with no consistent bias
    stable = pd.DataFrame({"huc": ["A", "B", "C", "D", "E"],
                          "peak_ratio": [2.5, 0.3, 1.0, 2.8, 0.2]})
    same_again = pd.DataFrame({"huc": ["A", "B", "C", "D", "E"],
                              "peak_ratio": [2.3, 0.35, 1.1, 3.0, 0.25]})
    result = cross_year_stability({"wy1": stable, "wy2": same_again})
    r = result["wy1_vs_wy2"]
    assert r["spearman_rho"] > 0.9, r["spearman_rho"]
    assert r["both_over_2x"] == 2, r["both_over_2x"]       # A and D
    assert r["both_under_half"] == 2, r["both_under_half"]  # B and E

    shuffled = pd.DataFrame({"huc": ["A", "B", "C", "D", "E"],
                            "peak_ratio": [0.3, 2.5, 2.8, 1.0, 0.2]})
    result2 = cross_year_stability({"wy1": stable, "wy2": shuffled})
    assert result2["wy1_vs_wy2"]["spearman_rho"] < 0.5, \
        "shuffled ratios should not show stability"

    print("target_comparison self-check passed (5 cases)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
