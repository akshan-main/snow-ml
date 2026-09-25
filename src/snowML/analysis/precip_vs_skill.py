# pylint: disable=C0103
"""
Does precipitation character explain model skill (KGE), or is skill really
explained by where the basin sits (elevation, temperature)?

For each basin, computes five precipitation-character statistics from its
model-ready daily series, then correlates each against the basin's test KGE,
both raw and after partialling out a confound (elevation or temperature) via
Spearman rank residuals.

Precipitation statistics, computed from the mean_pr column:
    pr_annual        mean daily precip * 365.25
    dry_frac         fraction of days with precip < DRY_THRESHOLD
    dry_spell_mean   mean length of consecutive dry-day runs
    pr_p95           95th percentile of daily precip
    storm_conc       share of total precip falling on the wettest 5% of days

Run standalone:
    python3 -m snowML.analysis.precip_vs_skill \
        --model-ready-dir /path/to/downloaded/model_ready_csvs \
        --ex2-metrics notebooks/Ex2_VarianceByHuc/single_all_metrics_w_snow_types_and_elev.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

DRY_THRESHOLD = 0.1  # mm/day, matches the thesis's dry-day definition
STORM_TOP_FRAC = 0.05


def dry_spell_lengths(pr, threshold=DRY_THRESHOLD):
    """Lengths of consecutive dry-day runs, as an array of run lengths."""
    dry = (pr < threshold).astype(int)
    padded = np.r_[0, dry, 0]
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return ends - starts


def storm_concentration(pr, top_frac=STORM_TOP_FRAC):
    """Share of total precipitation falling on the wettest top_frac of days."""
    total = pr.sum()
    if total <= 0:
        return np.nan
    k = max(1, int(top_frac * len(pr)))
    top = np.sort(pr)[::-1][:k]
    return top.sum() / total


def precip_signature(df, pr_col="mean_pr"):
    """One row of precipitation-character statistics for one basin's series."""
    pr = df[pr_col].to_numpy(dtype=float)
    spells = dry_spell_lengths(pr)
    return {
        "pr_annual": float(pr.mean() * 365.25),
        "dry_frac": float((pr < DRY_THRESHOLD).mean()),
        "dry_spell_mean": float(spells.mean()) if spells.size else np.nan,
        "pr_p95": float(np.percentile(pr, 95)),
        "storm_conc": float(storm_concentration(pr)),
    }


def build_table(model_ready_dir, huc_ids=None):
    """Read model-ready CSVs and compute precip_signature for each basin.

    Args:
        model_ready_dir: directory of model_ready_huc*.csv files (already
            downloaded; this module does no S3 access itself).
        huc_ids: optional list restricting which basins to read.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(model_ready_dir, "model_ready_huc*.csv"))):
        base = os.path.basename(f)
        if "_ucla" in base:
            continue
        huc = base.replace("model_ready_huc", "").replace(".csv", "")
        if huc_ids is not None and huc not in huc_ids:
            continue
        df = pd.read_csv(f)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # stray index col
        row = {"HUC_ID": huc}
        row.update(precip_signature(df))
        rows.append(row)
    return pd.DataFrame(rows)


def partial_spearman(df, x, y, z):
    """
    Spearman correlation of x and y after removing each variable's rank-linear
    dependence on z (a first-order partial correlation on ranks).

    Returns (r, p). Matches the closed-form partial-correlation formula
    rxy.z = (rxy - rxz*ryz) / sqrt((1-rxz^2)(1-ryz^2)) to within floating-point
    error; see the self-check.
    """
    rx, ry, rz = (stats.rankdata(df[c]) for c in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def correlate_with_kge(table, kge_col="Test KGE", confound="mean_elevation"):
    """Raw and partial (controlling for confound) Spearman rho of each
    precipitation statistic against kge_col."""
    stat_cols = ["pr_annual", "dry_frac", "dry_spell_mean", "pr_p95", "storm_conc"]
    out = []
    for col in stat_cols:
        raw_r, raw_p = stats.spearmanr(table[col], table[kge_col])
        par_r, par_p = partial_spearman(table, col, kge_col, confound)
        out.append({"stat": col, "raw_rho": raw_r, "raw_p": raw_p,
                    "partial_rho": par_r, "partial_p": par_p,
                    "confound": confound})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-ready-dir", required=True)
    ap.add_argument("--ex2-metrics", required=True)
    ap.add_argument("--confound", default="mean_elevation",
                    help="column in the Ex2 metrics CSV to partial out")
    args = ap.parse_args()

    ex2 = pd.read_csv(args.ex2_metrics, dtype={"HUC_ID": str})
    table = build_table(args.model_ready_dir, huc_ids=set(ex2.HUC_ID))
    merged = table.merge(ex2, on="HUC_ID", how="inner")
    print(f"basins with both precip data and Ex2 metrics: {len(merged)}")

    result = correlate_with_kge(merged, confound=args.confound)
    print(result.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# self-check: synthetic data where the "true" relationship is known, so the
# correlation and partial-correlation machinery is verified independently of
# any real basin data.

def _self_check():
    rng = np.random.default_rng(0)
    n = 40

    # 1. dry_spell_lengths on a known run pattern
    pr = np.array([5, 0, 0, 0, 7, 0, 0, 3, 0, 0, 0, 0, 9.0])
    spells = dry_spell_lengths(pr)
    assert list(spells) == [3, 2, 4], list(spells)

    # 2. storm_concentration: one dominant day should carry most of the share
    pr2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 50.0])
    conc = storm_concentration(pr2, top_frac=0.05)  # top 1 of 20 days
    assert abs(conc - 50 / 69) < 1e-9, conc

    # 3. partial correlation matches the closed-form formula
    elev = rng.normal(0, 1, n)
    confound_driven = elev + rng.normal(0, 0.3, n)   # correlates with elev
    kge = elev + rng.normal(0, 0.3, n)                # also correlates with elev
    df = pd.DataFrame({"x": confound_driven, "y": kge, "z": elev})
    r_resid, _ = partial_spearman(df, "x", "y", "z")
    rxy, rxz, ryz = (stats.spearmanr(df[a], df[b])[0]
                     for a, b in [("x", "y"), ("x", "z"), ("y", "z")])
    r_formula = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    assert abs(r_resid - r_formula) < 0.02, (r_resid, r_formula)

    # 4. a relationship that is PURELY driven by a confound should vanish once
    # that confound is partialled out (this is the Maritime/longitude case)
    n_big = 400  # a larger sample keeps the residual noise floor well below 0.3
    z = rng.normal(0, 1, n_big)
    x_pure_confound = z + rng.normal(0, 0.05, n_big)
    y_pure_confound = z + rng.normal(0, 0.05, n_big)
    df2 = pd.DataFrame({"x": x_pure_confound, "y": y_pure_confound, "z": z})
    raw, _ = stats.spearmanr(df2.x, df2.y)
    par, _ = partial_spearman(df2, "x", "y", "z")
    assert raw > 0.9, f"raw should be strong: {raw}"
    assert abs(par) < 0.3, f"partial should collapse toward zero: {par}"

    # 5. build_table / precip_signature round-trip on a synthetic basin file
    # storms on 5% of days (the same window storm_conc inspects) should then
    # account for most of the total
    synth_pr = np.full(365, 0.05)
    storm_days = np.arange(0, 365, 20)          # ~5% of days
    synth_pr[storm_days] = 20.0
    sig = precip_signature(pd.DataFrame({"mean_pr": synth_pr}))
    assert sig["dry_frac"] > 0.8, sig["dry_frac"]
    assert sig["storm_conc"] > 0.9, sig["storm_conc"]

    print("precip_vs_skill self-check passed (5 cases)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
