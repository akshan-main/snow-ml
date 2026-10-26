# pylint: disable=C0103
"""
Is the elevation -> KGE effect really an elevation effect? Finding 3 in
docs/Evaluation_Findings.md.

Ex2 found elevation explains about a third of KGE variance; Ex3 found it
stopped mattering once elevation was a feature. This checks whether elevation
is standing in for temperature: Spearman of elevation vs Ex2 test KGE, raw and
with each basin's mean air temperature partialled out, across all basins and
within each snow class.

Run standalone:
    python3 -m snowML.analysis.elevation_vs_temperature \
        --model-ready-dir data/model_ready \
        --ex2-metrics notebooks/Ex2_VarianceByHuc/single_all_metrics_w_snow_types_and_elev.csv
"""

import argparse
import glob
import os

import pandas as pd
from scipy import stats

from snowML.analysis.precip_vs_skill import partial_spearman

MIN_GROUP = 20  # snow classes with fewer basins are skipped


def mean_tair_table(model_ready_dir, huc_ids=None):
    """Mean daily air temperature over the full record, one row per basin."""
    rows = []
    for f in sorted(glob.glob(os.path.join(model_ready_dir, "model_ready_huc*.csv"))):
        base = os.path.basename(f)
        if "_ucla" in base:
            continue
        huc = base.replace("model_ready_huc", "").replace(".csv", "")
        if huc_ids is not None and huc not in huc_ids:
            continue
        rows.append({"HUC_ID": huc,
                     "tair_mean": pd.read_csv(f, usecols=["mean_tair"]).mean_tair.mean()})
    return pd.DataFrame(rows)


def elevation_vs_temperature(merged, kge_col="Test KGE"):
    """Raw and temperature-partialled Spearman of elevation vs KGE, for all
    basins and per snow class."""
    groups = [("ALL", merged)] + list(merged.groupby("Predominant_Snow"))
    out = []
    for name, g in groups:
        if len(g) < MIN_GROUP:
            continue
        raw_r, raw_p = stats.spearmanr(g.mean_elevation, g[kge_col])
        par_r, par_p = partial_spearman(g, "mean_elevation", kge_col, "tair_mean")
        out.append({"group": name, "n": len(g), "raw_rho": raw_r, "raw_p": raw_p,
                    "partial_rho_given_tair": par_r, "partial_p": par_p})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-ready-dir", required=True)
    ap.add_argument("--ex2-metrics", required=True)
    args = ap.parse_args()

    ex2 = pd.read_csv(args.ex2_metrics, dtype={"HUC_ID": str})
    tair = mean_tair_table(args.model_ready_dir, huc_ids=set(ex2.HUC_ID))
    merged = tair.merge(ex2, on="HUC_ID", how="inner")
    print(f"basins with both temperature and Ex2 metrics: {len(merged)}\n")
    print(elevation_vs_temperature(merged).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
