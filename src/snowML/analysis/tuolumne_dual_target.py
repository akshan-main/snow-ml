# pylint: disable=C0103
"""
Re-scores the same Tuolumne predictions against two reference targets: UA SWE
(what both models were trained on, PRISM-derived) and ERA5-Land SWE
(independent of PRISM). The follow-up to finding 5 in
docs/Evaluation_Findings.md.

Nothing is retrained. Only the reference changes, over the water years where
ERA5 and the predictions overlap (WY2016-2021 with the files on hand). If a
model's advantage holds against both targets it is skill; if it only holds
against UA it may just be agreement with the target it was trained on.

Needs the ERA5-Land snow CSVs (snow_cover_era5_land/ERA5_Snow_{WY-1}.csv in
the snowml-model-ready bucket).

Run standalone:
    python3 -m snowML.analysis.tuolumne_dual_target \
        --pred-dir mlflow_data/predictions --era5-dir data/era5_tum
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from snowML.LSTM import LSTM_compare as cmp
from snowML.LSTM import LSTM_signatures as sig
from snowML.analysis.target_comparison import load_era5_year
from snowML.analysis.tuolumne_compare import load_predictions

TARGETS = {"ua": "mean_swe", "era5": "swe_era5"}
SIGS = ["peak_swe_bias_abs", "peak_doy_err_abs", "meltout_doy_err_abs"]


def dual_target_table(pred_dir, era5_dir):
    """One row per (model, unit) with each signature scored against both
    targets, columns prefixed ua_ and era5_."""
    era5 = pd.concat([load_era5_year(f) for f in
                      sorted(glob.glob(os.path.join(era5_dir, "ERA5_Snow_*.csv")))],
                     ignore_index=True)
    water_years = sorted(set(sig.water_year(era5.day)))
    rows = []
    for huc, kind, p in load_predictions(pred_dir):
        j = p.merge(era5.loc[era5.huc_id == huc, ["day", "swe_era5"]], on="day")
        j = j[np.isin(sig.water_year(j.day), water_years)]
        if len(j) < 300:
            continue
        r = {"model": kind, "unit": huc}
        for tag, col in TARGETS.items():
            s = sig.signatures_by_water_year(j, obs_col=col, sim_col="mean_swe_predict")
            r.update(sig.summarize(s, prefix=tag))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", default="mlflow_data/predictions")
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--out", default=None, help="optional CSV path for the table")
    args = ap.parse_args()

    df = dual_target_table(args.pred_dir, args.era5_dir)
    if args.out:
        df.to_csv(args.out, index=False)
    print(f"basins x models: {len(df)}  {df.model.value_counts().to_dict()}")

    for tag in TARGETS:
        metrics = [f"{tag}_{s}" for s in SIGS]
        _, pw = cmp.compare_models(df, metrics,
                                   extra_polarity={m: "low" for m in metrics})
        print(f"\n=== {tag} as reference: local vs multi ===")
        print(pw[["metric", "median_a", "median_b", "n_pairs", "p_adj",
                  "significant", "winner"]].round(4).to_string(index=False))

    print("\nbasins whose melt-out error gets worse when the reference is ERA5:")
    for kind, g in df.groupby("model"):
        worse = (g.era5_meltout_doy_err_abs > g.ua_meltout_doy_err_abs).sum()
        print(f"  {kind:6s} {worse}/{len(g)}")


if __name__ == "__main__":
    main()
