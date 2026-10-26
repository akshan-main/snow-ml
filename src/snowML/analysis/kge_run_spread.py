# pylint: disable=C0103
"""
How much does a basin's KGE move from one training run to the next? Finding 6
in docs/Evaluation_Findings.md.

Reads every MLflow metric dump in mlflow_data/run_id_data, takes each run's
per-basin test KGE (at the last logged step, and at the best step), and
reports the spread across runs for each Ex2 basin. Also checks whether Ex2's
published KGE equals any run's value exactly - it doesn't, so Ex2's runs are
not among these files.

The runs mix seeds and configurations, so this spread is an upper bound on
seed noise, not a measure of it.

Run standalone:
    python3 -m snowML.analysis.kge_run_spread \
        --run-dir mlflow_data/run_id_data \
        --ex2-metrics notebooks/Ex2_VarianceByHuc/single_all_metrics_w_snow_types_and_elev.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

KGE_METRIC_RE = r"^test_kge_\d{12}$"
WIDE_RANGE = 0.30


def load_kge_runs(run_dir):
    """Long frame of (run, huc, Step, Value) for every per-basin test_kge."""
    recs = []
    for f in glob.glob(os.path.join(run_dir, "metrics_from_*.csv")):
        d = pd.read_csv(f)
        d = d[d.Metric.str.match(KGE_METRIC_RE)]
        if d.empty:
            continue
        d = d.assign(huc=d.Metric.str[-12:],
                     run=os.path.basename(f)[len("metrics_from_"):-len(".csv")])
        recs.append(d[["run", "huc", "Step", "Value"]])
    return pd.concat(recs, ignore_index=True)


def per_basin_spread(kge_by_run):
    """IQR, range and median across runs, per basin, from a Series indexed by
    (run, huc)."""
    g = kge_by_run.groupby("huc")
    return pd.DataFrame({"iqr": g.quantile(.75) - g.quantile(.25),
                         "range": g.max() - g.min(),
                         "median": g.median()})


def exact_matches(kge_by_run, ex2_kge, tol=1e-6):
    """Number of Ex2 basins whose published KGE equals some run's value."""
    hits = 0
    for huc, val in ex2_kge.items():
        if huc in kge_by_run.index.get_level_values("huc"):
            hits += bool(np.any(np.abs(kge_by_run.xs(huc, level="huc") - val) < tol))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="mlflow_data/run_id_data")
    ap.add_argument("--ex2-metrics", required=True)
    args = ap.parse_args()

    d = load_kge_runs(args.run_dir)
    ex2 = pd.read_csv(args.ex2_metrics, dtype={"HUC_ID": str}).set_index("HUC_ID")["Test KGE"]
    print(f"runs: {d.run.nunique()} | basins: {d.huc.nunique()} | rows: {len(d)}\n")

    last = d.sort_values("Step").groupby(["run", "huc"]).Value.last()
    best = d.groupby(["run", "huc"]).Value.max()
    for name, s in [("last step", last), ("best step", best)]:
        per = per_basin_spread(s)
        per = per[per.index.isin(ex2.index)]
        gap = (ex2.reindex(per.index) - per["median"]).abs()
        print(f"[{name}] basins {len(per)} | median IQR {per.iqr.median():.3f}"
              f" | range > {WIDE_RANGE}: {(per.range > WIDE_RANGE).sum()}"
              f" | |Ex2 - run median| median {gap.median():.3f}")
        print(f"  Ex2 KGE equals some run's {name} value in "
              f"{exact_matches(s, ex2)}/{len(ex2)} basins")


# ----------------------------------------------------------------------------
# self-check

def _self_check():
    idx = pd.MultiIndex.from_tuples(
        [(r, h) for h in ("a", "b") for r in ("r1", "r2", "r3", "r4", "r5")],
        names=["run", "huc"])
    s = pd.Series([.5, .6, .7, .8, .9,      # basin a: wide spread
                   .7, .7, .7, .7, .7],     # basin b: identical every run
                  index=idx)
    per = per_basin_spread(s)
    assert np.isclose(per.loc["a", "range"], .4) and np.isclose(per.loc["a", "iqr"], .2)
    assert per.loc["b", "range"] == 0 and per.loc["b", "iqr"] == 0

    ex2 = pd.Series({"a": .6, "b": .65, "missing": .9})
    assert exact_matches(s, ex2) == 1, "only basin a's .6 appears in a run"

    print("kge_run_spread self-check passed (3 cases)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
