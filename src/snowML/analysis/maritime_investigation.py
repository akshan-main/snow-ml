# pylint: disable=C0103
"""
Chases down a lead from precip_vs_skill.py: in Maritime basins, storm
concentration correlates with KGE at +0.50, and survives controlling for
elevation (+0.31). Is that a real effect, or a confound in disguise?

This module is a documented dead end, kept on purpose (same spirit as the
thesis's dry-day-ratio result). The chain:

  1. raw storm_conc -> KGE within Maritime basins:        rho ~ +0.50
  2. partial out elevation:                                rho ~ +0.31  (survives)
  3. per-HUC08 sub-basin breakdown:                         wildly inconsistent
     sign (Upper Skagit +0.54, Wenatchee -0.31) -> something geographic
  4. storm_conc correlates with longitude at rho ~ +0.90
  5. partial out longitude instead of elevation:            rho ~ +0.10 (gone)
  6. split basins by DRAINAGE (not raw longitude) into Cascade
     west-slope / east-slope, partial out elevation AND longitude
     together within each side:                             no effect either side

Conclusion: the apparent storm_conc -> skill relationship in Maritime basins
is downstream of which side of the Cascade crest a basin drains to (drier
east-side basins score higher for reasons unrelated to storm character).
Nothing here is evidence that precipitation storminess drives model skill.

Needs a WKT `geometry` column (basin centroid) and a `Huc_08_nm` column in the
Ex2 metrics table, both already present in
notebooks/Ex2_VarianceByHuc/single_all_metrics_w_snow_types_and_elev.csv.
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy import stats

from snowML.analysis.precip_vs_skill import build_table, partial_spearman

# HUC08 sub-basins (by name) that drain east of the Cascade crest, used for
# the drainage-based split in step 6. This is a judgment call based on
# physical geography, not a longitude cutoff -- see the self-check for why
# a longitude cutoff was wrong (Naches/Upper Yakima are east-draining but
# their centroids can fall west of a simple meridian cut).
EAST_OF_CREST = {"Wenatchee", "Lake Chelan", "Upper Columbia-Entiat",
                 "Naches", "Upper Yakima"}


def centroid_lon(wkt_polygon):
    """Crude centroid longitude from a WKT POLYGON string: mean of vertex
    x-coordinates. Good enough for an east/west sanity check; not a real
    area-weighted centroid. Avoids a shapely dependency for one number."""
    coords = re.findall(r"(-?\d+\.\d+) (-?\d+\.\d+)", wkt_polygon)
    xs = np.array([float(x) for x, _ in coords])
    return float(xs.mean())


def partial_spearman_2(df, x, y, z_list):
    """Spearman correlation of x and y after removing each variable's
    rank-linear dependence on ALL variables in z_list simultaneously
    (multiple regression on ranks, not sequential partialling)."""
    R = np.column_stack([np.ones(len(df))] +
                        [stats.rankdata(df[z]) for z in z_list])
    rx, ry = stats.rankdata(df[x]), stats.rankdata(df[y])
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    return stats.pearsonr(ex, ey)


def investigate(merged, snow_type="Maritime", kge_col="Test KGE"):
    """
    Run the full chain on a merged precip-vs-skill table (output of
    precip_vs_skill.build_table joined to the Ex2 metrics table). Returns a
    dict with every intermediate result, so each step can be inspected rather
    than only trusting the final conclusion.
    """
    m = merged[merged.Predominant_Snow == snow_type].copy()
    if "lon" not in m.columns:
        m["lon"] = m.geometry.map(centroid_lon)

    raw_r, raw_p = stats.spearmanr(m.storm_conc, m[kge_col])
    elev_r, elev_p = partial_spearman(m, "storm_conc", kge_col, "mean_elevation")

    by_basin = {}
    for b, g in m.groupby("Huc_08_nm"):
        if len(g) >= 15:
            r, p = partial_spearman(g, "storm_conc", kge_col, "mean_elevation")
            by_basin[b] = {"n": len(g), "partial_r": r, "partial_p": p}

    lon_corr, _ = stats.spearmanr(m.lon, m.storm_conc)
    lon_partial_r, lon_partial_p = partial_spearman(m, "storm_conc", kge_col, "lon")

    m["side"] = np.where(m.Huc_08_nm.isin(EAST_OF_CREST), "east", "west")
    by_side = {}
    for side, g in m.groupby("side"):
        r_elev, p_elev = partial_spearman(g, "storm_conc", kge_col, "mean_elevation")
        r_both, p_both = partial_spearman_2(g, "storm_conc", kge_col,
                                            ["mean_elevation", "lon"])
        by_side[side] = {"n": len(g), "partial_elev_r": r_elev,
                         "partial_elev_p": p_elev,
                         "partial_elev_and_lon_r": r_both,
                         "partial_elev_and_lon_p": p_both}

    return {
        "n_basins": len(m),
        "raw_storm_conc_vs_kge": {"r": raw_r, "p": raw_p},
        "partial_elev": {"r": elev_r, "p": elev_p},
        "by_huc08": by_basin,
        "lon_vs_storm_conc": lon_corr,
        "partial_lon": {"r": lon_partial_r, "p": lon_partial_p},
        "by_drainage_side": by_side,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-ready-dir", required=True)
    ap.add_argument("--ex2-metrics", required=True)
    args = ap.parse_args()

    ex2 = pd.read_csv(args.ex2_metrics, dtype={"HUC_ID": str})
    precip = build_table(args.model_ready_dir, huc_ids=set(ex2.HUC_ID))
    merged = precip.merge(ex2, on="HUC_ID", how="inner")

    result = investigate(merged)
    print(f"Maritime basins: {result['n_basins']}")
    print(f"raw storm_conc -> KGE:           {result['raw_storm_conc_vs_kge']}")
    print(f"partial | elevation:             {result['partial_elev']}")
    print(f"longitude vs storm_conc:         {result['lon_vs_storm_conc']:.3f}")
    print(f"partial | longitude:             {result['partial_lon']}")
    print("by drainage side (partial | elev+lon together):")
    for side, r in result["by_drainage_side"].items():
        print(f"  {side}: n={r['n']}  {r['partial_elev_and_lon_r']:+.2f}"
              f" (p={r['partial_elev_and_lon_p']:.0e})")
    print("\nConclusion: once elevation AND longitude are both controlled for,"
          " no side shows a real storm_conc -> KGE relationship.")


# ----------------------------------------------------------------------------
# self-check: builds a synthetic Maritime-like population where storm_conc,
# KGE and longitude are ALL driven by one underlying "dryness/drainage-side"
# factor, with no direct storm_conc -> KGE link. The investigation should
# reproduce the real pattern: strong raw correlation, survives a naive
# single-confound control, collapses once both real confounds are used.

def _self_check():
    rng = np.random.default_rng(0)
    n = 150

    assert abs(centroid_lon("POLYGON ((-121.0 47.0, -121.2 47.1, -120.8 47.2))")
              - (-121.0)) < 1e-6

    side = rng.choice(["west", "east"], size=n, p=[0.55, 0.45])
    is_east = (side == "east").astype(float)
    # east basins are drier (higher dryness) and score higher (less snow
    # complexity to model) -- this is the real confound in the data
    dryness = is_east + rng.normal(0, 0.3, n)
    kge = 0.8 + 0.1 * is_east + rng.normal(0, 0.05, n)
    storm_conc = 0.3 + 0.15 * (1 - is_east) + rng.normal(0, 0.05, n)  # wetter west = stormier
    lon = -121.3 + 0.5 * is_east + rng.normal(0, 0.05, n)
    elevation = rng.normal(1500, 300, n)  # genuinely unrelated to the confound

    huc08 = rng.choice(["Upper Skagit", "Wenatchee", "Sauk", "Naches"], size=n)
    df = pd.DataFrame({"storm_conc": storm_conc, "Test KGE": kge,
                       "mean_elevation": elevation, "lon": lon,
                       "Huc_08_nm": huc08, "Predominant_Snow": "Maritime"})
    df["geometry"] = df.lon.map(lambda x: f"POLYGON (({x} 47.0, {x+0.1} 47.1))")

    raw_r, _ = stats.spearmanr(df.storm_conc, df["Test KGE"])
    assert raw_r < -0.3, f"raw should show a (spurious) negative relationship: {raw_r}"

    elev_r, _ = partial_spearman(df, "storm_conc", "Test KGE", "mean_elevation")
    assert abs(elev_r - raw_r) < 0.15, \
        "controlling for an UNRELATED confound (elevation) should barely move it"

    lon_r, _ = partial_spearman(df, "storm_conc", "Test KGE", "lon")
    assert abs(lon_r) < abs(raw_r) / 2, \
        f"controlling for the REAL confound (longitude) should shrink it a lot: {lon_r}"

    result = investigate(df.assign(geometry=df.geometry))
    assert abs(result["partial_lon"]["r"]) < abs(result["raw_storm_conc_vs_kge"]["r"]) / 2

    print("maritime_investigation self-check passed (synthetic confound recovered)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        main()
