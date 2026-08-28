# pylint: disable=C0103
"""
Snow signatures: per-water-year diagnostics of *how* a SWE simulation is wrong,
as opposed to the aggregate error metrics in LSTM_metrics.

These are the SWE analogues of the hydrological signatures used for streamflow
(peak magnitude, peak timing, recession behavior). A snowpack has one
accumulation-melt cycle per water year, so every signature is computed per water
year and then aggregated across years.

Deliberately avoids log-ratio signatures such as the streamflow %BiasFLV.
Observed SWE is exactly zero for 30-60% of days in these basins, so any log ratio
diverges.

Signatures, per water year:
    peak_swe       maximum SWE (m)
    peak_doy       day-of-water-year of that maximum (Oct 1 = 1)
    meltout_doy    first day-of-water-year after the peak when the pack is gone
    melt_rate      peak_swe / (meltout_doy - peak_doy), in m/day

Reported as simulated-minus-observed errors, plus percent bias for magnitudes.

Robustness
----------
Simulated SWE can be noisy - an LSTM with a LeakyReLU output produces day-scale
spikes and small negative values, especially when applied outside its training
region. Two guards, applied identically to the observed and simulated series so
the comparison stays fair:

1. The peak is located on a centered rolling median, so an isolated spike cannot
   be mistaken for the seasonal maximum, and its magnitude and date are then
   read off the raw series within that neighbourhood so both stay exact.
2. Melt-out requires the pack to stay below the bare-ground threshold for
   PERSIST_DAYS, so a transient mid-melt dip does not end the season early.
"""

import numpy as np
import pandas as pd

# SWE at or below this (m) counts as bare ground. Model output passes through a
# LeakyReLU and so can be slightly negative; observed values are exactly 0.
SWE_THRESHOLD = 0.01
# Window (days) of the centered rolling median used to locate peaks.
SMOOTH_DAYS = 7
# Days the pack must stay below threshold before melt-out is declared.
PERSIST_DAYS = 14
# Observed peak SWE (m) below which percent-bias signatures are left undefined,
# because the denominator is too small for a percentage to mean anything.
MIN_PEAK_SWE = 0.10


def water_year(dates):
    """Water year for each date. WY2015 runs Oct 1 2014 - Sep 30 2015."""
    dates = pd.DatetimeIndex(dates)
    return np.where(dates.month >= 10, dates.year + 1, dates.year)


def day_of_water_year(dates):
    """Day-of-water-year, 1-based. Oct 1 = 1."""
    dates = pd.DatetimeIndex(dates)
    wy = water_year(dates)
    wy_start = pd.to_datetime({"year": wy - 1, "month": 10, "day": 1})
    return (dates - pd.DatetimeIndex(wy_start)).days + 1


def _smooth(swe, window):
    """Centered rolling median, edges filled with the raw values."""
    if window <= 1:
        return np.asarray(swe, dtype=float)
    s = pd.Series(np.asarray(swe, dtype=float))
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()


def _one_series(swe, doy, threshold, smooth_days, persist_days):
    """Signatures for one water year of one series.

    Returns NaNs when the year holds no meaningful snowpack, and a NaN melt-out
    when the pack never clears before the water year ends.
    """
    nan_result = {"peak_swe": np.nan, "peak_doy": np.nan,
                  "meltout_doy": np.nan, "melt_rate": np.nan}

    swe = np.asarray(swe, dtype=float)
    smoothed = _smooth(swe, smooth_days)

    # Locate the peak on the smoothed series so a spike cannot win, then take
    # the true maximum within that neighbourhood so the reported magnitude and
    # date are the real ones rather than smoothed-down values.
    # ponytail: a spike landing within smooth_days of the true peak still wins.
    # That is a few days of timing error, not a season-scale one; tighten only
    # if it shows up.
    approx_i = int(np.argmax(smoothed))
    lo = max(0, approx_i - smooth_days)
    hi = min(len(swe), approx_i + smooth_days + 1)
    peak_i = lo + int(np.argmax(swe[lo:hi]))
    peak = float(swe[peak_i])

    if smoothed[approx_i] <= threshold:  # snow-free year, signatures undefined
        return nan_result

    # Melt-out: first day at or below threshold that stays there for
    # persist_days. Near the end of the water year the window is truncated, so a
    # late melt-out only needs the remaining days to be clear.
    meltout = np.nan
    n = len(smoothed)
    for i in range(peak_i, n):
        if smoothed[i] <= threshold:
            if np.all(smoothed[i:min(i + persist_days, n)] <= threshold):
                meltout = float(doy[i])
                break

    if np.isnan(meltout):
        melt_rate = np.nan
    else:
        span = meltout - doy[peak_i]
        melt_rate = peak / span if span > 0 else np.nan

    return {"peak_swe": peak, "peak_doy": float(doy[peak_i]),
            "meltout_doy": meltout, "melt_rate": melt_rate}


def signatures_by_water_year(df, obs_col="mean_swe", sim_col="mean_swe_predict",
                             date_col="day", threshold=SWE_THRESHOLD,
                             smooth_days=SMOOTH_DAYS,
                             persist_days=PERSIST_DAYS,
                             min_peak_swe=MIN_PEAK_SWE, min_days=300):
    """
    Compute observed and simulated signatures for every complete water year.

    Args:
        df: DataFrame with a date column and observed/simulated SWE columns.
        obs_col, sim_col, date_col: column names.
        threshold: SWE (m) at or below which the ground counts as bare.
        smooth_days: window of the centered rolling median used to find peaks.
        persist_days: days below threshold required to declare melt-out.
        min_peak_swe: observed peak SWE (m) below which the percent-bias
            signatures are set to NaN. Absolute errors are always reported.
        min_days: water years with fewer days of data are dropped as partial.

    Returns:
        DataFrame indexed by water year with obs_/sim_ columns for each
        signature plus the simulated-minus-observed errors. Empty if no water
        year is complete.
    """
    d = df[[date_col, obs_col, sim_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).set_index(date_col)

    rows = []
    for wy, grp in d.groupby(water_year(d.index)):
        if len(grp) < min_days:
            continue
        # Reindex onto the full calendar so the rolling window and the
        # persistence check span real days even if the input has gaps.
        full = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full).interpolate(limit_direction="both")
        doy = day_of_water_year(grp.index)

        obs = _one_series(grp[obs_col].to_numpy(), doy, threshold,
                          smooth_days, persist_days)
        sim = _one_series(grp[sim_col].to_numpy(), doy, threshold,
                          smooth_days, persist_days)
        row = {"wy": int(wy), "n_days": len(grp)}
        row.update({f"obs_{k}": v for k, v in obs.items()})
        row.update({f"sim_{k}": v for k, v in sim.items()})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # timing errors in days, signed: positive means the simulation is late
    out["peak_doy_err"] = out["sim_peak_doy"] - out["obs_peak_doy"]
    out["meltout_doy_err"] = out["sim_meltout_doy"] - out["obs_meltout_doy"]
    # magnitude errors in metres, signed: positive means simulated too high.
    # No denominator, so these stay finite in low-snow years.
    out["peak_swe_err"] = out["sim_peak_swe"] - out["obs_peak_swe"]
    # ... and as percent bias, which is comparable across basins of different
    # snowiness but divides by the observed peak. In a year with a few
    # centimetres of snow that denominator is tiny and the percentage explodes -
    # the same small-denominator failure that makes the streamflow %BiasFLV
    # useless here. min_peak_swe blanks those years instead of reporting a
    # spurious several-hundred-percent bias.
    usable = out["obs_peak_swe"] >= min_peak_swe
    out["peak_swe_bias"] = np.where(
        usable, 100 * out["peak_swe_err"] / out["obs_peak_swe"], np.nan)
    out["melt_rate_bias"] = np.where(
        usable,
        100 * (out["sim_melt_rate"] - out["obs_melt_rate"]) / out["obs_melt_rate"],
        np.nan)
    out["below_min_peak"] = ~usable

    return out.set_index("wy")


SIGNATURE_COLS = ["peak_swe_err", "peak_swe_bias", "peak_doy_err",
                  "meltout_doy_err", "melt_rate_bias"]


def summarize(sig_df, prefix="test"):
    """
    Collapse per-water-year signatures into one value per signature, matching the
    shape of LSTM_metrics.calc_metrics so results log the same way.

    Reports the median signed error, plus the median absolute error for the
    timing terms: a model that alternates ten days early and ten days late has a
    median near zero but is not accurate.
    """
    if sig_df.empty:
        return {}
    out = {}
    for col in SIGNATURE_COLS:
        vals = sig_df[col].to_numpy(dtype=float)
        if np.all(np.isnan(vals)):       # nanmedian warns on an all-NaN slice
            out[f"{prefix}_{col}"] = np.nan
            out[f"{prefix}_{col}_abs"] = np.nan
            continue
        # Both are needed. The signed median says whether the model leans high
        # or low; the absolute median says how wrong a typical year is. Errors
        # that alternate in sign cancel in the signed median, so a model can
        # look unbiased while being badly wrong every single year.
        out[f"{prefix}_{col}"] = float(np.nanmedian(vals))
        out[f"{prefix}_{col}_abs"] = float(np.nanmedian(np.abs(vals)))
    out[f"{prefix}_n_water_years"] = int(sig_df[SIGNATURE_COLS].notna().any(axis=1).sum())
    return out


# ----------------------------------------------------------------------------
# self-check

def _triangle(peak_swe, peak_doy, meltout_doy, n_days=365):
    """Synthetic snowpack: linear rise to a peak, linear melt to bare ground."""
    doy = np.arange(1, n_days + 1)
    rise = peak_swe * doy / peak_doy
    melt = peak_swe * (meltout_doy - doy) / (meltout_doy - peak_doy)
    return np.maximum(0.0, np.where(doy <= peak_doy, rise, melt))


def _frame(obs, sim, start="2014-10-01"):
    days = pd.date_range(start, periods=len(obs), freq="D")
    return pd.DataFrame({"day": days, "mean_swe": obs, "mean_swe_predict": sim})


def _self_check():
    # 1. clean triangular year: locations exact, magnitudes within smoothing loss
    obs = _triangle(2.0, 180, 280)
    sim = _triangle(1.8, 190, 280)
    r = signatures_by_water_year(_frame(obs, sim)).iloc[0]
    assert r.obs_peak_doy == 180, r.obs_peak_doy
    assert r.sim_peak_doy == 190, r.sim_peak_doy
    assert r.peak_doy_err == 10, r.peak_doy_err
    assert abs(r.obs_peak_swe - 2.0) < 1e-9, r.obs_peak_swe
    assert abs(r.sim_peak_swe - 1.8) < 1e-9, r.sim_peak_swe
    assert abs(r.peak_swe_bias + 10.0) < 1e-6, r.peak_swe_bias
    assert abs(r.obs_meltout_doy - 280) <= 2, r.obs_meltout_doy
    assert abs(r.meltout_doy_err) <= 2, r.meltout_doy_err
    # obs melts 2.0 m over 100 days, sim 1.8 m over 90: same rate
    assert abs(r.melt_rate_bias) < 5, r.melt_rate_bias

    # 2. regression: a one-day autumn spike taller than the seasonal peak must
    #    not be mistaken for it (this is what corrupted the multi-HUC results)
    spiky = _triangle(1.0, 180, 280).copy()
    spiky[29] = 3.0
    r = signatures_by_water_year(_frame(obs, spiky)).iloc[0]
    assert r.sim_peak_doy == 180, f"spike stole the peak: doy {r.sim_peak_doy}"
    assert r.sim_peak_swe < 1.1, r.sim_peak_swe
    assert r.sim_meltout_doy > 250, r.sim_meltout_doy

    # 3. regression: a brief mid-melt dip to bare ground must not end the season
    dippy = _triangle(1.0, 180, 280).copy()
    dippy[219:222] = 0.0          # three clear days at doy 220-222
    r = signatures_by_water_year(_frame(obs, dippy)).iloc[0]
    assert r.sim_meltout_doy > 250, f"dip ended the season at {r.sim_meltout_doy}"

    # 4. a pack that never clears has no melt-out and no melt rate
    never = _triangle(2.0, 180, 400)[:365]
    r = signatures_by_water_year(_frame(obs, never)).iloc[0]
    assert np.isnan(r.sim_meltout_doy), r.sim_meltout_doy
    assert np.isnan(r.sim_melt_rate), r.sim_melt_rate

    # 5. snow-free year yields NaN rather than a spurious peak
    flat = np.zeros(365)
    assert np.isnan(signatures_by_water_year(_frame(flat, flat)).iloc[0].obs_peak_swe)

    # 5b. a trace snow year must not produce a several-hundred-percent bias.
    #     Observed peaks at 4 cm, simulated at 40 cm: the absolute error is
    #     honest, the percentage is not, so the percentage is withheld.
    trace = _triangle(0.04, 180, 240)
    big = _triangle(0.40, 180, 240)
    r = signatures_by_water_year(_frame(trace, big)).iloc[0]
    assert r.below_min_peak, "trace year should be flagged"
    assert np.isnan(r.peak_swe_bias), f"percent bias leaked: {r.peak_swe_bias}"
    assert np.isnan(r.melt_rate_bias), r.melt_rate_bias
    assert abs(r.peak_swe_err - 0.36) < 1e-6, r.peak_swe_err
    # and a normal year still reports both
    r = signatures_by_water_year(_frame(obs, sim)).iloc[0]
    assert not r.below_min_peak and not np.isnan(r.peak_swe_bias)

    # 6. partial water years are dropped
    assert signatures_by_water_year(_frame(obs, sim).head(100)).empty

    # 7. water year boundaries: Oct 1 opens a year, Sep 30 closes it
    d = pd.DatetimeIndex(["2014-09-30", "2014-10-01", "2015-09-30"])
    assert list(water_year(d)) == [2014, 2015, 2015]
    assert list(day_of_water_year(d)) == [365, 1, 365]

    # 8. gaps in the input are healed rather than shifting the calendar
    f = _frame(obs, sim)
    r = signatures_by_water_year(f.drop(f.index[100:110])).iloc[0]
    assert r.obs_peak_doy == 180, r.obs_peak_doy

    print("LSTM_signatures self-check passed (9 cases)")


if __name__ == "__main__":
    _self_check()
