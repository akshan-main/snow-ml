# pylint: disable=C0103
"""
Paired statistical comparison of models evaluated on the same basins.

Comparing models on a shared set of basins is a *paired* problem: basin-to-basin
variation is far larger than the difference between models, so an unpaired test
(such as the Welch t-test used elsewhere in this repo) throws away the pairing
and loses most of its power.

Implements the chain used in Querbes Duhart (2026), sec. 3.6:

    1. Friedman test across all models for a given metric. It ranks models
       within each basin, so a basin that is simply hard does not swamp the
       comparison.
    2. If Friedman is not significant, the metric is NOT tested pairwise and is
       reported as "not tested" rather than "no difference".
    3. Otherwise, Wilcoxon signed-rank on every model pair.
    4. Bonferroni correction across the pairwise tests within that metric.

With only two models the Friedman step is undefined and is skipped: a single
Wilcoxon test needs no multiple-comparison correction.

Metric polarity
---------------
"Better" is not the same direction for every metric, and getting this wrong
silently inverts conclusions. Three cases:

    high   larger is better            KGE, NSE, R2
    low    smaller is better           MSE, RMSE, absolute errors
    zero   closer to zero is better    biases, signed timing errors

Every metric is mapped to a score where lower is always better, and both the
significance test and the winner are decided on that score. Reported medians are
the raw values, so tables read naturally. An unknown metric name raises rather
than guessing a direction.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

# Longest suffix wins, so "peak_doy_err_abs" resolves to low, not zero.
POLARITY = {
    "kge": "high", "nse": "high", "r2": "high",
    "mse": "low", "rmse": "low", "mae": "low",
    "peak_swe_bias": "zero",
    "peak_swe_err": "zero",
    "melt_rate_bias": "zero",
    "peak_doy_err": "zero",
    "meltout_doy_err": "zero",
    "peak_doy_err_abs": "low",
    "meltout_doy_err_abs": "low",
    "peak_swe_err_abs": "low",
    "peak_swe_bias_abs": "low",
    "melt_rate_bias_abs": "low",
}


def polarity_of(metric, extra=None):
    """Resolve a metric name to 'high', 'low' or 'zero'.

    Matches the longest known suffix, so prefixes such as "test_" are ignored.
    Raises KeyError on an unknown metric: guessing a direction here would
    silently invert results.
    """
    table = dict(POLARITY)
    if extra:
        table.update(extra)
    hits = [k for k in table if metric == k or metric.endswith("_" + k)]
    if not hits:
        raise KeyError(
            f"unknown metric '{metric}'; add it to POLARITY or pass "
            f"extra_polarity={{'{metric}': 'high'|'low'|'zero'}}")
    return table[max(hits, key=len)]


def to_score(values, polarity):
    """Map a metric onto a scale where smaller is always better."""
    v = np.asarray(values, dtype=float)
    if polarity == "high":
        return -v
    if polarity == "low":
        return v
    if polarity == "zero":
        return np.abs(v)
    raise ValueError(f"bad polarity {polarity!r}")


def _wide(df, metric, model_col, unit_col):
    """Units x models table for one metric, complete cases only."""
    w = df.pivot_table(index=unit_col, columns=model_col, values=metric,
                       aggfunc="first")
    return w.dropna(axis=0, how="any")


def compare_models(df, metrics, model_col="model", unit_col="unit",
                   alpha=ALPHA, extra_polarity=None):
    """
    Run the Friedman -> Wilcoxon -> Bonferroni chain for each metric.

    Args:
        df: long-form frame, one row per (model, unit), with a column per metric.
        metrics: metric column names to test.
        model_col, unit_col: column names identifying the model and the basin.
        alpha: significance level, applied to Friedman and to adjusted p-values.
        extra_polarity: optional {metric: 'high'|'low'|'zero'} for metrics not in
            POLARITY.

    Returns:
        (friedman, pairwise) DataFrames. `pairwise` carries one row per
        (metric, model_a, model_b) with raw and Bonferroni-adjusted p-values,
        the winner, and the number of usable pairs.
    """
    models = sorted(df[model_col].unique())
    if len(models) < 2:
        raise ValueError("need at least two models to compare")

    fried_rows, pair_rows = [], []

    for metric in metrics:
        pol = polarity_of(metric, extra_polarity)
        wide = _wide(df, metric, model_col, unit_col)
        n_units = len(wide)

        # --- Friedman gate -------------------------------------------------
        if len(models) == 2:
            # undefined for two groups; a single Wilcoxon needs no correction
            tested, fstat, fp = True, np.nan, np.nan
        elif n_units < 2:
            tested, fstat, fp = False, np.nan, np.nan
        else:
            scores = [to_score(wide[m], pol) for m in models]
            fstat, fp = stats.friedmanchisquare(*scores)
            tested = fp < alpha

        fried_rows.append({"metric": metric, "polarity": pol, "n_units": n_units,
                           "friedman_stat": fstat, "friedman_p": fp,
                           "tested": tested})

        pairs = list(combinations(models, 2))
        n_comp = len(pairs)

        for a, b in pairs:
            row = {"metric": metric, "polarity": pol,
                   "model_a": a, "model_b": b,
                   "median_a": float(np.nanmedian(wide[a])) if n_units else np.nan,
                   "median_b": float(np.nanmedian(wide[b])) if n_units else np.nan,
                   "n_units": n_units, "n_pairs": n_units, "tested": tested,
                   "p_raw": np.nan, "p_adj": np.nan,
                   "significant": False, "winner": None}

            if tested and n_units >= 2:
                sa, sb = to_score(wide[a], pol), to_score(wide[b], pol)
                diff = sa - sb
                if np.allclose(diff, 0):
                    # identical everywhere: no difference to test
                    row.update(p_raw=1.0, p_adj=1.0)
                else:
                    # zero_method="wilcox" drops exact ties, matching the
                    # thesis note that ties reduce the usable sample
                    _, p = stats.wilcoxon(sa, sb, zero_method="wilcox")
                    p_adj = min(1.0, p * n_comp)
                    row.update(p_raw=float(p), p_adj=float(p_adj),
                               significant=bool(p_adj < alpha))
                    row["n_pairs"] = int(np.count_nonzero(diff))
                    if row["significant"]:
                        md = float(np.median(diff))
                        if md == 0:  # symmetric spread; fall back to the mean
                            md = float(np.mean(diff))
                        row["winner"] = a if md < 0 else (b if md > 0 else None)

            pair_rows.append(row)

    return pd.DataFrame(fried_rows), pd.DataFrame(pair_rows)


# ----------------------------------------------------------------------------
# comparison matrix figure (thesis Figs. 15 and 18)

CELL_COLORS = {
    "a": "#E8622A",        # first model better, significant
    "b": "#1F6FB4",        # second model better, significant
    "similar": "#DFF0D8",  # tested, no significant difference
    "untested": "#D9D9D9",  # Friedman not significant, pairwise not run
}


def _auto_fmt(values):
    """Pick a decimal precision that keeps small-magnitude metrics readable."""
    v = np.abs(np.asarray([x for x in values if np.isfinite(x)], dtype=float))
    v = v[v > 0]
    if v.size == 0:
        return "{:.2f}"
    m = np.median(v)
    if m < 0.01:
        return "{:.4f}"
    if m < 1:
        return "{:.3f}"
    return "{:.2f}"


def plot_comparison_matrix(pairwise, metrics=None, title=None, path=None,
                           fmt=None):
    """
    Render the model-pair x metric matrix from thesis Figs. 15 and 18.

    Each cell shows the two medians, coloured by which model won and whether the
    difference survived correction. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # local import: keeps stats usable headless

    metrics = metrics or list(dict.fromkeys(pairwise["metric"]))
    pairs = list(dict.fromkeys(zip(pairwise.model_a, pairwise.model_b)))

    fig, ax = plt.subplots(figsize=(1.15 * len(metrics) + 2.2,
                                    1.15 * len(pairs) + 2.6))
    ax.set_xlim(0, len(metrics))
    ax.set_ylim(0, len(pairs))
    ax.invert_yaxis()
    ax.set_aspect("equal")  # keep cells square however many pairs there are

    # per-metric precision unless the caller pinned one
    fmts = {m: (fmt or _auto_fmt(
        pd.concat([pairwise.loc[pairwise.metric == m, "median_a"],
                   pairwise.loc[pairwise.metric == m, "median_b"]]).tolist()))
        for m in metrics}
    idx = pairwise.set_index(["metric", "model_a", "model_b"])
    for r, (a, b) in enumerate(pairs):
        for c, metric in enumerate(metrics):
            try:
                row = idx.loc[(metric, a, b)]
            except KeyError:
                continue
            if not row["tested"]:
                color = CELL_COLORS["untested"]
            elif not row["significant"]:
                color = CELL_COLORS["similar"]
            else:
                color = CELL_COLORS["a"] if row["winner"] == a else CELL_COLORS["b"]
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=2))
            dark = color in (CELL_COLORS["a"], CELL_COLORS["b"])
            txt = "white" if dark else "black"
            cf = fmts[metric]
            ax.text(c + .5, r + .26, cf.format(row["median_a"]), ha="center",
                    va="center", fontsize=8.5, fontweight="bold", color=txt)
            ax.plot([c + .38, c + .62], [r + .5, r + .5], color=txt,
                    lw=.8, alpha=.7)
            ax.text(c + .5, r + .74, cf.format(row["median_b"]), ha="center",
                    va="center", fontsize=8.5, fontweight="bold", color=txt)

    ax.set_xticks(np.arange(len(metrics)) + .5)
    ax.set_xticklabels([m.replace("test_", "").replace("_", "\n")
                        for m in metrics], fontsize=8)
    ax.set_yticks(np.arange(len(pairs)) + .5)
    ax.set_yticklabels([f"{a}\nvs\n{b}" for a, b in pairs], fontsize=8)
    ax.xaxis.set_ticks_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=28)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CELL_COLORS[k])
               for k in ("a", "b", "similar", "untested")]
    ax.legend(handles,
              ["first model better", "second model better",
               f"no difference (p ≥ {ALPHA} after Bonferroni)",
               "not tested (Friedman n.s.)"],
              loc="upper center", bbox_to_anchor=(.5, -0.10),
              ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# self-check

def _long(units, **models):
    """Build a long-form frame from {model_name: {metric: [values]}}."""
    rows = []
    for name, metrics in models.items():
        for i, u in enumerate(units):
            row = {"model": name, "unit": u}
            row.update({k: v[i] for k, v in metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _self_check():
    units = [f"b{i}" for i in range(20)]
    rng = np.random.default_rng(0)

    # 1. polarity resolution, including the longest-suffix rule
    assert polarity_of("test_kge") == "high"
    assert polarity_of("test_mse") == "low"
    assert polarity_of("test_peak_swe_bias") == "zero"
    assert polarity_of("test_peak_doy_err") == "zero"
    assert polarity_of("test_peak_doy_err_abs") == "low", "suffix precedence"
    try:
        polarity_of("test_mystery")
        raise AssertionError("unknown metric must raise, not guess")
    except KeyError:
        pass

    # 2. 'high' metric: A genuinely better on KGE
    df = _long(units,
               A={"test_kge": list(0.80 + rng.normal(0, .01, 20))},
               B={"test_kge": list(0.60 + rng.normal(0, .01, 20))})
    _, pw = compare_models(df, ["test_kge"])
    r = pw.iloc[0]
    assert r.significant and r.winner == "A", (r.winner, r.p_adj)

    # 3. 'low' metric: smaller MSE wins
    df = _long(units,
               A={"test_mse": list(2.0 + rng.normal(0, .05, 20))},
               B={"test_mse": list(1.0 + rng.normal(0, .05, 20))})
    _, pw = compare_models(df, ["test_mse"])
    assert pw.iloc[0].winner == "B", pw.iloc[0].winner

    # 4. 'zero' metric: the sign must not decide. A is -30%, B is +5%;
    #    B is closer to zero and must win even though its raw median is larger.
    df = _long(units,
               A={"test_peak_swe_bias": list(-30 + rng.normal(0, 1, 20))},
               B={"test_peak_swe_bias": list(5 + rng.normal(0, 1, 20))})
    _, pw = compare_models(df, ["test_peak_swe_bias"])
    r = pw.iloc[0]
    assert r.winner == "B", f"polarity inverted: {r.winner}"
    assert r.median_a < r.median_b, "raw medians should still be reported"

    # 5. identical models are not significantly different
    same = list(0.7 + rng.normal(0, .01, 20))
    df = _long(units, A={"test_kge": same}, B={"test_kge": list(same)})
    _, pw = compare_models(df, ["test_kge"])
    assert not pw.iloc[0].significant and pw.iloc[0].winner is None

    # 6. Bonferroni is applied and scales with the number of pairs
    base = 0.5 + rng.normal(0, .02, 20)
    df = _long(units,
               A={"test_kge": list(base + .04)},
               B={"test_kge": list(base)},
               C={"test_kge": list(base - .04)})
    fr, pw = compare_models(df, ["test_kge"])
    assert fr.iloc[0].tested, "Friedman should fire on a real 3-model spread"
    ab = pw[(pw.model_a == "A") & (pw.model_b == "B")].iloc[0]
    assert np.isclose(ab.p_adj, min(1.0, ab.p_raw * 3)), (ab.p_raw, ab.p_adj)

    # 7. Friedman gate: three indistinguishable models are not tested pairwise
    flat = 0.7 + rng.normal(0, .01, 20)
    df = _long(units, A={"test_kge": list(flat)},
               B={"test_kge": list(flat + rng.normal(0, 1e-6, 20))},
               C={"test_kge": list(flat + rng.normal(0, 1e-6, 20))})
    fr, pw = compare_models(df, ["test_kge"])
    if not fr.iloc[0].tested:
        assert (~pw.significant).all() and pw.p_raw.isna().all(), \
            "ungated metrics must report NaN p-values, not p=1"

    # 8. unpaired units are dropped rather than silently misaligned
    df = _long(units, A={"test_kge": list(0.8 + rng.normal(0, .01, 20))},
               B={"test_kge": list(0.6 + rng.normal(0, .01, 20))})
    df = df.drop(df[(df.model == "B") & (df.unit.isin(["b0", "b1"]))].index)
    _, pw = compare_models(df, ["test_kge"])
    assert pw.iloc[0].n_pairs == 18, pw.iloc[0].n_pairs

    print("LSTM_compare self-check passed (8 cases)")


if __name__ == "__main__":
    _self_check()
