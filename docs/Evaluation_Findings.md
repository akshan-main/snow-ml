# SnowML evaluation: findings to date

Akshan Krithick, September 2026

## What I did

The goal was to bring the evaluation methods from Hernán Querbes Duhart's thesis (signature metrics, paired statistics, characterizing inputs before training) onto the SnowML repo. I didn't retrain anything. Everything below comes from files already in the repo, or from the S3 buckets once I got read access.

Two new modules. `LSTM_signatures.py` computes, per water year, peak SWE, peak date, melt-out date and melt rate from a predicted vs observed series. `LSTM_compare.py` runs the Friedman, Wilcoxon signed-rank and Bonferroni chain and draws the comparison matrix from the thesis. Both have self-checks covering the failure cases I hit along the way.

Data used: the 44 saved prediction files in `mlflow_data/predictions` (22 Tuolumne HUC12s, each with a locally trained and a multi-HUC model), the 137 MLflow metric dumps in `mlflow_data/run_id_data`, the Ex2 per-basin metrics CSV, and from S3 the 534 Ex2 model-ready files plus the ERA5-Land snow files for water years 2017 and 2021.

## Findings

Ordered by how much I trust them.

**1. UA SWE and ERA5-Land SWE disagree about how much snow some basins hold, and the disagreement is stable from year to year.**

Daily correlation between the two products is 0.97 (median across 534 basins), so they agree on shape. Peak magnitude is another story. In water year 2021, ERA5 peak SWE was more than double UA in 51 basins and less than half in 24. That isn't noise: a basin's ERA5/UA peak ratio in WY2017 predicts its ratio in WY2021 with Spearman 0.76, and 31 basins are over 2x in both years.

This needs no model. It's a property of the two target datasets. It matters because UA SWE is built on PRISM precipitation and so is GRIDMET, the only forcing the repo uses. ERA5 is the first PRISM-independent target sitting in the bucket, and it's there for every Ex2 basin.

**2. Precipitation character does not explain model skill once you control for where the basin is.**

I computed annual total, dry-day fraction, mean dry-spell length, 95th percentile daily rain, and the share of rain falling on the wettest 5% of days, for all 534 Ex2 basins, and correlated each with Ex2's test KGE. Everything is weak (|rho| under 0.17, except dry-spell length at -0.33), and every one of them flips sign when elevation is held fixed.

The one thing that looked real was storm concentration in Maritime basins: rho +0.50, still +0.31 after controlling for elevation, bootstrap CI [+0.17, +0.44]. I chased it. It's heterogeneous inside Maritime (Upper Skagit +0.54, Wenatchee -0.31). Storm concentration correlates with longitude at 0.90. Control for longitude and the effect drops to +0.10. Split basins by which side of the Cascade crest they drain to, control for both elevation and longitude, and it's -0.14 on the west side (p=0.2) and zero on the east. There's nothing there. Drier basins east of the crest score higher, and stormier precipitation was a proxy for that.

I'm reporting this dead end on purpose. It has the same shape as the dry-day-ratio result in the thesis.

**3. The elevation effect on KGE is mostly a temperature effect.**

Ex2 found elevation explains about a third of KGE variance. Ex3 found it stopped mattering once elevation was a feature. Both are right. Raw Spearman between elevation and KGE is +0.64. Partial out mean air temperature and it drops to +0.23. Inside Montane Forest basins it's +0.04. Colder basins are easier to model, and elevation was standing in for cold.

**4. Signed error medians hide errors that alternate in sign.**

When I first reported peak SWE bias for the locally trained Tuolumne models I got -3.6%, which looked nearly unbiased. The median absolute error is 13.8%. Year-to-year sign consistency is only 0.58 to 0.74 across the signatures, so a model that's 14% high one year and 14% low the next reports as unbiased. Switching to absolute error flipped one statistical verdict (melt rate bias went from not significant to significant at p=0.01). Both modules now report signed and absolute together.

**5. A model trained on Pacific Northwest basins and applied to the Sierra gets peak timing right but magnitude and melt-out wrong.**

The 22 Tuolumne basins aren't in any training split, so the multi-HUC predictions there are a true out-of-region test. Over the 12 water years both models cover: peak timing error isn't different (6.5 vs 9.25 days, p=0.93). Peak SWE absolute error is 13.8% local vs 30.5% multi, p<0.001. Melt-out absolute error is 5.75 vs 11.5 days, p<0.001. KGE says 0.89 vs 0.67 and stops there.

Caveat that matters: 22 basins in one HUC08 aren't independent, so those p-values are optimistic. And "local vs multi" differs on region, training scope, and whether the model saw the basin, all at once. This describes how the transferred model fails. It doesn't say why.

**6. Per-basin KGE moves a lot from run to run.**

Across the 130 runs in the repo's MLflow dumps, the typical basin's KGE has an interquartile range of 0.13 (last epoch) or 0.08 (best epoch), and over 430 of 534 basins have a range above 0.30. The runs mix seeds and configurations, so I can't split that into seed noise vs bad runs without the MLflow parameters. Either way, one run per basin isn't stable enough to rank basins or compare them between experiments, which is what Ex2 and Ex3 do.

One thing I had wrong and withdrew: I initially framed Ex2's numbers as one draw from this distribution. They aren't. Ex2's KGE values match none of the 130 runs; its runs aren't in the repo.

## Things I checked that turned out fine

The KGE implementation I used reproduces the stored values to 1e-7 across all 44 prediction files. Observed SWE is identical between the local and multi files for every basin. The partial correlation code matches the closed-form formula. Every number above was recomputed independently before I wrote this.

## Things that aren't established

Whether a different precipitation product changes skill (not run). Why the transferred model overestimates peak SWE (finding 5 is descriptive). Whether Ex2's specific per-basin numbers are stable (needs its runs).

## Bucket facts that differ from the docs

3,155 HUC12s plus 11 Skagit HUC10s in `snowml-model-ready`, not "over 500". Only the 534 Ex2 basins have published metrics. Two file schemas coexist (one-hot snow classes vs predominant snow plus forest cover plus lags). There's ERA5-Land snow, DHSVM output, and Hive-partitioned snow cover parquet in the bucket, all added between February and July 2026. Someone is actively working on this and I'd like to know who so I don't duplicate them.

## What I'd do next

Get Ex2's MLflow run IDs so finding 6 can be pinned down properly. Rerun the Tuolumne evaluation against ERA5 SWE as the target and see whether the timing/magnitude split holds when the target isn't PRISM-derived. Then the precipitation swap, with ERA5 as the independent target so the result is interpretable.
