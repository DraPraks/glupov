# RQ1 methodology

RQ1 asks how strongly selected automated scores associate with human enjoyment of a clip. The unit is the clip. The procedure is the confirmatory path in `notebooks/rq1-aidrama.ipynb`, written by `scripts/build_rq1_notebooks.py`. This note records that procedure. It reports no coefficients.

## Eligible clips

Ratings are integers on a 1–5 scale. The clip outcome is the mean of those ratings. A clip needs at least 3 ratings to be eligible. N is the number of eligible clips. The association requires at least 4 eligible clips.

## Confirmatory dimensions

The family is the `scored` rows of `data/dimension_applicability.csv`, in table order, by `column_name`. On this pilot those columns are the six VBench 1.0 scores plus `vbench2_human_anatomy`:

- `vbench_subject_consistency`
- `vbench_background_consistency`
- `vbench_motion_smoothness`
- `vbench_dynamic_degree`
- `vbench_aesthetic_quality`
- `vbench_imaging_quality`
- `vbench2_human_anatomy`

Identity, clothes, Diversity, and Multi-View Consistency stay out.

## Coefficient

The coefficient is Kendall's tau-b, `scipy.stats.kendalltau(..., variant="b")`:

\[
\tau_b = \frac{P - Q}{\sqrt{(P + Q + T)(P + Q + U)}}
\]

P and Q are concordant and discordant clip pairs. T counts pairs tied only on the score. U counts pairs tied only on mean enjoyment. Pairs tied on both are in neither count. SciPy ranks the two columns as part of the coefficient. There is no separate saved rank table. Scores are not rounded or jittered. Missing scores stay missing.

A constant column leaves tau-b undefined. On this pilot that column is dynamic degree (`vbench_dynamic_degree`). The undefined coefficient is not stored as zero. The dimension stays in the family and enters Holm as an internal p of 1.

## Uncertainty

The interval is a paired-clip bootstrap. Clips are resampled with replacement and tau-b is recomputed. The notebook default is 2000 replicates. The reported interval is the 95% percentile interval, and only when at least 100 replicates are defined. A replicate that makes either column constant is undefined and is left out of the percentiles. The intervals are not family-wide.

## Test

The test is two-sided. Mean enjoyment is permuted and the scores stay fixed. The statistic is absolute tau-b. When complete cases are 7 or fewer, every enjoyment permutation is enumerated. Otherwise the test is Monte Carlo. The notebook default is 4999 draws, and that p-value is never zero. Holm is applied across the scored family. Median enjoyment repeats tau-b as a sensitivity check, not as a second confirmatory test. Dimensions are not ranked by correlation.

## Placeholder data

The checked-in `data/human_ratings.csv` and the `source_id` values `drama_01`–`drama_11` are placeholder proof-of-concept data. They are not collected ratings or real drama IDs. A file the notebook writes from them (`rq1_outputs/rq1_kendall_results.csv`) is not a study result.
