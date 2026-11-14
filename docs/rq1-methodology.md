# RQ1 methodology

RQ1 asks how strongly selected automated scores associate with human enjoyment of a clip. The unit is the clip. The procedure is the confirmatory path in `notebooks/rq1-aidrama.ipynb`, written by `scripts/build_rq1_notebooks.py`. This note records that procedure. It reports no coefficients.

1. Keep a clip when it has at least 3 enjoyment ratings. Ratings are integers on a 1–5 scale, and the clip outcome is their mean. N is the number of eligible clips. Stop if N is below 4. Also stop if a `source_id` is missing, provisional (`unlabeled_pilot`), or repeated. A missing column, a blank value, or one provisional or repeated value among the eligible clips stops the run.

2. Take the confirmatory columns from the `scored` rows of `data/dimension_applicability.csv`, in table order, by `column_name`. On this pilot those columns are the six VBench 1.0 scores plus `vbench2_human_anatomy`: `vbench_subject_consistency`, `vbench_background_consistency`, `vbench_motion_smoothness`, `vbench_dynamic_degree`, `vbench_aesthetic_quality`, `vbench_imaging_quality`, and `vbench2_human_anatomy`. Identity, clothes, Diversity, and Multi-View Consistency stay out.

3. For each column, keep the complete cases: eligible clips with a finite score. Scores are not rounded or jittered. Missing scores stay missing. Fewer than 4 complete cases leaves that dimension untestable.

4. Take every pair of clips that both have a finite score and a mean enjoyment. The pair is concordant (P) when the clip with the higher score also has the higher enjoyment. It is discordant (Q) when the clip with the higher score has the lower enjoyment. T is a tie on the score only. U is a tie on enjoyment only. A tie on both is in neither count. Tau-b is (P−Q)/sqrt((P+Q+T)(P+Q+U)). SciPy ranks the two columns as part of the coefficient and does this inside `scipy.stats.kendalltau(..., variant="b")`. There is no separate saved rank table.

5. A constant column leaves tau-b undefined. On this pilot that column is dynamic degree (`vbench_dynamic_degree`). The undefined coefficient is not stored as zero. Every untestable scored dimension enters Holm as an internal p of 1: a constant score or constant mean enjoyment, a missing column, fewer than 4 complete cases, or all-missing scores. The reported coefficient stays unavailable.

6. For uncertainty, resample paired clips with replacement and recompute tau-b. The notebook default is 2000 replicates. Report the 95% percentile interval only when at least 100 replicates are defined. A replicate that makes either column constant is undefined and is left out of the percentiles. The intervals are not family-wide. The default seed is 42 (`RQ1_SEED`). Each dimension uses that seed plus its zero-based index in the scored family, so order changes later dimensions' bootstrap and Monte Carlo draws.

7. Test with a two-sided permutation of mean enjoyment. The scores stay fixed, and the statistic is absolute tau-b. When complete cases are 7 or fewer, enumerate every enjoyment permutation. Otherwise use Monte Carlo. The notebook default is 4999 draws, and that p-value is never zero.

8. Apply Holm across the scored family at α = 0.05.

9. Repeat tau-b on median enjoyment as a sensitivity check, not as a second confirmatory test.

10. Do not rank dimensions by correlation.

11. The checked-in `data/human_ratings.csv` and the `source_id` values `drama_01`–`drama_11` are placeholder proof-of-concept data. They are not collected ratings or real drama IDs. A file the notebook writes from them (`rq1_outputs/rq1_kendall_results.csv`) is not a study result.
