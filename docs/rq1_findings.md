# RQ1 findings: survey enjoyment and VBench scores

These numbers are pilot associations. `source_id` on every clip is still a placeholder (`drama_01`–`drama_11`, `source_id_basis` = `placeholder`), not a verified drama id. The notebook wrote coefficients because those ids are unique and are not the blocked label `unlabeled_pilot`. They do not show that the ten clips are independent dramas. Do not read this file as a confirmatory claim about independent dramas.

Spearman was not estimated. Like and dislike text was not entered into the coefficient.

## Sample

The survey is one row per respondent. Each of the 29 rows received its own rater id, `r01` through `r29`, in file order. A blank or repeated name would still have received its own id. Names are not in this repository.

Every enjoyment cell for clips 01–10 was an integer from 1 to 5. No cell was dropped. The ratings file has 290 rows: 10 clips × 29 raters. Each eligible clip has 29 ratings, above the notebook’s minimum of 3.

| Item | Count |
|------|------:|
| Respondents | 29 |
| Clips with enjoyment | 10 (`drama_clip_01`–`drama_clip_10`) |
| Enjoyment rows kept | 290 |
| Enjoyment cells dropped | 0 |
| Partial-watch ratings kept | 30 |
| Respondents with at least one partial watch | 8 |

Age group: 18–24, 14; 25–34, 8; 35–44, 3; 45–54, 2; 55 or older, 2.

Before this study, had they watched AI-generated short dramas or microdramas: Yes 25, No 3, Unsure 1.

Watching frequency, regardless of how the drama was made: a few times a month 10, a few times a week 8, less than once a month 6, never 3, daily or almost daily 2.

General interest in short dramas or microdramas, on a 1–5 item that is separate from clip enjoyment: 1 (2 people), 2 (7), 3 (10), 4 (6), 5 (4).

## Clips used

Clip columns in the survey run from Clip 01 through Clip 10. There is no Clip 11 enjoyment column. No clip 11 rating was invented. `drama_clip_11` remains in `data/metric_scores.csv` with placeholder `source_id` `drama_11` and its existing VBench scores. The notebook excluded it: 0 ratings, below the minimum of 3. Eligible clips for Kendall’s tau-b: N = 10.

`video_id` maps Clip 01–10 to `drama_clip_01`–`drama_clip_10`. The notebook’s clip table, printed to three decimals, is the mean and sample SD of the kept 1–5 ratings:

| video_id | source_id | n_raters | mean enjoyment | SD | Partial watches kept |
|----------|-----------|--------:|---------------:|---:|---------------------:|
| drama_clip_01 | drama_01 | 29 | 3.276 | 1.032 | 4 |
| drama_clip_02 | drama_02 | 29 | 2.828 | 1.104 | 1 |
| drama_clip_03 | drama_03 | 29 | 2.966 | 1.117 | 4 |
| drama_clip_04 | drama_04 | 29 | 2.931 | 1.223 | 2 |
| drama_clip_05 | drama_05 | 29 | 3.103 | 0.817 | 4 |
| drama_clip_06 | drama_06 | 29 | 2.828 | 1.104 | 3 |
| drama_clip_07 | drama_07 | 29 | 3.000 | 0.926 | 1 |
| drama_clip_08 | drama_08 | 29 | 3.345 | 0.936 | 2 |
| drama_clip_09 | drama_09 | 29 | 3.034 | 1.117 | 5 |
| drama_clip_10 | drama_10 | 29 | 3.241 | 0.988 | 4 |

A partial watch is a respondent–clip cell where the answer to “Did you watch the video in its entirety?” was No. All 30 of those cells had enjoyment in 1–5, and all 30 stayed in the mean. The 30 cells come from 8 of the 29 respondents. In `data/human_ratings.csv` those rows are `partial_watch` 1, and the other 260 rows are 0. The association notebook does not read that column.

VBench floats in `data/metric_scores.csv` were not changed.

## What the coefficient uses

The outcome is the clip-level mean of the kept enjoyment ratings. The predictors are the scored columns in `data/dimension_applicability.csv`, in table order:

- `vbench_subject_consistency`
- `vbench_background_consistency`
- `vbench_motion_smoothness`
- `vbench_dynamic_degree`
- `vbench_aesthetic_quality`
- `vbench_imaging_quality`
- `vbench2_human_anatomy`

Identity, clothes, Diversity, and Multi-View Consistency stay out of the family. The notebook’s settings for this run: seed 42, 2000 bootstrap replicates, 4999 Monte Carlo permutations. Complete cases are 10, which is above 7, so the permutation p-values are Monte Carlo, not exact. The Monte Carlo p-value is `(1 + count of |tau| at least the observed |tau|) / (1 + 4999)` and is never zero.

Each 95% interval is a marginal paired-clip bootstrap percentile interval. It conditions on this viewer panel. It is not a simultaneous family-wide interval. Holm uses the seven scored dimensions at alpha 0.05. A dimension with no variation stays in that family as an internal p of 1. Its coefficient and p-values are left blank in the results file.

The results file records 9 distinct mean-enjoyment values on the 10 eligible clips, so at least one enjoyment tie enters tau-b.

## Coefficients

Quoted from `rq1_outputs/rq1_kendall_results.csv`. Blank cells in that file are blank here.

| Dimension | Status | tau-b | 95% CI lower | 95% CI upper | Raw permutation p | Holm p | Holm rejects at 0.05 | Median-enjoyment tau-b |
|-----------|--------|------:|-------------:|-------------:|------------------:|-------:|----------------------|------------------------|
| vbench_subject_consistency | tested | -0.08989331499509895 | -0.6585365853658537 | 0.46173336783101965 | 0.804 | 1.0 | False | blank |
| vbench_background_consistency | tested | 0.08989331499509895 | -0.4361119579623079 | 0.6216443749874531 | 0.7834 | 1.0 | False | blank |
| vbench_motion_smoothness | tested | -0.13483997249264842 | -0.7948717948717948 | 0.4014775342734831 | 0.667 | 1.0 | False | blank |
| vbench_dynamic_degree | no_variation | blank | blank | blank | blank | blank | False | blank |
| vbench_aesthetic_quality | tested | 0.26967994498529685 | -0.4146341463414634 | 0.7948717948717948 | 0.3176 | 1.0 | False | blank |
| vbench_imaging_quality | tested | 0.22473328748774737 | -0.42128667156473765 | 0.8111666584329661 | 0.4336 | 1.0 | False | blank |
| vbench2_human_anatomy | tested | -0.08989331499509895 | -0.7 | 0.4618016825408862 | 0.7868 | 1.0 | False | blank |

Every tested row has `n_eligible_clips` 10, `n_complete_cases` 10, `n_missing_metric` 0, `n_unique_metric_values` 10, `n_unique_enjoyment_values` 9, `permutation_method` `monte_carlo`, `bootstrap_attempted_replicates` 2000, `bootstrap_valid_replicates` 2000, and `bootstrap_undefined_replicates` 0.

Holm does not reject at 0.05 for any scored dimension. Every tested Holm p-value in the results file is 1.0, and every `reject_holm` value is False. Every reported 95% interval includes both negative and positive values.

### Dynamic degree is undefined

`vbench_dynamic_degree` is 1.0 on every clip in `data/metric_scores.csv`, including the 10 eligible clips. The results row is `analysis_status` `no_variation`, with `n_unique_metric_values` 1, `n_complete_cases` 10, and `n_missing_metric` 0. Tau-b is undefined. The results file leaves `kendall_tau_b`, both CI limits, `p_permutation_raw`, and `p_permutation_holm` blank. `permutation_method` is `not_applicable`. Bootstrap attempted, valid, and undefined counts are 0. `reject_holm` is False. The dimension stays in the Holm family as an internal p of 1. That blank coefficient is not a tau-b of zero.

### Median-enjoyment tau-b is undefined

The sensitivity check repeats tau-b with the clip median instead of the mean. It is not a second confirmatory test. Every eligible clip has median enjoyment 3, so the median outcome does not vary. `kendall_tau_b_median` is blank on every row of the results file, including the six tested dimensions. The notebook printed those cells as NA.

## Reading the pilot

None of the scored dimensions is significant after Holm at 0.05. The largest tau-b among defined coefficients is aesthetic quality, 0.26967994498529685, then imaging quality, 0.22473328748774737. Subject consistency and human anatomy share tau-b -0.08989331499509895. Background consistency is 0.08989331499509895. Motion smoothness is -0.13483997249264842. With 10 clips, the intervals are wide.

The pairs inside tau-b are not extra independent clips. More raters tighten each clip mean. They do not raise the number of clip-level observations above 10. The intervals condition on this panel. The associations are not causal. Nonsignificance is not evidence that an association is absent.

The source-id limit remains. `drama_01`–`drama_11` are still placeholders. These coefficients are pilot associations, not confirmatory claims about independent dramas.
