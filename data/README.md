# RQ1 data

| File | Produced by | Notes |
|------|-------------|-------|
| `clip_manifest.csv` | `notebooks/vbench_score_clips.ipynb` | Clip registry |
| `metric_scores.csv` | `notebooks/vbench_score_clips.ipynb` | VBench / VBench-2.0 custom-input scores |
| `dimension_applicability.csv` | `notebooks/vbench_score_clips.ipynb` | Applicability and failures |
| `human_ratings.csv` | survey, anonymized | Columns: `video_id`, `rater_id`, `enjoyment` (integers 1–5), `partial_watch` (1 if the respondent did not watch the whole clip). Clips 01–10 only. No names. |
| `vbench_raw/` | scoring notebook | Regenerable JSON |
| `staged_clips/` | scoring notebook | Regenerable staged mp4/gif copies |

`human_ratings.csv` contains the anonymous survey ratings for clips 01–10. Clip 11 has saved VBench scores and no survey ratings. [RQ1 summary](../docs/rq1_summary.md) · [Results CSV](../rq1_outputs/rq1_kendall_results.csv)
