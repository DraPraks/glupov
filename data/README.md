# RQ1 data

| File | Produced by | Notes |
|------|-------------|-------|
| `clip_manifest.csv` | `notebooks/vbench_score_clips.ipynb` | Clip registry |
| `metric_scores.csv` | `notebooks/vbench_score_clips.ipynb` | VBench / VBench-2.0 custom-input scores |
| `dimension_applicability.csv` | `notebooks/vbench_score_clips.ipynb` | Applicability and failures |
| `human_ratings.csv` | survey, anonymized | Columns: `video_id`, `rater_id`, `enjoyment` (integers 1–5), `partial_watch` (1 if the respondent did not watch the whole clip). Clips 01–10 only. No names. |
| `vbench_raw/` | scoring notebook | Regenerable JSON |
| `staged_clips/` | scoring notebook | Regenerable staged mp4/gif copies |

`source_id` values `drama_01`–`drama_11` are still placeholders (`source_id_basis` is `placeholder`), not verified drama ids. `human_ratings.csv` replaces the earlier invented ratings with anonymous survey enjoyment for clips 01–10. Clip 11 has no rows in `human_ratings.csv`. The survey also has no Clip 11 enjoyment column. The VBench score columns are from the pilot scoring run and are unchanged. Machine-readable association output is in `rq1_outputs/rq1_kendall_results.csv`. The summary is `docs/rq1_findings.md`.
