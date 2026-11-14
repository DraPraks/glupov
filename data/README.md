# RQ1 data

| File | Produced by | Notes |
|------|-------------|-------|
| `clip_manifest.csv` | `notebooks/vbench_score_clips.ipynb` | Clip registry |
| `metric_scores.csv` | `notebooks/vbench_score_clips.ipynb` | VBench / VBench-2.0 custom-input scores |
| `dimension_applicability.csv` | `notebooks/vbench_score_clips.ipynb` | Applicability and failures |
| `human_ratings.csv` | **placeholder** | Columns: `video_id`, `rater_id`, `enjoyment` (1–5). Not collected ratings. |
| `vbench_raw/` | scoring notebook | Regenerable JSON |
| `staged_clips/` | scoring notebook | Regenerable staged mp4/gif copies |

Placeholder proof-of-concept data for 11 clips. `source_id` values `drama_01`–`drama_11` and `human_ratings.csv` are invented so `notebooks/rq1-aidrama.ipynb` can write an association file. They are not real drama IDs or collected enjoyment ratings. The VBench score columns are from the pilot scoring run.
