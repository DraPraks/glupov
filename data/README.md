# RQ1 data

| File | Produced by | Notes |
|------|-------------|-------|
| `clip_manifest.csv` | `notebooks/vbench_score_clips.ipynb` | Clip registry |
| `metric_scores.csv` | `notebooks/vbench_score_clips.ipynb` | VBench / VBench-2.0 custom-input scores |
| `dimension_applicability.csv` | `notebooks/vbench_score_clips.ipynb` | Applicability and failures |
| `human_ratings.csv` | **manual** (Google Docs) | Columns: `video_id`, `rater_id`, `enjoyment` (1–5) |
| `vbench_raw/` | scoring notebook | Regenerable JSON |
| `staged_clips/` | scoring notebook | Regenerable staged mp4/gif copies |

Pilot `source_id` is `unlabeled_pilot` until real drama IDs are assigned.
