# Evaluating Agreement Between Automated Metrics and Human Enjoyment of AI-Generated Drama Clips

Context document (RQ1 + RQ2)

**Paper title:** Evaluating Agreement Between Automated Metrics and Human Enjoyment of AI-Generated Drama Clips

This document consolidates the research questions, methodology, analysis plans, data structures, and reporting guidance for the study. It is intended as shared context for further paper development, code implementation, and analysis.

## Overview

The study evaluates two complementary automated approaches against human enjoyment judgments of short AI-generated drama clips (typically 15–30 seconds):

- **RQ1** tests whether selected automated video-quality metrics (VBench and VBench-2.0 dimensions in custom-input mode) are associated with human enjoyment.
- **RQ2** tests how closely video-capable Large Multimodal Model (LMM) judges align with humans on both numerical ratings and stated reasons.

All analyses measure association or alignment, not causation. Conclusions are limited to the sampled clips and participant population. Human–human baselines should be reported where relevant for context.

**Primary distinction:** RQ1 treats video-quality metrics as potential indicators of enjoyment. RQ2 treats LMM judges as potential approximations of human judgments (ratings + explanations).

## Research Questions

**RQ1 — Automated video-quality metrics**

How strongly are scores on selected VBench and VBench-2.0 dimensions associated with human viewers’ enjoyment ratings of AI-generated drama clips?

**RQ2 — LMM judges**

How closely do LMM judges’ enjoyment ratings and stated reasons align with those of human viewers for AI-generated drama clips?

RQ2 has two separate components that should be reported independently:

- **Rating alignment:** Does the LMM produce similar scores and rank clips similarly to humans?
- **Reasoning alignment:** Does the LMM identify the same aspects humans mention and make compatible claims about them?

Matching a numerical score does not imply matching explanations. Outputs are observable justifications, not evidence of internal model reasoning.

## Shared Study Design Assumptions

- Unit of analysis for primary correlations/alignment: **clip** (video).
- Multiple human ratings per clip are aggregated to a clip-level mean (and SD) for rating comparisons. Individual human records are retained for reasoning analysis and human–human baselines.
- Independent clips assumed for basic inference. Clips from the same source/drama may be dependent; either sample one per independent source or use cluster-aware methods. Notebooks/code should flag repeated `source_id`s.
- Fixed 1–5 integer enjoyment scale with shared anchors for all humans and the LMM (e.g., 1 = “not enjoyable at all”, 5 = “extremely enjoyable”).
- LMM is blind to human ratings/explanations and to VBench scores.
- Same clips, scale, and (as far as possible) audiovisual information available to humans and LMM.
- Dimensions/metrics, exclusions, aggregation rules, and analysis choices fixed before inspecting associations.
- Example scale for planning: ~30 clips, ~5 human viewers per clip, one independent LMM assessment per clip. Actual N may vary; report sample sizes at each stage.

## RQ1 Methodology

**Goal:** Measure the strength of monotonic association between each selected automated dimension and mean human enjoyment across clips.

### Selected Dimensions (Custom-Input Mode)

VBench (official custom-input supported):

- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `dynamic_degree`
- `aesthetic_quality`
- `imaging_quality`

The confirmatory set is every row in `data/dimension_applicability.csv` with status `scored`: those six VBench 1.0 dimensions plus VBench-2.0 `Human_Anatomy` (`vbench2_human_anatomy`). Column names for the VBench 1.0 dimensions are `vbench_` plus the dimension name.

VBench-2.0 `Human_Identity` and `Human_Clothes` are excluded for this multi-person pilot: both score one person, and the drama clips show several. Diversity and Multi-View Consistency need more than one clip per prompt. `data/dimension_applicability.csv` marks identity and clothes `not_applicable_multi_person`. Their columns in `data/metric_scores.csv` stay blank. `Human_Anatomy` looks at each person and is `scored` in that table, so `vbench2_human_anatomy` is in the confirmatory set.

`dynamic_degree` is in the set. On this pilot it is 1 for every clip, so Kendall's tau-b is undefined until the column varies.

### Data Preparation

1. Clip registry / manifest: `video_id`, path, optional `source_id`, duration, language, permissions.
2. Metric scores: one row per clip with `video_id` + numeric dimension columns (0–1 typical range). Missing/inapplicable scores remain missing (not zero).
3. Human ratings: one row per viewer–clip pair (`video_id`, `rater_id`, `enjoyment` [1–5]). Optional free-text reasons unused for RQ1.
4. Aggregate: mean enjoyment per clip

\[
\bar{y}_i = \frac{1}{r_i} \sum_{u=1}^{r_i} y_{iu}
\]

where \( r_i \) is the number of ratings for clip \( i \). Report SD and number of raters. Exclude clips below a minimum rater threshold if prespecified.

### Primary Analysis

For each dimension \( j \):

\[
\tau_{b,j} = \frac{P_j - Q_j}{\sqrt{(P_j + Q_j + T_j)(P_j + Q_j + U_j)}}
\]

(Kendall's tau-b). \(P_j\) and \(Q_j\) are the concordant and discordant clip pairs for dimension \(j\). \(T_j\) counts pairs tied only on the dimension score, and \(U_j\) counts pairs tied only on mean enjoyment. Pairs tied on both are in neither count. This is the tie-adjusted coefficient ([SciPy `kendalltau`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html), `variant='b'`, the default). With no ties it equals Kendall's tau-a, \((P - Q) / \binom{n}{2}\). Positive \(\tau_b\) means higher dimension scores tend to accompany higher enjoyment.

**Uncertainty and inference:**

- Bootstrap 95% percentile confidence intervals. Resample paired clip rows with replacement and recalculate \(\tau_b\). A replicate that makes either column constant is undefined: count it, leave it out of the percentiles, and warn. Report the interval only when at least 100 replicates are defined; otherwise the limits are unavailable and the notebook warns. Intervals condition on the observed viewer panel and are not simultaneous family-wide intervals.
- Two-sided permutation tests of no association. Hold the dimension scores fixed and permute mean enjoyment. The test statistic is absolute \(\tau_b\). When the complete-case count is 7 or fewer, enumerate every enjoyment permutation (`7! = 5040`). For larger counts, use Monte Carlo. The Monte Carlo p-value is \((1 + \#\{|\tau| \ge |\tau_{\mathrm{obs}}|\}) / (1 + \text{permutations})\) and is never zero. The default is 4999 permutations and seed 42. Both are configurable, as is the bootstrap replicate count.
- Holm correction across the scored dimensions in `data/dimension_applicability.csv` (the six VBench 1.0 columns plus `vbench2_human_anatomy`), \(\alpha = 0.05\). A prespecified dimension that is untestable is entered as \(p = 1\) so the family is not shrunk. Its coefficient, raw p-value, and adjusted p-value are reported as unavailable, with a status, not as zero. Identity, clothes, Diversity, and Multi-View Consistency are not in this family.
- Report eligible clips, complete cases, unique values, \(\tau_b\), 95% CI, raw and Holm-adjusted permutation p-values, whether the permutation result is exact or Monte Carlo, bootstrap valid and undefined counts, and missingness per dimension.

**Sensitivity:** Repeat \(\tau_b\) with median enjoyment instead of the mean. That check is not a second confirmatory test. Spearman is not estimated. Tau-b and Spearman use different scales and are not compared.

**Interpretation notes:**

- \(\tau_b\) describes association within the observed clips, not causal effect of the dimension.
- Close \(\tau_b\) values should not be over-ranked.
- Largest \( |\tau_b| \) does not by itself prove one dimension is stronger than others in the population.
- More viewers improve the estimate of each clip’s mean enjoyment but do not increase the number of independent clip-level observations.
- The pairs counted by \(\tau_b\) are not additional independent clips. A constant dimension, noisy ranks, or a restricted score range are not fixed by the choice of coefficient.

**Optional extensions (not required for primary RQ1):**

- Ordinal mixed-effects regression to examine conditional associations while accounting for viewer and video random effects.
- Predictive evaluation via cross-validation that holds out entire videos.

**Implementation note:** `notebooks/rq1-aidrama.ipynb` loads the CSVs, aggregates enjoyment, and computes Kendall's tau-b with `scipy.stats.kendalltau` (`variant='b'`), plus the bootstrap, permutation test, Holm correction, and median sensitivity above. Scoring is a separate notebook. It writes VBench / VBench-2.0 custom-input scores to `metric_scores.csv`. The association notebook does not score clips. It does not report a coefficient while enjoyment ratings or real drama IDs are still blocked.

## RQ2 Methodology

**Goal:** Quantify rating alignment and reasoning (aspect + claim) alignment between one independent LMM judge and multiple human viewers per clip. Report the two components separately.

### End-to-End Process

| Stage | Input | Output to store | Purpose |
|-------|-------|-----------------|---------|
| 0. Clip registry | Source clips + metadata | `video_id`, source, duration, language, permissions | Reproducible stimuli |
| 1. Human evaluations | Video + shared 1–5 question | `human_id`, `video_id`, rating, explanation | Reference judgments |
| 2. LMM assessment | Same clip + fixed neutral prompt | `video_id`, model, rating, explanation, settings | Independent automated prediction |
| 3. Rating preparation | Ratings for same video | human_mean, human_SD, LMM_rating, abs_error | One paired rating per clip |
| 4. Rating analysis | Clip-level pairs | MAE, Kendall's \(\tau_b\), uncertainty | Numerical alignment |
| 5. Aspect extraction | Each full explanation | Claim list (aspect, stance, enjoyment effect) | Makes statements comparable |
| 6. Aspect matching | Human/LMM aspects for same clip | matched_aspects, coverage per human–LMM pair | Reasons LMM did/did not mention |
| 7. NLI classification | Matched claim pairs | entailment / contradiction / neutral (+ scores) | Compatibility of claims |
| 8. Aggregation + review | Pair-level results | Per-video and study-level summaries + human baseline | Reasoning alignment |

### Rating Alignment (Primary Metrics)

**Mean Absolute Error (MAE)**

\[
\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} |A_i - \bar{H}_i|
\]

where \( A_i \) is the LMM rating and \( \bar{H}_i \) is the human mean for clip \( i \). Lower is closer. Report in rating points (1–5 scale).

**Kendall's tau-b** between the vector of human means and the vector of LMM ratings across clips. Report \(\tau_b\), 95% CI, and p-value. Higher positive \(\tau_b\) indicates more similar ordering (not necessarily matching absolute scores). Use the same tie-adjusted definition as RQ1.

Optional normalized closeness (rescaled absolute error) may be reported but is not an independent finding. Compare against human–human variability and a naive baseline (e.g., always predict the training-set mean) before claiming judge replacement.

### Reasoning Alignment (Primary Metrics)

**Aspect extraction**

For each human and LMM explanation, extract claims using a small documented taxonomy (e.g., narrative, dialogue, animation, humor, pacing, emotion, audio). Map synonyms (storyline ↔ narrative). Distinguish:

- How the aspect is described.
- Effect on enjoyment (positive / negative / unspecified / mixed).

Do not silently infer enjoyment effects. Apply identical rules to human and LMM text. Validate a subset with human coders. Extraction may be manual or assisted, but rules must be consistent.

**Aspect coverage (human-referenced recall)**

For each human–LMM pair on the same clip:

\[
\text{Coverage} = \frac{|A_H \cap A_L|}{|A_H|}
\]

where \( A_H \) = distinct aspects mentioned by the human and \( A_L \) = distinct aspects mentioned by the LMM.

- Measures what the LMM mentions relative to the human, not agreement about those aspects.
- Coverage = 0 when human aspects exist but none match.
- Coverage is undefined (null) if the human supplies no identifiable aspect (report count of undefined comparisons).
- Optional: also report precision (shared / all LMM aspects) to detect excessive listing.
- Coverage is asymmetric; LMM-only aspects do not increase recall.

**Natural Language Inference (NLI) on matched claims**

For claims linked to the *same matched aspect*, run NLI with:

- Premise = human claim
- Hypothesis = LMM claim

Standard labels: entailment, contradiction, neutral. Keep model scores and exact label mappings.

- Entailment is directional; for two-way agreement consider both directions or a predefined human-validated compatibility rubric.
- Do not conflate technical-quality claims with enjoyment-effect claims.
- NLI can fail on negation, sarcasm, irony, or “bad but funny” cases. Manually validate a sample (including hard cases) and report disagreements.

**Agreement rule (define in advance)**

Example: a matched claim pair counts as agreement only if judged mutually compatible under the validated rule. Contradiction and neutral do not count as agreement but are reported separately.

For a human–LMM pair with \( k \) matched claims:

\[
\text{pair_agreement} = \frac{\text{number of agreeing matched claims}}{k}
\]

If \( k = 0 \), the rate is undefined (null).

### Aggregation

- Per human–LMM pair: coverage, NLI agreement rate (or null), matched-claim count.
- Per video: mean coverage across humans, mean NLI agreement across humans (defined scores only), human mean/SD, LMM rating, absolute error, counts of undefined cases.
- Study-level: mean of video-level coverage values, mean of video-level NLI agreement rates, overall MAE, overall Kendall's \(\tau_b\), with clip-level confidence intervals and full sample-size reporting at each stage.

Five comparisons from one clip are not five independent videos.

Also report human–human baselines for both rating agreement and aspect coverage / claim compatibility where feasible.

### Interpretation Guidance for RQ2

Report rating metrics and reasoning metrics side-by-side. Example patterns to investigate:

- Low rating error + high coverage/NLI agreement → similar scores and similar considerations.
- Low rating error + low coverage/NLI agreement → similar scores but different explanations.
- High rating error + high coverage/NLI agreement → similar considerations but different weighting or evaluation.
- High rating error + low coverage/NLI agreement → divergence on both score and explanation.

Suggested summary phrasing: “The LMM showed [observed rank association] and [observed rating error]. Its explanations mentioned [observed coverage] of viewer-identified aspects, and [observed agreement] of comparable claims were compatible under the defined rule. These measurements jointly describe rating and stated-reason alignment but do not prove that an LMM can replace human viewers.”

**Note on earlier embedding ideas:** Cosine similarity of embeddings is not part of the primary methodology. Aspect coverage + NLI more directly address whether the LMM identifies the same factors and agrees about them. Cosine similarity may be retained only as an optional exploratory supplement.

### Suggested Deliverable Files

- `clips.csv` — one row per video: `video_id`, source, duration, language, …
- `human_evaluations.csv` — one row per human–video: `video_id`, `human_id`, rating, explanation
- `lmm_evaluations.csv` — one row per LMM–video: `video_id`, model_version, rating, explanation, settings
- `claims.csv` — one row per extracted claim: evaluator, `video_id`, aspect, claim_text, enjoyment_effect
- `pair_results.csv` — one row per human–LMM–video pair: coverage, agreement, matched_count, undefined_flag
- `video_summary.csv` — one row per video: human_mean, human_sd, lmm_rating, abs_error, mean_coverage, mean_nli, …
- `rq2_summary.json` — overall MAE, \(\tau_b\), coverage, NLI agreement, uncertainty, sample counts

## Shared Reporting Checklist

- Clip and viewer counts; source sampling strategy.
- Exact scale anchors and prompt text (human and LMM).
- Metric / model versions, configurations, and repository commits.
- Chosen dimensions or taxonomy + rationale.
- Missingness, exclusions, and undefined cases at each stage.
- Per-dimension or per-component sample sizes, point estimates, CIs, adjusted p-values where applicable.
- Human–human baselines.
- Limitations: clip independence, viewer sampling, NLI reliability, extraction validation, lack of causal claims.
- Associations/alignment are not causal effects. Nonsignificance is not evidence of no association.

## Limitations and Scope Notes

- Design measures association/alignment, not experimental effects of changing a dimension or prompt.
- Clip-level bootstrap intervals condition on the observed viewer panel.
- NLI and aspect extraction require human validation; automated steps alone are insufficient for strong claims.
- VBench custom-input covers only a subset of dimensions; results are not full leaderboard evaluations.
- `Human_Identity` and `Human_Clothes` score one person. Identity keeps a clip only when the opening frame shows exactly one face and at least 20 frames do. Clothes asks whether only one person is present throughout and leaves the clip unscored otherwise. This multi-person pilot excludes both. `data/metric_scores.csv` leaves those columns blank, and `data/dimension_applicability.csv` marks them `not_applicable_multi_person`. `Human_Anatomy` looks at each person in a frame. The scoring notebook installs the CUDA `mmcv` wheel for the current Colab PyTorch and writes `vbench2_human_anatomy`. That column is scored and is in the confirmatory association set.
- Precedent exists for testing automated enjoyment prediction against human references (e.g., Marcinek et al., 2025, SIGDIAL), but aspect coverage + claim-level NLI are additional methods proposed here and must be validated in this study.

## Pilot Status (November 2026)

The six files `dataset/drama_clip_01.mp4` through `drama_clip_06.mp4` are compiled pilot clips, not stand-in videos and not the study sample. The planned scale remains about 30 clips and about 5 viewers per clip. Scores from this pilot are a pipeline check. They are not RQ1 results.

- Every clip is `source_id = unlabeled_pilot`. Drama grouping is unknown, so these clips are not an independent set. Confirmatory clip-level Kendall's tau-b stays off until real drama IDs exist, or the analysis is cluster-aware.
- Six VBench 1.0 dimensions were scored and written to `data/metric_scores.csv`, with `data/clip_manifest.csv` and `data/dimension_applicability.csv`. `dynamic_degree` is 1 for every clip, so tau-b is undefined for that column. Imaging quality is a per-clip MUSIQ mean (about 50–74), not a 0–1 score. Tau-b is still valid when a column varies, including columns with ties.
- `vbench2_human_identity` and `vbench2_human_clothes` are blank. Both dimensions are `not_applicable_multi_person` in `data/dimension_applicability.csv` because these clips show several people. `vbench2_human_anatomy` is scored. The applicability table marks it `scored`, and the association family includes that column. The scoring notebook leaves a column unchanged once every clip already has a score.
- Human enjoyment is not in the files. Ratings will be entered by hand from Google Docs into `data/human_ratings.csv`. Do not invent placeholder ratings. `notebooks/rq1-aidrama.ipynb` runs confirmatory association only after both real ratings and real drama IDs are present. Scoring lives in `notebooks/vbench_score_clips.ipynb`.

## Implementation Status Notes

- RQ1 association notebook (`notebooks/rq1-aidrama.ipynb`) computes Kendall's tau-b with `scipy.stats.kendalltau` (`variant='b'`), bootstrap, permutation, Holm, and median sensitivity. It loads `data/metric_scores.csv` and runs confirmatory tests only when real enjoyment ratings and real drama IDs are present. The unlabeled pilot does not produce those numbers.
- RQ1 scoring notebook (`notebooks/vbench_score_clips.ipynb`) builds `data/clip_manifest.csv` and official custom-input scores. The local 16 GB laptop run produced empty scores. The six VBench 1.0 columns above came from a Colab A100 run. The same notebook now also scores `vbench2_human_anatomy` on that runtime.
- RQ2 requires collection pipelines for human and LMM evaluations, aspect extraction (manual or assisted), coverage calculation, NLI inference, and multi-level aggregation.
- With this six-clip pilot, treat numeric output as a pipeline test only. Skip inferential claims.

This document can be extended with exact prompts, taxonomy definitions, annotation guidelines, model identifiers, and finalized dimension lists as the study progresses.
