# RQ1 findings: survey enjoyment and VBench scores

No VBench dimension had a statistically significant association with viewer enjoyment after Holm correction at the 5% level. All six tested dimensions had adjusted p-values of 1.00, and all six 95% bootstrap intervals included zero.

## What we analyzed
29 respondents rated each of 10 study clips on a 1–5 enjoyment scale. All 290 ratings were kept, including 30 from partial watches. Clip 11 had VBench scores but no survey ratings, so it was excluded.

The rerun used the saved VBench scores. We compared each metric’s clip rankings with the rankings by average enjoyment using Kendall’s τ-b. A positive value means higher-scoring clips tended to be enjoyed more.

## Results
Kendall’s τ-b (95% bootstrap interval):

- Aesthetic quality: +0.27 (−0.41 to +0.79)
- Imaging quality: +0.22 (−0.42 to +0.81)
- Background consistency: +0.09 (−0.44 to +0.62)
- Subject consistency: −0.09 (−0.66 to +0.46)
- Human anatomy: −0.09 (−0.70 to +0.46)
- Motion smoothness: −0.13 (−0.79 to +0.40)

Dynamic degree could not be tested: every clip scored 1.0. The median-enjoyment check was also undefined: all 10 clips had the same median of 3.

The analysis has 10 clip-level observations, even though it uses 290 ratings. The intervals reflect uncertainty across clips for this viewer panel and assume the clips can be treated as independent.

[Full results CSV](../rq1_outputs/rq1_kendall_results.csv)
