# RQ1 findings

## 1. Analysis sample

Twenty-nine viewers rated each of clips 01–10 on a 1–5 enjoyment scale. The analysis retained all 290 ratings, including 30 partial watches. Clip 11 received no survey ratings and was excluded from the analysis. The association notebook used the saved VBench and VBench-2.0 scores and compared one score per metric and one mean enjoyment value per clip. [Survey ratings](../data/human_ratings.csv) · [RQ1 summary](rq1_summary.md)

## 2. Results

The six tested dimensions each used 10 complete clips, 4,999 Monte Carlo permutations, and 2,000 defined bootstrap replicates. All six 95% intervals included zero. [Results CSV](../rq1_outputs/rq1_kendall_results.csv)

| Metric | Kendall’s $\tau_b$ | 95% bootstrap CI | Permutation $p$ | Holm-adjusted $p$ |
| --- | ---: | --- | ---: | ---: |
| Subject consistency | −0.09 | [−0.66, +0.46] | 0.8040 | 1.00 |
| Background consistency | +0.09 | [−0.44, +0.62] | 0.7834 | 1.00 |
| Motion smoothness | −0.13 | [−0.79, +0.40] | 0.6670 | 1.00 |
| Dynamic degree | — | — | — | — |
| Aesthetic quality | +0.27 | [−0.41, +0.79] | 0.3176 | 1.00 |
| Imaging quality | +0.22 | [−0.42, +0.81] | 0.4336 | 1.00 |
| Human anatomy | −0.09 | [−0.70, +0.46] | 0.7868 | 1.00 |

Dynamic degree was undefined because every clip scored 1.0. The median sensitivity check was undefined because every rated clip had median enjoyment of 3.

The intervals describe uncertainty across clips for the participating viewers and assume the clips are independent. [RQ1 conclusion](rq1_conclusion.md)
