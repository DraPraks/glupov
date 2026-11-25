# RQ1 summary

## 1. Research question and study design

How strongly are scores on selected VBench and VBench-2.0 dimensions associated with viewer enjoyment of AI-generated drama clips?

The class study pairs automated scores with mean enjoyment on a 1–5 scale. Each clip contributes one observation per metric. Kendall’s tau-b measures how closely the metric scores and enjoyment rank the clips.

## 2. Clip scoring and survey data

### Clip scoring

```mermaid
flowchart TD
    clips["Drama clips in dataset/"] --> registry["Register clips by video_id"]
    registry --> manifest["clip_manifest.csv"]
    registry --> scoring["Assemble VBench and VBench-2.0 scores"]
    saved["Complete score columns from earlier runs"] --> scoring
    scoring --> metrics["metric_scores.csv with 11 clips and 7 score columns"]
    scoring --> applicability["dimension_applicability.csv"]
```

The scoring notebook reads the videos in `dataset/` and records each clip’s `video_id`, file path, duration, and `source_id` in the clip manifest. It uses custom-input mode and writes one row of scores per clip. It reuses saved columns that already have a score for every clip. Missing scores stay blank. [Scoring notebook](../notebooks/vbench_score_clips.ipynb) · [Clip manifest](../data/clip_manifest.csv) · [Metric scores CSV](../data/metric_scores.csv)

The applicability table marks seven dimensions as `scored`. Six come from VBench 1.0 (subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality). The seventh is VBench-2.0 human anatomy. Human identity and clothing require single-person clips. Diversity requires multiple clips for the same prompt, and multi-view consistency requires multiple views. [Applicability table](../data/dimension_applicability.csv)

Write the score columns as a matrix $M=(M_{ij})$ with $N$ rows for scored clips and $d$ columns for metric dimensions. Use `video_id` as the row key and `source_id` as metadata. The study has $N=11$ and $d=7$, with a score in every cell, so $M\in\mathbb R^{11\times7}$.

The recorded Colab run used an **NVIDIA A100-SXM4 80 GB GPU** and **about 170 GiB of system RAM**. The notebook logged 79.3 GiB of GPU memory and 167.1 GiB of system RAM. This run reused the six saved VBench 1.0 columns and scored VBench-2.0 human anatomy. [Scoring notebook](../notebooks/vbench_score_clips.ipynb)

### Survey ratings

The anonymized survey file contains one row per viewer–clip pair, with `video_id`, `rater_id`, integer `enjoyment` from 1 to 5, and a `partial_watch` flag. The study retained all 290 ratings from 29 viewers, including 30 partial watches. Clips 01–10 each received 29 ratings. Clip 11 was excluded from the analysis because it received no survey ratings. [Anonymized survey CSV](../data/human_ratings.csv)

## 3. From data to associations

```mermaid
flowchart TD
    ratings["human_ratings.csv"] --> summary["Validate and aggregate available ratings by clip"]
    summary --> joined["Join by video_id and keep clips with at least 3 ratings"]
    metrics["Read score matrix M from metric_scores.csv"] --> joined
    joined --> sources["Check source labels on eligible clips"]
    sources --> pairs["Pair each column of X with enjoyment on complete rows"]
    applicability["dimension_applicability.csv"] --> pairs
    pairs --> tau["Calculate Kendall's tau-b with mean enjoyment"]
    tau --> inference["Bootstrap intervals and calculate permutation p-values"]
    inference --> holm["Apply Holm correction"]
    tau --> median["Repeat tau-b with median enjoyment"]
    holm --> results["rq1_kendall_results.csv"]
    median --> results
```

### Align scores and enjoyment

The association notebook validates the ratings, drops rows with missing enjoyment, and calculates mean enjoyment, median enjoyment, standard deviation, and rater count for each clip. It joins these summaries to the metric scores by `video_id` and keeps clips with at least three ratings. Let $X$ be the submatrix of $M$ for the $n$ eligible clips, ordered to match the rating summaries. Here $n=10$ and $X\in\mathbb R^{10\times7}$. [Association notebook](../notebooks/rq1-aidrama.ipynb)

For clip $i$, let $y_{iu}$ be rating $u$ and $r_i$ be the number of valid ratings. The clip-level mean $\bar y_i$, sample standard deviation $s_i$, and median $\tilde y_i$ are

$$
\begin{aligned}
\bar y_i &= \frac{1}{r_i}\sum_{u=1}^{r_i} y_{iu}, \\
s_i &= \sqrt{\frac{1}{r_i-1}\sum_{u=1}^{r_i}(y_{iu}-\bar y_i)^2}, \\
\tilde y_i &= \operatorname{median}(y_{i1},\ldots,y_{ir_i}).
\end{aligned}
$$

All $r=29$ viewers rated each eligible clip in this study, so the ratings form a matrix $Y=(y_{iu})\in\{1,2,3,4,5\}^{n\times r}$. The row-mean function yields the enjoyment vector $\bar{\mathbf y}\in\mathbb R^{10}$.

$$
A:\{1,2,3,4,5\}^{n\times r}\to\mathbb R^n,
\qquad
A(Y)=\frac{1}{r}Y\mathbf{1}_r=\bar{\mathbf y},
$$

where $\mathbf{1}_r$ is the length-$r$ vector of ones.

### Kendall’s tau-b

For dimension $j$, restrict column $X_{\cdot j}$ and the enjoyment vector $\bar{\mathbf y}$ to the same $n_j$ complete clip rows. Call the resulting vectors $\mathbf x_j$ and $\mathbf y_j$. Here $n_j=10$ for all seven dimensions. The notebook requires $n_j\ge4$ and variation in both vectors. On the subset $\mathcal D_{n_j}\subseteq\mathbb R^{n_j}\times\mathbb R^{n_j}$ satisfying those conditions, tau-b returns a single coefficient.

$$
\tau_b:\mathcal D_{n_j}\to[-1,1],
\qquad
\tau_{b,j}=\tau_b(\mathbf x_j,\mathbf y_j) =
\frac{P_j-Q_j}
{\sqrt{(P_j+Q_j+T_j)(P_j+Q_j+U_j)}}.
$$

$P_j$ and $Q_j$ count concordant and discordant clip pairs. $T_j$ counts pairs tied only on the metric score, and $U_j$ counts pairs tied only on mean enjoyment. Pairs tied on both are excluded from both counts. Positive tau-b means higher-scoring clips tended to have higher mean enjoyment. For each tested metric, the sensitivity check computes $\tau_b(\mathbf x_j,\tilde{\mathbf y}_j)$, where $\tilde{\mathbf y}_j$ holds median enjoyment for those same clips.

### Bootstrap confidence interval

For each replicate $\ell$, draw $n_j$ row indices with replacement. Represent that draw by a selection matrix $S_\ell\in\{0,1\}^{n_j\times n_j}$ with one $1$ per row. Apply it to both vectors so the clip pairs stay together, then compute $\tau_b(S_\ell\mathbf x_j,S_\ell\mathbf y_j)$. Let $\mathcal B_j$ be the list of defined replicate estimates. The 95% percentile interval is

$$
\left[Q_{0.025}(\mathcal B_j),\ Q_{0.975}(\mathcal B_j)\right],
$$

where $Q_q$ is the quantile at probability $q$. The notebook attempts 2,000 replicates by default and requires at least 100 defined estimates to report the interval. A replicate with a constant metric or enjoyment column is undefined.

### Permutation test and Holm correction

Each shuffle maps $\mathbb R^{n_j}$ to $\mathbb R^{n_j}$ by applying a permutation matrix $\Pi$ to the enjoyment vector. Keep $\mathbf x_j$ fixed and compute $\tau_b(\mathbf x_j,\Pi\mathbf y_j)$. Let $E_j$ count permutations whose absolute tau-b is at least as large as the observed absolute tau-b, with a tolerance of $10^{-12}$ for numerical equality. The test uses all $n_j!$ permutations when $n_j\le 7$ and uses $B=4{,}999$ Monte Carlo permutations by default for larger samples, including this study.

$$
p_{\mathrm{exact},j} = \frac{E_j}{n_j!},
\qquad
p_{\mathrm{MC},j} = \frac{1+E_j}{1+B}.
$$

The notebook uses base seed 42 by default and adds each metric’s zero-based index. Holm correction maps $[0,1]^d$ to $[0,1]^d$, with one p-value for each of the $d=7$ scored dimensions. Use $p=1$ for an untestable dimension when forming the input vector. Sort its values as $p_{(1)} \le \cdots \le p_{(d)}$. The adjusted p-value for sorted rank $k$ is

$$
p^{\mathrm{Holm}}_{(k)} =
\min\left\{1,\ \max_{1\le \ell\le k} (d-\ell+1)p_{(\ell)}\right\}.
$$

Map the adjusted p-values back to the original metric order and compare each testable dimension’s value with $\alpha=0.05$. An untestable dimension has blank estimates and p-values in the output.

## 4. Results

All seven dimensions had 10 complete clips. Each of the six testable dimensions had a Holm-adjusted p-value of 1.00 and a 95% bootstrap confidence interval that included zero. [Results CSV](../rq1_outputs/rq1_kendall_results.csv)

| Metric | Complete clips | Kendall’s $\tau_b$ | 95% bootstrap CI | Holm-adjusted $p$ |
| --- | ---: | ---: | --- | ---: |
| Subject consistency | 10 | −0.09 | [−0.66, +0.46] | 1.00 |
| Background consistency | 10 | +0.09 | [−0.44, +0.62] | 1.00 |
| Motion smoothness | 10 | −0.13 | [−0.79, +0.40] | 1.00 |
| Dynamic degree | 10 | — | — | — |
| Aesthetic quality | 10 | +0.27 | [−0.41, +0.79] | 1.00 |
| Imaging quality | 10 | +0.22 | [−0.42, +0.81] | 1.00 |
| Human anatomy | 10 | −0.09 | [−0.70, +0.46] | 1.00 |

Dynamic degree was constant at 1.0, so its association was undefined. Every clip had a median enjoyment of 3, which made the median sensitivity check undefined for each tested dimension.

## 5. Interpretation and conclusion

None of the tested dimensions reached statistical significance after Holm correction. Aesthetic quality and imaging quality had the largest positive estimates. The intervals remain compatible with both positive and negative associations, so the direction and strength of the relationships remain uncertain.

The 290 ratings were used to estimate enjoyment for 10 clips. The bootstrap intervals reflect uncertainty across clips for this viewer panel. The notebook checks for missing or repeated source IDs and a provisional source label. The distinct placeholder IDs in this study pass those checks, leaving clip independence as an assumption. [Clip manifest](../data/clip_manifest.csv) · [Association notebook](../notebooks/rq1-aidrama.ipynb)

## 6. Reproduce the analysis

1. Use the checked-in `metric_scores.csv`, `dimension_applicability.csv`, and `human_ratings.csv` in `data/`. The notebook uses `clip_manifest.csv` as a fallback for missing source IDs. [Data files](../data/README.md)
2. Run all cells of `rq1-aidrama.ipynb` from the repository root or `notebooks/` in a Python environment with NumPy, pandas, SciPy, and statsmodels. [Association notebook](../notebooks/rq1-aidrama.ipynb)
3. Read `rq1_outputs/rq1_kendall_results.csv` for complete-case counts, missingness, estimates, raw and adjusted p-values, and bootstrap diagnostics. Set `RQ1_OUT_DIR` before running to write the rerun to a separate directory. [Results CSV](../rq1_outputs/rq1_kendall_results.csv)

To score new clips, run `vbench_score_clips.ipynb` and extract its `rq1_vbench_csvs.zip` into `data/`. The notebook pins the benchmark checkout to `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`. [Scoring notebook](../notebooks/vbench_score_clips.ipynb)
