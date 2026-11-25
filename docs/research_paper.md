# Do Automated Evaluations Capture Viewer Enjoyment of AI-Generated Drama?

## Abstract

Video-quality benchmarks measure properties such as motion smoothness and visual consistency. Whether these scores track what viewers enjoy in a drama is a separate empirical question. We studied 10 AI-generated microdrama clips rated by 29 viewers. We first compared seven VBench and VBench-2.0 dimensions with mean enjoyment. We then evaluated a video-language model (VLM) that watched each clip, rated its enjoyment, and explained what it liked and disliked. Across six testable benchmark dimensions, Kendall’s $\tau_b$ ranged from −0.13 to +0.27, and all 95% bootstrap intervals included zero. The VLM ranked clips similarly to viewers (Spearman’s $\rho=0.69$), although assigning every clip the scale midpoint gave lower mean absolute error (0.145 versus 0.393 points). A text judge found at least one matching reason in 25 of 451 scorable human–VLM comparisons (5.5%). On a score-stratified human check, exact agreement with the judge was 55.6% across 36 comparable cases. These results show how a model can track relative differences in enjoyment while offering explanations that seldom match an individual viewer’s stated reasons under a strict rubric.

## 1. Introduction

A viewer can enjoy a clip for the timing of a joke, the tension in an exchange, or the appearance of a character. A video-quality benchmark can measure aspects of the same clip, such as how consistently the character is rendered across frames. Both judgments concern the video, but they ask different questions. For automated evaluation to be informative about audience experience, we need to know how its scores relate to the judgments viewers actually make.

VBench evaluates specific dimensions of generated video and validates them against human preferences elicited for each dimension (Huang et al., 2024). VBench-2.0 extends this approach to capabilities that include human fidelity and physical plausibility (Zheng et al., 2025). We ask whether these metrics track how much viewers enjoy a short drama. A second approach is to ask a video-language model the same questions as a viewer. Its rating can then be compared with the panel’s mean, and its explanation can be compared with the reasons individual viewers gave.

We study two questions using the same clips and viewer panel.

- **RQ1.** How strongly are selected VBench and VBench-2.0 scores associated with mean viewer enjoyment across clips?
- **RQ2.** How closely do a VLM’s enjoyment ratings and its stated likes and dislikes align with viewers’ responses?

We measure how the automated evaluations order the clips, how closely the VLM estimates the panel’s ratings, and how often it gives a reason an individual viewer also gave. A model could recover the ordering while exaggerating differences between clips. It could also assign the panel’s exact mean to a clip while praising a feature that few viewers mentioned.

## 2. Related work

**Video-quality evaluation.** In VBench’s human validation, annotators compare videos on a designated feature. The benchmark then compares automated and human win ratios across generation models for that feature. VBench-2.0 also evaluates alignment within defined capabilities. Our unit of analysis is an individual drama clip, and the human response is a single overall enjoyment rating. A dimension can measure its intended visual property accurately even when clips with higher scores elicit little additional enjoyment. [Huang et al., 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) · [Zheng et al., 2025](https://arxiv.org/abs/2503.21755)

**Models as viewers and judges.** Marcinek et al. (2025) evaluated model estimates of enjoyment against human judgments in human–robot dialogue. Their results motivate specifying whose enjoyment serves as the reference, and at what level it is measured. Here the reference for a rating is the panel mean for a clip. The reference for an explanation is one viewer’s answer to one question about that clip. Comparing these free-text responses introduces a further measurement problem. Zheng et al. (2023) documented sensitivity to response position and verbosity in language-model judging. We therefore report a human check of the text judge alongside its agreement scores. [Marcinek et al., 2025](https://aclanthology.org/2025.sigdial-1.46/) · [Zheng et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

## 3. Methods

### 3.1 Clips and human judgments

We scored 11 AI-generated drama clips. The survey covered clips 01–10, which ran from 16 to 39 seconds. Each of 29 respondents rated every survey clip on a five-point enjoyment scale and was asked what they liked and disliked about it and why. The survey also asked whether the respondent watched the entire clip. We retained all 290 valid ratings, including 30 marked as partial watches. Twenty-two respondents were between 18 and 34 years old, and 25 reported having previously watched AI-generated short dramas or microdramas.

For the rated clips, let $Y\in\{1,\ldots,5\}^{10\times29}$ contain one row per clip and one column per viewer. The reference for both rating analyses was the vector of clip means, $\bar{\mathbf y}=Y\mathbf 1_{29}/29$. For the explanation analysis, we retained each individual response so that a match had to be supported by what a particular viewer said.

~~~mermaid
flowchart TD
    clips["10 survey clips"] --> humans["29 viewers rate each clip and give likes and dislikes"]
    clips --> metrics["VBench and VBench-2.0 scores"]
    clips --> viewer["VLM gives one rating and two explanations per clip"]
    humans --> means["Mean human enjoyment per clip"]
    means --> rq1["RQ1 · Metric–enjoyment rank association"]
    metrics --> rq1
    means --> ratings["RQ2 · Rating error and rank agreement"]
    viewer --> ratings
    humans --> reasons["RQ2 · Match reasons within each clip and question"]
    viewer --> reasons
    reasons --> review["Human check of text-judge scores"]
~~~

*Figure 1. Both rating analyses use the same ten human means. The reason comparison retains the distinction between individual viewers and between the liked and disliked questions.*

### 3.2 RQ1 — Video-quality scores and enjoyment

We used the benchmarks in custom-input mode. The six VBench dimensions were subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. VBench-2.0 supplied human anatomy. The selected dimensions apply to individual clips containing multiple characters. The recorded scoring run used an NVIDIA A100-SXM4 80 GB GPU and 167.1 GiB of system RAM visible to the notebook. It reused the six complete VBench score columns and computed human anatomy.

The full score matrix contained 11 clips and seven columns. Joining it to the survey by clip ID gave $X\in\mathbb R^{10\times7}$. For each dimension $j$, we compared $X_{\cdot j}$ with $\bar{\mathbf y}$ using Kendall’s tie-adjusted rank correlation,

$$
\tau_{b,j}=\frac{P_j-Q_j}{\sqrt{(P_j+Q_j+T_j)(P_j+Q_j+U_j)}}.
$$

Here $P_j$ and $Q_j$ count concordant and discordant clip pairs. $T_j$ and $U_j$ count pairs tied only on the metric and only on enjoyment, respectively. We estimated a 95% percentile interval from 2,000 paired-clip bootstrap resamples. For each testable metric, we shuffled the ten enjoyment means 4,999 times and used absolute $\tau_b$ for a two-sided permutation test. We applied Holm correction at $\alpha=0.05$ across the seven selected dimensions. An undefined test contributed $p=1$ internally to the correction. We also repeated the coefficient calculation using median enjoyment as a sensitivity check.

### 3.3 RQ2 — Model ratings and stated reasons

**Rating comparison.** The viewer prompt asked the VLM to evaluate each clip on its own, give an integer rating from 1 (not enjoyable at all) to 5 (extremely enjoyable), and explain one like and one dislike in one or two sentences each. The archived responses do not identify the viewer model or its version. Let $a_i$ denote its rating for clip $i$. We measured absolute error in rating points and agreement in the ordering of clips using Spearman’s rank correlation. Tied values received their average rank.

$$
\mathrm{MAE}=\frac{1}{10}\sum_{i=1}^{10}|a_i-\bar y_i|,
\qquad
\rho_s=\operatorname{corr}\!\left(\operatorname{rank}(\mathbf a),\operatorname{rank}(\bar{\mathbf y})\right).
$$

We report $\rho_s$ descriptively. As a reference for interpreting MAE on this scale, we also calculated the error of assigning every clip the fixed midpoint of 3. This reference requires no fitting.

**Reason comparison.** A text judge (Grok 4.7, temperature 0) received the human and VLM responses for one viewer–clip pair. Each of the 290 requests contained both questions. The judge scored likes and dislikes independently, producing 580 question-level judgments. Responses were labeled A and B, with the human consistently in position A. When a viewer referred back to an earlier answer, the pipeline supplied context for the corresponding question where applicable (21 question sides).

The judge extracted distinct evaluative reasons from each answer. A match required the same feature, a substantively equivalent judgment, and the same evaluative direction. Reasons were matched one-to-one. If the two answers contained $n_A$ and $n_B$ reasons and shared $m$ matches, their respective coverage values were $m/n_A$ and $m/n_B$. Blank answers, missing-value markers, or answers that supplied no substantive reason yielded an unscorable comparison. For scorable comparisons, the rubric assigned the following ordinal scores.

| Score | Agreement criterion |
| ---: | --- |
| 0 | No matched reason. Related topics or conflicting evaluations can occur at this score. |
| 1 | At least one match, with coverage below half for either response. |
| 2 | Coverage reaches at least half on both sides, with exactly half on at least one side or a direct conflict. |
| 3 | Coverage exceeds half on both sides, with at least one unmatched reason and no direct conflict. |
| 4 | Every reason matches on both sides, with no direct conflict. |

We summarize the distribution by question type. Counts describe individual comparisons, which share viewers, clips, and the VLM’s fixed explanation for each clip. For validation, the study sampled 40 originally scorable judgments from the saved judge key, stratified by score (20 zeros, 10 ones, and 10 twos). Human review assessed scorability and assigned an ordinal score to accepted cases. We report exact agreement, agreement within one point, mean absolute error on the 0–4 scale, and quadratic-weighted Cohen’s $\kappa$ for the comparable scores.

## 4. Results

### 4.1 RQ1 — Visual metrics and enjoyment

None of the six testable dimensions reached significance after Holm correction (Table 1). Aesthetic quality and imaging quality had the largest positive estimates, at +0.27 and +0.22. Their intervals also included negative associations. Dynamic degree scored 1.0 for every clip, leaving no ordering to compare. All ten clips had median enjoyment of 3, so the median-based sensitivity check was likewise undefined.

*Table 1. Association with mean enjoyment across ten clips. Intervals are per-metric 95% bootstrap intervals. Dashes indicate an undefined test.*

| Dimension | Kendall’s $\tau_b$ | 95% interval | Permutation $p$ | Holm-adjusted $p$ |
| --- | ---: | --- | ---: | ---: |
| Subject consistency | −0.09 | [−0.66, +0.46] | 0.8040 | 1.00 |
| Background consistency | +0.09 | [−0.44, +0.62] | 0.7834 | 1.00 |
| Motion smoothness | −0.13 | [−0.79, +0.40] | 0.6670 | 1.00 |
| Dynamic degree | — | — | — | — |
| Aesthetic quality | +0.27 | [−0.41, +0.79] | 0.3176 | 1.00 |
| Imaging quality | +0.22 | [−0.42, +0.81] | 0.4336 | 1.00 |
| Human anatomy | −0.09 | [−0.70, +0.46] | 0.7868 | 1.00 |

### 4.2 RQ2 — Agreement on ratings

The VLM’s ratings had a positive rank association with mean human enjoyment ($\rho_s=0.690$). Its MAE was 0.393 points. Human means occupied a narrow band from 2.83 to 3.34. The model assigned two clips a 2, six clips a 3, and two clips a 4. Its largest error was on clip 04, which it rated 2 against a viewer mean of 2.93.

The fixed-midpoint reference had an MAE of 0.145 points. Thus the VLM captured some of the ordering, while its departures from the midpoint were larger than the differences among panel means. A constant rating has undefined rank correlation because it assigns all clips the same rank.

### 4.3 RQ2 — Agreement on reasons

Of 580 question-level comparisons, 451 contained substantive reasons on both sides. The judge identified at least one match in 12 of 244 scorable likes (4.9%) and 13 of 207 scorable dislikes (6.3%). All 20 clip-by-question groups had a median score of zero. The judge assigned no scores of 3 or 4 in the full run.

*Table 2. Distribution of reason-alignment scores. The match percentage uses only scorable comparisons. Unscorable responses are reported separately.*

| Question | Attempted | Unscorable | Scorable | Score 0 | Score 1 | Score 2 | Scores 3–4 | Any match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Liked | 290 | 46 | 244 | 232 | 4 | 8 | 0 | 4.9% |
| Disliked | 290 | 83 | 207 | 194 | 5 | 8 | 0 | 6.3% |
| Total | 580 | 129 | 451 | 426 | 9 | 16 | 0 | 5.5% |

Clip 07 illustrates the distinction between the two RQ2 outcomes. The VLM rated it 3, exactly the panel mean, yet none of the 23 scorable liked responses matched its praise under the rubric. One viewer praised the appearance of a character’s eyes. The VLM emphasized an emotional reversal, the lighting, and the framing. Clip 02 provides a positive example. A viewer and the VLM both criticized the lack of narrative context. The viewer also mentioned missing speech, while the VLM criticized choppy editing. The judge assigned partial alignment (2) for the shared context criticism.

### 4.4 Human validation of the reason scores

Human review accepted 36 of the 40 sampled cases as scorable. Among those 36, human and judge scores agreed exactly in 20 cases (55.6%) and were within one point in 30 (83.3%). Mean absolute difference was 0.67 points and quadratic-weighted $\kappa$ was 0.367. In two cases the human reviewer assigned complete agreement (4) to a pair that the saved judge key scored 1. This discrepancy matters when interpreting the absence of strong matches in Table 2.

The validation sample deliberately overrepresents nonzero scores and contains only cases originally deemed scorable. Its agreement rates describe that sample. The latest validation export compares human judgments with its archived key. One sampled dislike for clip 05 has score 1 in that key and 0 in the final export. Table 2 uses the final export throughout.

## 5. Discussion

**Ordering clips and estimating enjoyment are different tasks.** All ten panel means were within 0.35 points of the midpoint. In that setting, a model can place more-enjoyed clips above less-enjoyed clips and still incur a larger absolute error than a constant response. Reporting rank correlation alone would conceal that difference. The RQ1 estimates answer a narrower question about each visual dimension. With ten observations and wide intervals, the data leave substantial uncertainty about the direction and size of most associations. The two RQs also use different rank coefficients, so their numerical magnitudes do not form a common performance scale.

**An explanation adds an observable test of alignment.** A single score allows many routes to agreement. Two viewers can enjoy a scene equally while attending to its dialogue, visual design, or plot. The clip 07 example makes that ambiguity concrete. Under the study rubric, a zero means that the judge found no equivalent stated evaluation of a feature. Silence about a feature, a different reason for liking it, and an explicit disagreement can all yield zero. The observed 5.5% match rate therefore measures how often this judge identified shared reasons under this rule. Human review shows that the coding decision itself is a consequential source of uncertainty.

**Limits of the evidence.** The 29 ratings per clip help estimate panel means, while the number of distinct rated clips remains ten. Bootstrap intervals condition on this viewer panel. Thirty ratings followed partial watches, and each clip has a single VLM response. The design gives limited evidence about variation across model runs, audience populations, or a wider range of drama. The fixed A/B order leaves possible position effects unmeasured. A human–human reason comparison on the same clips would help distinguish model-specific divergence from the diversity of reasons people themselves supply. Further human review of both scorable and unscorable cases would also permit a broader assessment of the rubric.

## 6. Conclusion

For these ten AI-generated drama clips, the selected visual benchmark dimensions had uncertain associations with mean viewer enjoyment. The VLM showed positive rank agreement, while a fixed midpoint estimated the panel means more closely in rating points. Its stated reasons rarely received a match with an individual viewer’s reasons under the automated rubric, and human review exposed meaningful disagreement about those matches. Whether a model can stand in for a viewer therefore depends on whether the evaluation needs a ranking, a rating in points, or an account of what the viewer valued.

## References

1. Huang, Z., et al. (2024). *VBench: Comprehensive Benchmark Suite for Video Generative Models*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21807–21818. [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html)
2. Zheng, D., et al. (2025). *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness*. arXiv:2503.21755. [Paper](https://arxiv.org/abs/2503.21755)
3. Marcinek, L., Irfan, B., Skantze, G., Pereira, A., & Gustafsson, J. (2025). *Role of Reasoning in LLM Enjoyment Detection: Evaluation Across Conversational Levels for Human-Robot Interaction*. Proceedings of SIGDIAL, 573–590. [Paper](https://aclanthology.org/2025.sigdial-1.46/)
4. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. Advances in Neural Information Processing Systems, 36. [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

## Study materials

The repository contains the two RQ1 notebooks and full-precision estimates, including the bootstrap diagnostics. [Scoring notebook](../notebooks/vbench_score_clips.ipynb) · [Association notebook](../notebooks/rq1-aidrama.ipynb) · [RQ1 results](../rq1_outputs/rq1_kendall_results.csv)

RQ2 materials include the viewer prompt and responses, the paired reason-comparison rubric, the ten clip-level rating comparisons, the final alignment distribution, and the human validation snapshot used here. [Viewer prompt](../rq2_prompts/viewer.md) · [Viewer responses](../RQ2%20Setup/vlm_watched_results.md) · [Scoring rubric](../RQ2%20Setup/judge.md) · [Clip ratings](../RQ2%20Setup/rq2_setup/paired_v2_xai_grok47/outputs/enjoyment_rating_comparison.csv) · [Reason scores](../RQ2%20Setup/rq2_setup/paired_v2_xai_grok47/outputs/alignment_score_distribution.csv) · [Validation metrics](../RQ2%20Setup/rq2_setup/paired_v2_xai_grok47/outputs/validation/20261122T213056_766944Z/validation_metrics.json)
