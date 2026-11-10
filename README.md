# Early Stopping in Self-Distillation for Linear Regression

Category: General Machine Learning, Theory & Optimization  
Project type: Application — small-scale empirical reproduction and method comparison  
Student: Junjian Fan (范峻健)  
Student email: st135290@student.spbu.ru  
Institution: Saint Petersburg State University

## Research question and scope

Can validation-based selection of a self-distillation round improve prediction over the initial minimum-norm linear model and a validation-tuned ridge regression baseline?

The experiment follows the fresh-input linear self-distillation setting in Algorithm 1, Figure 1(b), and Appendix H of Wu, Yang, and Sun [1]. It is a small-scale reproduction of one setting, with an additional validation-selected stopping comparison; it is not a full reproduction of the paper, a new algorithm, or a reproduction of its iterated GCV procedure.

## Main findings

Across the 10 fixed seeds in each noise condition, validation-selected self-distillation has lower mean test MSE than the initial model and the round-15 model. Its reduction in mean test MSE relative to Initial is approximately 10.1%, 49.3%, and 73.9% for noise standard deviations 0.5, 1, and 2, respectively.

Ridge has the lowest mean test MSE among the four methods in all three conditions. These observations describe this synthetic setting; no statistical-significance or universal-superiority claim is made.

## Running the project in Google Colab

The supported execution workflow is Google Colab.

1. Open `ML_Final_Project.ipynb` in Google Colab, or upload it through Colab's notebook-opening dialog.
2. Use a CPU runtime. The experimental computations do not require a GPU, external dataset download, paid API, or manually installed system software.
3. Start from a fresh runtime and choose **Runtime → Run all**, or execute every code cell in order from top to bottom. Do not run the result-reading or plotting cells before the experiment cells.
4. Allow any download cells to finish. The code generates its synthetic data and writes the experiment records and figures beneath `project_outputs/`.
5. Check that the formal run finishes with 30 trials, 120 method records, 480 round records, 150 ridge-candidate records, and 12 summary records.
6. Inspect the three figures and retain the notebook, result files, and configuration. Downloading a results ZIP does not save the notebook source itself.

Earlier development/demo cells use separate random seeds and are excluded from the formal summaries. Do not replace the formal per-trial selected parameters with the development example's lambda or stopping round.

### Reference environment recorded by the formal run

| Component | Recorded version |
|---|---|
| Python | 3.13.18 |
| NumPy | 2.4.6 |
| Matplotlib | 3.10.0 |

Other imports are from the Python standard library; download cells additionally use `google.colab.files`. Colab package versions can change. Record the environment when rerunning, and do not assume bitwise-identical results across different numerical-library versions.

## Synthetic data and fixed settings

No external dataset is used. All inputs are generated from a zero-mean Gaussian distribution with independent coordinates:

- Initial labeled training set: 60 samples, 120 features.
- Validation set: 500 independent samples.
- Test set: 2,000 independent samples.
- Feature covariance: diagonal with entries 25, 1, ..., 1; equivalently, coordinate standard deviations are 5, 1, ..., 1.
- True coefficient vector: the first coordinate is 1 and all other coordinates are 0, so the noise-free regression function is the first input feature.
- Training and validation labels: first input feature plus independent zero-mean Gaussian noise with standard deviation 0.5, 1, or 2.
- Test targets: the noise-free first input feature.
- No intercept, feature standardization, or additional derived features are used.
- Each distillation round receives a fresh independent batch of 60 input vectors with the same feature distribution.
- Rounds 0 through 15 are recorded; round 0 is the initial model.
- Ten formal seeds per noise condition: 0 through 9, for 30 distinct trials in total.
- Ridge lambda candidates: 0.01, 0.1, 1, 10, and 100.

Within a trial, all four methods use the same original labeled training data, validation set, and test set. Self-distillation additionally uses fresh unlabeled input batches; the methods therefore do not have identical total input-data budgets.

### Random-number scheme

For each trial, `noise_code = int(round(1000 * noise_std))`. The code uses `np.random.SeedSequence([seed, noise_code]).spawn(4)`, assigning its child streams in this order:

1. Initial training inputs and label noise.
2. Validation inputs and label noise.
3. Fresh distillation inputs.
4. Test inputs.

Formal trials do not reuse the development arrays. The same numerical seed at different noise levels refers to a different seed/noise-code pair.

## Methods and model selection

### Initial

Fit the minimum-Euclidean-norm least-squares solution on the original noisy training data with `np.linalg.lstsq(X0, y0, rcond=None)`. Since there are more features than labeled training samples, near-zero original training error is expected and is not evidence of perfect generalization.

### Ridge

Minimize the original training MSE plus lambda times the squared coefficient norm:

$$
\frac{1}{n}\|X_0\beta-y_0\|_2^2 + \lambda\|\beta\|_2^2.
$$

The implementation solves the system with matrix `X0.T @ X0 + n * lambda * I`. The factor `n` is required by this MSE-based objective; lambda here is not the unscaled penalty in a sum-of-squared-errors objective.

Choose lambda only by validation MSE. For an exact tie, use the smallest candidate lambda.

### Fresh-input self-distillation

Starting from Initial, at each round:

1. Generate a new 60-by-120 input matrix.
2. Use the previous model's predictions as pseudo-labels.
3. Fit a new minimum-norm least-squares model to these pseudo-labels.
4. Do not add new noise to the pseudo-labels.
5. Record error on the original noisy training set and on the validation set.

The teacher is the previous minimum-norm model, not the ridge model. Reusing exactly the same inputs would leave the minimum-norm model essentially unchanged; the development checks test this numerical property.

### EarlyStop

After recording rounds 0–15, select the saved checkpoint with the smallest validation MSE. An exact tie is resolved in favor of the earliest round, and round 0 is allowed.

This is retrospective validation-based checkpoint selection, not a claim that computation actually stops online at the selected round. Later rounds are retained for analysis and for the Final baseline.

### Final

Use round 15 without validation-based round selection.

### Test-set separation

Choose ridge lambda and the EarlyStop checkpoint before drawing the independent test inputs and computing test metrics. The saved test-error-by-round curve is descriptive; its minimum is not used to choose a checkpoint, change the seed list, or retune the experiment.

## Metrics and formal results

All values below are mean ± sample standard deviation across 10 seeds; standard deviation uses `ddof=1`. These are not confidence intervals.

Training MSE is measured against the original noisy labels. Validation MSE is measured against independent noisy labels. Test MSE is measured against the noise-free regression function, so the raw train–test difference is not a conventional same-target generalization gap.

Per trial, the split sizes are training 60, validation 500, and test 2,000; each input has 120 features. Distillation uses 60 fresh input samples per round.

| Noise SD | Method | Original training MSE | Validation MSE | Clean-target test MSE |
|---|---|---|---|---|
| 0.5 | Initial | 3.13e-29 ± 1.78e-29 | 0.538 ± 0.085 | 0.314 ± 0.071 |
| 0.5 | Ridge | 0.055 ± 0.047 | 0.458 ± 0.069 | 0.231 ± 0.050 |
| 0.5 | EarlyStop | 0.160 ± 0.158 | 0.504 ± 0.084 | 0.282 ± 0.071 |
| 0.5 | Final | 6.287 ± 1.493 | 5.696 ± 0.919 | 5.506 ± 0.798 |
| 1.0 | Initial | 6.38e-29 ± 3.35e-29 | 2.016 ± 0.314 | 1.106 ± 0.310 |
| 1.0 | Ridge | 0.257 ± 0.058 | 1.388 ± 0.107 | 0.458 ± 0.132 |
| 1.0 | EarlyStop | 1.035 ± 0.364 | 1.502 ± 0.216 | 0.561 ± 0.231 |
| 1.0 | Final | 6.229 ± 0.942 | 6.283 ± 0.803 | 5.475 ± 0.508 |
| 2.0 | Initial | 6.34e-29 ± 5.18e-29 | 7.846 ± 1.383 | 3.922 ± 1.096 |
| 2.0 | Ridge | 0.806 ± 0.185 | 4.957 ± 0.462 | 0.980 ± 0.168 |
| 2.0 | EarlyStop | 4.494 ± 0.837 | 4.943 ± 0.352 | 1.022 ± 0.434 |
| 2.0 | Final | 11.132 ± 2.617 | 9.593 ± 0.939 | 5.836 ± 1.315 |

### Validation-selected rounds

| Noise SD | Selected-round counts across 10 seeds |
|---|---|
| 0.5 | Round 0: 4; round 1: 5; round 2: 1 |
| 1.0 | Round 1: 2; round 2: 6; round 3: 2 |
| 2.0 | Round 3: 6; round 4: 4 |

The mean test score of individually selected checkpoints need not equal any point on the mean fixed-round test curve: different trials can select different rounds.

## Files and outputs

| File | Purpose |
|---|---|
| `ML_Final_Project.ipynb` | Development checks, formal experiments, evaluation, and plotting |
| `README.md` | Scope, dependencies, data generation, settings, reproduction steps, and interpretation |
| `project_outputs/formal_config.json` | Frozen formal experiment configuration |
| `project_outputs/formal_run_metadata.json` | Configuration snapshot/hash, completion status, record counts, and first-trial repeat check |
| `project_outputs/formal_runs.csv` | 120 records: one for each trial/method combination |
| `project_outputs/formal_rounds.csv` | 480 records: one for each trial/round combination |
| `project_outputs/formal_ridge_candidates.csv` | 150 records: candidate lambda, training MSE, and validation MSE |
| `project_outputs/formal_summary.csv` | 12 records: means and sample standard deviations by noise level and method |
| `project_outputs/figures/01_test_comparison.png` | Method comparison with sample-SD error bars |
| `project_outputs/figures/02_test_error_by_round.png` | Fixed-round test-error curves with sample-SD bands and reference means |
| `project_outputs/figures/03_stopping_rounds.png` | Validation-selected stopping-round distributions |

No failed trials or unfavorable seeds are intentionally omitted. An additional repeat of the first trial checks deterministic consistency in the same runtime and is not included as a 31st independent trial. This limited repeat check is not a substitute for verifying a fresh-runtime execution of the complete notebook.

## Interpretation and limitations

The observed mean fixed-round test curves initially improve and later deteriorate. Validation-based checkpoint selection avoids committing to the late-round model, but does not outperform the tuned ridge baseline on mean test MSE in these experiments.

The experiment does not separately estimate denoising and signal-forgetting components, so the MSE curves alone do not prove a causal decomposition of those mechanisms. The study is limited to synthetic Gaussian inputs, one spiked covariance, one coefficient direction, an aspect ratio of 2, a finite lambda grid, 15 distillation rounds, and 10 repeats per condition.

The test metric measures recovery of a known noise-free function, not prediction error on noisy real-world outcomes. Uncertainty summaries are sample standard deviations, and no formal significance test is reported.

## Contribution and assistance

This is an individual student project by Junjian Fan. The student is responsible for understanding the referenced setting, running and checking experiments, interpreting results, and preparing the final submission and presentation.

ChatGPT assistance was used for project planning, code drafting/debugging, and writing support. The method is adapted from [1] and implemented using NumPy linear algebra; it is not presented as the paper authors' original source code. Any course-specific AI-use and attribution rules must also be followed.

## Reference

[1] Mingqi Wu, Archer Yi Yang, and Qiang Sun. 2026. *Why Self-Distillation Helps and Hurts: Denoising vs. Signal Forgetting*. ICML 2026, Proceedings of Machine Learning Research, volume 306.  
https://proceedings.mlr.press/v306/wu26bs.html

