# When does protecting one direction help linear self-distillation?

Fan Junjian · Saint Petersburg State University · Machine Learning final project

This repository accompanies the English poster. The project reproduces early stopping in high-dimensional linear self-distillation and tests a modification that preserves the initial model's coefficient along the leading direction of its training inputs.

## Run the notebook in Google Colab

1. Open `ML_final.ipynb` in Google Colab using File → Open notebook → Upload (or the GitHub tab once the repository is online).
2. Select a CPU runtime and run all cells from the top. The notebook generates its own data. It needs no external dataset, credentials, mounted drive, or GPU.
3. Code cell 17 reports the independent crossover experiment, cell 19 the five-method poster comparison, and cell 20 the supplementary PCR baseline.

For a local run, use Python 3.12 with `python -m pip install -r requirements.txt`, then run `jupyter lab ML_final.ipynb` from the repository root. The source notebook retains the author's Colab outputs and Chinese learning notes.

## Locate the experiments

| Code cell | Question or output |
| --- | --- |
| 1–3 | What can a minimum-norm student learn from a toy teacher? |
| 4–8 | For a single seed, how do noisy and clean training labels affect the trajectory? |
| 9–10 | How does the ordinary trajectory change across ten seeds and three feature-to-sample ratios? |
| 11–14 | Can a student match its teacher and preserve a training-derived direction? How does it perform after five rounds? |
| 15–16 | How do the comparison and the predicted crossover change when signal lies in two directions? |
| 17 | Do 30 independent seeds support the predicted change of sign? |
| 18–19 | How do fivefold ridge regression and the paper's iGCV stopping rule compare? |
| 20 | Does protection improve on ordinary one-component regression along the same training-derived direction? |

## Reproduce the poster table

Run `python poster_metrics.py` to regenerate `results/poster_mse.csv`. The script uses the same data streams, stopping rule and ridge folds as the notebook. It also calculates the poster's training MSE for all five methods. Training MSE is evaluated on the 500 original noisy labels. Test MSE uses 2,000 independent clean targets. The 500 pseudo-label residuals within each distillation round answer a separate question. Column names and the method labels in the CSV are in English.

| Method | Test MSE, 100% main signal | Test MSE, 90% main signal |
| --- | ---: | ---: |
| Initial minimum-norm teacher | 1.1224 | 2.3230 |
| Standard distillation, five rounds | 1.2334 | 3.5301 |
| Direction protection, five rounds | 0.2464 | 2.6673 |
| Fivefold ridge regression | 0.4747 | 1.9786 |
| Standard distillation, iGCV stopping | 0.6657 | 2.3934 |

## Design and interpretation

The design follows the Gaussian simulation in Wu et al., Algorithm 1, Figure 1(b) and Appendix H. The first coordinate has variance 25, the other coordinates have variance 1, and the initial label noise has standard deviation 1. Every round uses 500 fresh inputs. Methods are paired on the same inputs within a trial. The reproduction varies p/n across 1.5, 2.0 and 2.5. The method comparison fixes p/n = 2. Ridge chooses its penalty by fivefold validation and iGCV chooses from rounds 0–15 using training data. Test-curve minima in cell 10 are descriptive.

The main experiments use seeds 0–9. The independent test uses seeds 200–229. In the signal-share sweep the total signal variance stays at 25, and each seed gets a randomly signed second component. The protection direction is estimated from the initial training inputs.

Cell 16 estimates a 92.6% crossover for **five protected rounds versus the initial teacher**. Cell 17 returns paired test-MSE gaps of +0.3607 at 90% and −0.2369 at 95%. Negative values favor protection. The intervals describe mean differences at each share. These results apply to the specified simulation and comparison. The five-row plotting data are in `results/boundary_check.csv`.

## Supplementary one-component baseline

Code cell 20 compares five protected rounds with one-component principal component regression (PCR) on the same initial inputs, labels and test samples. PCR fits one least-squares slope through the origin on the training-derived direction. It uses the original feature scales and does not require fresh pseudo-labelled inputs. Each seed also checks that the initial teacher's two test errors match the main comparison.

| Main signal share | PCR MSE | Protected MSE | Mean gap (protected minus PCR) | Paired 95% interval | Seeds favoring protection |
| --- | ---: | ---: | ---: | --- | ---: |
| 100% | 0.2691 | 0.2464 | −0.0227 | [−0.0462, +0.0009] | 8/10 |
| 90% | 2.7386 | 2.6673 | −0.0713 | [−0.1146, −0.0279] | 10/10 |

At full alignment the interval spans zero. At 90% the protected model improves on PCR, while ridge has the lowest mean test MSE among the six evaluated methods (1.9786). The 92.6% predicted crossover compares protection with the initial teacher. The original poster table contains the five methods in cell 19. CSV files for this supplement are `results/pcr_by_seed.csv` and `results/pcr_summary.csv`.

## Source and contributions

M. Wu, A. Y. Yang and Q. Sun. *Why Self-Distillation Helps and Hurts: Denoising vs. Signal Forgetting*. ICML 2026. [Paper and PDF](https://proceedings.mlr.press/v306/wu26bs.html).

Fan Junjian ran the original Colab experiments and checked their outputs. AI assistance supported code drafting, analysis, reproducibility checks, the supplementary PCR comparison and preparation of project materials.

