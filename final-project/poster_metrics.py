"""Reproduce the poster's train/test MSE table with the same data streams as ML_final.ipynb."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits


def draw_inputs(rng, count):
    inputs = rng.normal(size=(count, 1000))
    inputs[:, 0] *= 5
    return inputs


def clean_targets(inputs, sign):
    first = inputs[:, 0]
    mixed = np.sqrt(0.9) * first + sign * 5 * np.sqrt(0.1) * inputs[:, 1]
    return np.column_stack([first, mixed])


def fit_ridge(inputs, labels, seed):
    penalties = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    validation_mse = np.zeros((len(penalties), labels.shape[1]))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train, valid in folds.split(inputs):
        for j, penalty in enumerate(penalties):
            model = Ridge(alpha=len(train) * penalty, fit_intercept=False, solver="cholesky")
            model.fit(inputs[train], labels[train])
            residual = model.predict(inputs[valid]) - labels[valid]
            validation_mse[j] += (residual ** 2).mean(axis=0) / 5
    columns = []
    for j, index in enumerate(validation_mse.argmin(axis=0)):
        model = Ridge(alpha=len(inputs) * penalties[index], fit_intercept=False, solver="cholesky")
        columns.append(model.fit(inputs, labels[:, j]).coef_)
    return np.column_stack(columns)


def evaluate_seed(seed):
    base = 1000 * seed
    rng = np.random.default_rng(base)
    sign = np.random.default_rng([seed, 9001]).choice([-1, 1])
    inputs = draw_inputs(rng, 500)
    labels = clean_targets(inputs, sign) + rng.normal(size=(500, 1))
    test = draw_inputs(rng, 2000)
    truth = clean_targets(test, sign)

    inverse_gram = solve(inputs @ inputs.T, np.eye(len(inputs)), assume_a="pos")
    mapping = inputs.T @ inverse_gram
    initial = mapping @ labels
    direction = np.linalg.svd(inputs, full_matrices=False)[2][0]
    anchor = direction @ initial
    protected = initial.copy()
    correction = (inverse_gram @ labels) / np.trace(inverse_gram)
    path, scores = [], []
    models = {"Initial": initial}

    for round_id in range(16):
        if round_id:
            fresh = draw_inputs(np.random.default_rng(base + round_id), 500)
            mapping = fresh.T @ solve(fresh @ fresh.T, fresh @ mapping, assume_a="pos")
            if round_id <= 5:
                augmented = np.vstack([fresh, direction])
                augmented_targets = np.vstack([fresh @ protected, anchor])
                protected = np.linalg.lstsq(augmented, augmented_targets, rcond=None)[0]
        standard = mapping @ labels
        trace_h = np.einsum("ij,ji->", inputs, mapping)
        residual = labels - inputs @ standard + trace_h * correction
        scores.append((residual ** 2).mean(axis=0))
        path.append(standard)
        if round_id == 5:
            models["Standard, round 5"] = standard
            models["Protected, round 5"] = protected

    selected = np.argmin(scores, axis=0)
    models["Standard, iGCV"] = np.column_stack([path[t][:, j] for j, t in enumerate(selected)])
    models["Ridge, fivefold"] = fit_ridge(inputs, labels, seed)

    records = []
    for method, weights in models.items():
        train_mse = ((inputs @ weights - labels) ** 2).mean(axis=0)
        test_mse = ((test @ weights - truth) ** 2).mean(axis=0)
        for j, share in enumerate((1.0, 0.9)):
            records.append((seed, method, share, train_mse[j], test_mse[j]))
    return records


def main():
    rows = []
    with threadpool_limits(limits=2, user_api="blas"):
        for seed in range(10):
            rows.extend(evaluate_seed(seed))
            print(f"Seed {seed} complete", flush=True)
    scores = pd.DataFrame(rows, columns=["seed", "method", "signal_share", "train_mse", "test_mse"])
    table = scores.groupby(["method", "signal_share"], sort=False)[["train_mse", "test_mse"]].mean()
    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    table.to_csv(results / "poster_mse.csv")
    print(table.round(4))


if __name__ == "__main__":
    main()

