from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "hestongp_matplotlib_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize


FEATURE_COLUMNS = ["v0", "kappa", "theta", "sigma_v", "rho", "K", "T"]
TARGET_COLUMN = "implied_vol"


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.mean


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    columns = data.dtype.names or ()
    missing = [name for name in [*FEATURE_COLUMNS, TARGET_COLUMN] if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    x = np.column_stack([data[name] for name in FEATURE_COLUMNS])
    y = np.asarray(data[TARGET_COLUMN], dtype=float)

    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if not finite.all():
        dropped = int((~finite).sum())
        print(f"Dropping {dropped:,} non-finite rows before training.")
        x = x[finite]
        y = y[finite]

    return x, y


def train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    n_validation = int(round(validation_fraction * len(y)))
    validation_idx = indices[:n_validation]
    train_idx = indices[n_validation:]
    return x[train_idx], x[validation_idx], y[train_idx], y[validation_idx]


def squared_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scales: np.ndarray,
) -> np.ndarray:
    x1_scaled = x1 / length_scales
    x2_scaled = x2 / length_scales
    x1_norm = np.sum(x1_scaled * x1_scaled, axis=1)[:, None]
    x2_norm = np.sum(x2_scaled * x2_scaled, axis=1)[None, :]
    distances = x1_norm + x2_norm - 2.0 * x1_scaled @ x2_scaled.T
    return np.maximum(distances, 0.0)


def rbf_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scales: np.ndarray,
    signal_std: float,
) -> np.ndarray:
    return signal_std**2 * np.exp(-0.5 * squared_distance(x1, x2, length_scales))


def unpack_params(params: np.ndarray, n_features: int) -> tuple[np.ndarray, float, float]:
    length_scales = np.exp(params[:n_features])
    signal_std = float(np.exp(params[n_features]))
    noise_std = float(np.exp(params[n_features + 1]))
    return length_scales, signal_std, noise_std


def negative_log_marginal_likelihood(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    length_scales, signal_std, noise_std = unpack_params(params, x.shape[1])
    kernel = rbf_kernel(x, x, length_scales, signal_std)
    kernel[np.diag_indices_from(kernel)] += noise_std**2 + 1e-8

    try:
        factor = cho_factor(kernel, lower=True, check_finite=False)
        alpha = cho_solve(factor, y, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf

    log_det = 2.0 * np.sum(np.log(np.diag(factor[0])))
    return float(0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(y) * np.log(2.0 * np.pi))


def optimize_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_rows: int,
    maxiter: int,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        sample_idx = rng.choice(len(y), size=max_rows, replace=False)
        x_fit = x[sample_idx]
        y_fit = y[sample_idx]
    else:
        x_fit = x
        y_fit = y

    n_features = x.shape[1]
    initial = np.r_[np.zeros(n_features), 0.0, np.log(1e-2)]
    bounds = [(np.log(0.05), np.log(10.0))] * n_features
    bounds += [(np.log(0.1), np.log(10.0)), (np.log(1e-6), np.log(0.5))]

    result = minimize(
        negative_log_marginal_likelihood,
        initial,
        args=(x_fit, y_fit),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-5},
    )
    if not result.success:
        print(f"Hyperparameter optimizer stopped with message: {result.message}")

    return unpack_params(result.x, n_features)


def fit_gp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    length_scales: np.ndarray,
    signal_std: float,
    noise_std: float,
) -> tuple[tuple[np.ndarray, bool], np.ndarray]:
    kernel = rbf_kernel(x_train, x_train, length_scales, signal_std)
    kernel[np.diag_indices_from(kernel)] += noise_std**2 + 1e-8
    factor = cho_factor(kernel, lower=True, check_finite=False)
    alpha = cho_solve(factor, y_train, check_finite=False)
    return factor, alpha


def predict_gp(
    x_query: np.ndarray,
    x_train: np.ndarray,
    factor: tuple[np.ndarray, bool],
    alpha: np.ndarray,
    length_scales: np.ndarray,
    signal_std: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.empty(x_query.shape[0])
    std = np.empty(x_query.shape[0])
    chol, lower = factor

    for start in range(0, x_query.shape[0], chunk_size):
        stop = min(start + chunk_size, x_query.shape[0])
        k_star = rbf_kernel(x_query[start:stop], x_train, length_scales, signal_std)
        mean[start:stop] = k_star @ alpha

        v = solve_triangular(chol, k_star.T, lower=lower, check_finite=False)
        variance = signal_std**2 - np.sum(v * v, axis=0)
        std[start:stop] = np.sqrt(np.maximum(variance, 0.0))

    return mean, std


def write_validation_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_implied_vol", "predicted_implied_vol", "predictive_std"])
        writer.writerows(zip(y_true, y_pred, y_std))


def plot_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (high - low)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", lw=1.5)
    ax.set_xlabel("True implied volatility")
    ax.set_ylabel("Predicted implied volatility")
    ax.set_title("Gaussian Process Emulator Validation")
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GP emulator for Heston implied volatility.")
    parser.add_argument("--data", type=Path, default=Path("data/simulated_training_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gp_emulator"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hyperopt-rows", type=int, default=1200)
    parser.add_argument("--max-gp-train-rows", type=int, default=2500)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--prediction-chunk-size", type=int, default=4096)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_training_data(args.data)
    x_train, x_validation, y_train, y_validation = train_validation_split(
        x,
        y,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train[:, None])
    x_train_scaled = x_scaler.transform(x_train)
    x_validation_scaled = x_scaler.transform(x_validation)
    y_train_scaled = y_scaler.transform(y_train[:, None]).ravel()

    rng = np.random.default_rng(args.seed)
    if len(y_train_scaled) > args.max_gp_train_rows:
        gp_idx = rng.choice(len(y_train_scaled), size=args.max_gp_train_rows, replace=False)
        x_gp = x_train_scaled[gp_idx]
        y_gp = y_train_scaled[gp_idx]
    else:
        x_gp = x_train_scaled
        y_gp = y_train_scaled

    print(f"Loaded {len(y):,} rows from {args.data}")
    print(f"Train rows: {len(y_train):,}; validation rows: {len(y_validation):,}")
    print(f"Exact GP fit rows: {len(y_gp):,}")
    print("Optimizing GP hyperparameters...")
    length_scales, signal_std, noise_std = optimize_hyperparameters(
        x_gp,
        y_gp,
        seed=args.seed,
        max_rows=args.max_hyperopt_rows,
        maxiter=args.maxiter,
    )

    print("Fitting GP posterior...")
    factor, alpha = fit_gp(x_gp, y_gp, length_scales, signal_std, noise_std)
    y_pred_scaled, y_std_scaled = predict_gp(
        x_validation_scaled,
        x_gp,
        factor,
        alpha,
        length_scales,
        signal_std,
        chunk_size=args.prediction_chunk_size,
    )

    y_pred = y_scaler.inverse_transform(y_pred_scaled[:, None]).ravel()
    y_std = y_std_scaled * y_scaler.scale[0]

    residual = y_pred - y_validation
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs_error = float(np.max(np.abs(residual)))
    r2 = float(1.0 - np.sum(residual**2) / np.sum((y_validation - y_validation.mean()) ** 2))

    np.savez(
        args.output_dir / "gp_emulator_model.npz",
        feature_columns=np.array(FEATURE_COLUMNS),
        target_column=np.array(TARGET_COLUMN),
        x_train_scaled=x_gp,
        y_train_scaled=y_gp,
        x_mean=x_scaler.mean,
        x_scale=x_scaler.scale,
        y_mean=y_scaler.mean,
        y_scale=y_scaler.scale,
        length_scales=length_scales,
        signal_std=signal_std,
        noise_std=noise_std,
        alpha=alpha,
        chol=factor[0],
        chol_lower=np.array(factor[1]),
    )
    write_validation_predictions(args.output_dir / "validation_predictions.csv", y_validation, y_pred, y_std)
    plot_predictions(args.output_dir / "predicted_vs_true_iv.png", y_validation, y_pred)

    metrics_path = args.output_dir / "validation_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"data_rows={len(y)}",
                f"train_rows={len(y_train)}",
                f"validation_rows={len(y_validation)}",
                f"exact_gp_fit_rows={len(y_gp)}",
                f"rmse={rmse:.8f}",
                f"mae={mae:.8f}",
                f"max_abs_error={max_abs_error:.8f}",
                f"r2={r2:.8f}",
                f"length_scales={','.join(f'{value:.8f}' for value in length_scales)}",
                f"signal_std={signal_std:.8f}",
                f"noise_std={noise_std:.8f}",
            ]
        )
        + "\n"
    )

    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation MAE:  {mae:.6f}")
    print(f"Validation R^2:  {r2:.6f}")
    print(f"Saved model and validation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
