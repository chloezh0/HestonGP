# HestonGP

This repository contains code for calibrating the Heston stochastic volatility model and comparing pricing/calibration approaches based on FFT, Monte Carlo, and Gaussian Process emulators.

## Repository Structure

- `code/` - notebooks, scripts, data, trained emulator artifacts, and generated outputs
- `code/data/` - SPY option data and synthetic Heston implied-volatility data
- `code/outputs/` - calibration summaries, validation metrics, model files, and comparison results
- `README.md` - project overview and usage notes

## Main Workflow

1. Pull and clean SPY option data.
2. Generate or load synthetic Heston training data.
3. Train GP or sparse GP emulators for implied volatility.
4. Calibrate Heston parameters using FFT, Monte Carlo, GP, and sparse GP methods.
5. Compare calibration accuracy and runtime across methods.

Key notebooks are located in `code/`, including:

- `pull_data.ipynb`
- `clean_spy_options.py`
- `generate_simulated_training_data.ipynb`
- `train_gp_emulator_sklearn.ipynb`
- `train_sparse_gp_emulator_sklearn.ipynb`
- `calibrate_fft_pricer.ipynb`
- `calibrate_monte_carlo_pricer.ipynb`
- `calibrate_gp_emulator.ipynb`
- `calibrate_sparse_gp_emulator.ipynb`
- `compare_calibrations_4.ipynb`
- `gp_uncertainty_analysis.ipynb`

## Requirements

The code is written in Python and uses Jupyter notebooks. Main dependencies include:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `yfinance`

Install the dependencies in your preferred Python environment before running the notebooks.

## Outputs

Generated outputs are saved under `code/outputs/`. These include calibration summaries, per-contract predictions, validation metrics, trained emulator files, and comparison tables.

## Notes

Run notebooks from the project root or from the `code/` directory so relative paths resolve correctly.
