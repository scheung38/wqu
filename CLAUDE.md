# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance repository focused on stochastic modeling techniques, primarily for educational purposes related to WQU (WorldQuant University) coursework. The project implements various financial models including Heston stochastic volatility model, Merton jump-diffusion model, and Hidden Markov Models for financial time series.

## Project Structure

- **Stochastic-Modeling/M2/**: Module 2 assignments implementing various stochastic processes and option pricing models using Merton jump-diffusion
- **Stochastic-Modeling/M3/**: Module 3 project focusing on Heston model calibration to market option data
- **old/**: Contains older work including HMM analysis of EuroMillion lottery data

## Development Environment

This project uses `uv` as the Python package manager with the following key dependencies:
- `hmmlearn>=0.3.3` - Hidden Markov Models
- `jupyter>=1.1.1` and `jupyterlab>=4.4.2` - Notebook environment
- `numpy>=2.0.2`, `pandas>=2.3.0`, `scipy>=1.13.1` - Scientific computing stack

### Environment Setup
```bash
# Set up virtual environment with pip support for Jupyter
uv venv --seed

# Launch Jupyter Lab
uv run --with jupyter jupyter lab
```

### Running Scripts
```bash
# Execute Python scripts in the virtual environment
uv run python Stochastic-Modeling/M3/main.py

# Run test scripts
bash test.sh
```

## Key Components

### Heston Model Implementation (M3)
- **Main file**: `Stochastic-Modeling/M3/main.py`
- Implements Heston (1993) stochastic volatility model for option pricing
- Uses Lewis (2001) Fourier inversion method for European option pricing
- Features two-stage calibration: brute-force global search followed by local optimization
- Current focus: resolving convergence issues in model calibration

### Merton Jump-Diffusion Model (M2)
- **Files**: `Stochastic-Modeling/M2/Question*.py`
- Implements Merton (1976) jump-diffusion model
- Uses Lewis (2001) approach for option pricing via characteristic functions

### Data Handling
- Option market data stored in Excel format and HDF5 files
- Market data includes call/put prices across different strikes and maturities
- Put-Call parity relationships are used for model validation

## Common Development Tasks

### Model Calibration
The Heston model calibration process involves:
1. Loading and preparing market option data
2. Two-stage optimization:
   - Stage 1: Brute-force search over parameter ranges
   - Stage 2: Local refinement using Nelder-Mead optimization
3. Visualization of calibration results vs market prices

### Parameter Constraints
- Feller condition: `2 * kappa * theta >= sigma^2`
- Parameter bounds ensure numerical stability
- Correlation parameter `rho` typically negative for equity options

### Known Issues
- Heston model convergence challenges with certain parameter combinations
- Integration errors in characteristic function evaluation may require penalty handling

## File Organization Notes

- Scripts are self-contained with relative path handling using `pathlib`
- Data files (Excel, HDF5) are co-located with analysis scripts
- Visualization outputs (PNG files) are saved alongside source code
- Backup versions of main files indicate iterative development approach