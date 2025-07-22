
-# Gemini CLI Tasks

This document outlines the tasks performed by the Gemini CLI agent.

## Current Tasks:

1.  **Fix Heston Model Convergence:**
    *   **Description:** The `main.py` script in `Stochastic-Modeling/M3/` is experiencing convergence issues when calibrating the Heston model.
    *   **Action:** Adjust the `brute` force search ranges for `kappa` and `rho` to be more granular and appropriate. Add a `tol` parameter to the `minimize` function. Comment out the print statement in `calibration_objective` to reduce output during the brute-force search.
    *   **Status:** In progress.
