# RCC Cabozantinib PK/PD Simulation (METEOR-Inspired)

This test case provides a Python re-implementation of a population PK/PD simulation for cabozantinib in RCC, including:

- Population PK with inter-individual variability and 2-week dose-modification logic
- Tumor dynamics with resistance attenuation and RECIST-based BOR/ORR
- PFS hazard ratio estimation via Cox proportional hazards
- Safety exposure-response hazard ratios (PPES, Fatigue, Hypertension, Diarrhea)

## Contents

- `cabozantinib_pkpd_simulation.py` - main model and simulation engine
- `run_large_simulation.py` - N=300 per dose production run entry point
- `simulation_summary.txt` - finalized summary table from the latest clean run
- `cabozantinib_pkpd_results.png` - finalized six-panel output figure

## Environment

- Python 3.12+
- numpy
- pandas
- scipy
- matplotlib
- lifelines

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib lifelines
```

## Run

From this folder:

```bash
python run_large_simulation.py
```

This writes/updates:

- `simulation_summary.txt`
- `cabozantinib_pkpd_results.png`

## Current Calibration Notes

The model reproduces:

- Correct dose ordering for PK, tumor endpoints, and ORR
- PFS HRs close to target via Cox PH
- Safety HRs exactly by construction from calibrated ER betas

Known limitation:

- Jointly matching both ORR and 1-year median tumor percent change exactly at all doses is structurally constrained without access to the original patient-level estimation data and full NONMEM fit outputs. Remaining discrepancies are expected for an aggregate-data re-implementation.

## Reproducibility

- Random seed is fixed in the simulation module (`np.random.seed(42)`).
- Production run default: 300 virtual patients per starting-dose group (20/40/60 mg).

## Intended Repository Location

This folder is prepared to be placed under:

- `test_cases/RCC`

in:

- `TAKAHUMAN/sim-engine-test-cases`
