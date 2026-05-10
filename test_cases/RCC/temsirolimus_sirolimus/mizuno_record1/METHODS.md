# Methods

This document describes the implemented equations, parameter sources,
simulation workflow, and validation endpoints.

## 1. PK Model

The PK model is a manual ODE implementation of the combined Mizuno et al.
temsirolimus-sirolimus parent-metabolite model.

### Compartments

Temsirolimus:

- `A1`: central amount.
- `A2`: peripheral 1 amount.
- `A3`: peripheral 2 amount.

Sirolimus:

- `A4`: central amount.
- `A5`: peripheral amount.

Amounts are micrograms. Volumes are liters. `A / V` is therefore micrograms/L,
which is numerically equal to ng/mL.

### ODE System

```text
C1 = A1 / V1
C2 = A2 / V2
C3 = A3 / V3
C4 = A4 / V4
C5 = A5 / V5

dA1/dt = R_inf(t) - CL_TEM*C1 - Q2*(C1-C2) - Q3*(C1-C3)
dA2/dt = Q2*(C1-C2)
dA3/dt = Q3*(C1-C3)
dA4/dt = Fm*CL_TEM*C1*(MW_SIR/MW_TEM) - CL_SIR*C4 - Q5*(C4-C5)
dA5/dt = Q5*(C4-C5)
```

The molecular-weight ratio is:

```text
MW_SIR / MW_TEM = 914.2 / 1030.3
```

### Allometric Scaling

All PK parameters are scaled from 70 kg using the Mizuno exponents:

- Clearances and intercompartmental clearances: `(BW/70)^0.75`.
- Volumes: `(BW/70)^1.0`.

The pediatric dose covariate is disabled by default because the centering
reference dose-per-kg was not published and the adult RCC application uses a
single fixed dose.

## 2. Adult RCC PK Adaptation

Adult RCC dosing:

```text
Dose = 25 mg IV weekly
Infusion duration = 0.5 h
Adult Fm = 0.70
Accumulation factor = 1.5
```

The adult profile is generated directly from the ODE model for each simulated
adult body weight. It is not interpolated from pediatric BSA-dose output.

## 3. Biomarker Model

The active concentration is the molar sum of temsirolimus and sirolimus:

```text
C_um = TEM_ng_ml / MW_TEM + SIR_ng_ml / MW_SIR
```

This conversion is dimensionally consistent because ng/mL equals micrograms/L
and molecular weight in g/mol is numerically micrograms/micromole.

pS6K1 inhibition follows a Hill Emax model:

```text
E(t) = Emax*C(t)^gamma / (EC50^gamma + C(t)^gamma)
pS6K1(t) = pS6K1_0 * (1 - E(t))
```

Parameters:

```text
Emax = 0.95
EC50 = 0.010 uM
gamma = 1.2
pS6K1_0 ~ lognormal(log(100), 0.3)
```

## 4. Tumor Growth and PFS

Tumor size is simulated on the log scale:

```text
dL/dt = lambda_growth - lambda_kill(t)*E(t)
```

where:

```text
lambda_growth ~ lognormal(log(0.0077), 0.5)
lambda_kill_0 ~ lognormal(log(lambda_kill_0_median), 0.6)
lambda_kill(t) = lambda_kill_0 * exp(-t / tau_resist)
```

Progression is defined as:

```text
L(t) >= L_0 + log(1.20)
```

Administrative censoring occurs at 365 days.

## 5. Adult RCC Population

The adult RCC PD cohort uses:

```text
Age: truncated normal, mean 62 years, SD 10, bounds 50-80
Body weight: truncated normal, mean 75 kg, SD 15, bounds 50-110
Sex: 70% male, 30% female
MSKCC risk: 30% favorable, 50% intermediate, 20% poor
Prior therapy: VEGF TKI failure
Baseline tumor size: lognormal median 70 mm, sigma 0.4
```

## 6. Calibration

The final model calibrates two parameters:

```text
lambda_kill_0_median
tau_resist
```

Targets:

```text
median PFS = 4.9 months
6-month PFS = 0.26
12-month PFS = 0.075
```

Loss:

```text
loss =
  ((median - 4.9) / 4.9)^2
+ ((pfs_6m - 0.26) / 0.26)^2
+ ((pfs_12m - 0.075) / 0.075)^2
```

The calibration routine uses common random numbers: each objective evaluation
uses the same seed and cohort size. This makes parameter comparisons stable
despite simulation noise.

The final committed validation uses:

```text
lambda_kill_0_median = 0.02455 /day
tau_resist = 39.50 days
N = 500
seed = 20240501
```

## 7. Validation

The final calibrated model is evaluated against RECORD-1 aggregate endpoints:

```text
Median PFS within 4.0-5.5 months
6-month PFS within 0.221-0.299
12-month PFS within 0.05-0.10
Median steady-state S6K1 inhibition >= 0.80
```

Final seeded result:

```text
Median PFS = 4.01 months
6-month PFS = 29.8%
12-month PFS = 7.4%
Median S6K1 inhibition = 79.9%
```

## 8. Reproducibility

Run:

```bash
pytest tests -q -p no:cacheprovider
python -m simulation.main
```

The primary calibrated report is:

```text
outputs/pd_calibrated_validation_report.txt
```

