# Scientific Deviations from Published Papers

This document explains the assumptions, adaptations, and deviations made when
linking the published PK, biomarker, and efficacy evidence into one adult RCC
PK/PD simulation.

## 1. Everolimus Biomarker Parameters Transposed to Sirolimus/Temsirolimus

### Deviation

O'Donnell et al. studied everolimus pS6K1 suppression. This model applies the
same Emax relationship to molar-summed temsirolimus plus sirolimus exposure.

### Justification

- Everolimus, sirolimus, and temsirolimus all act through FKBP12-mTORC1.
- pS6K1 is downstream of mTORC1 and is a pathway-response biomarker rather than
  a drug-specific assay.
- The model uses molar concentration, not mass concentration, to compare active
  molecules.
- RECORD-1 calibration provides an empirical clinical check on the transposed
  biomarker relationship.

### Risk

If sirolimus has a meaningfully different effective EC50 in PBMC or tumor
tissue, the model may over- or under-estimate target engagement. This is a
moderate-risk core assumption and should be validated prospectively with adult
RCC PK/PD sampling.

## 2. Adult Fm Set to 0.70 Instead of Mizuno Pediatric 0.459

### Deviation

Mizuno et al. report pediatric `Fm = 0.459`. The adult RCC model uses
`Fm = 0.70`.

### Justification

- Mizuno et al. note that sirolimus-to-temsirolimus AUC ratios were lower in
  children than adults.
- Higher adult sirolimus exposure implies greater effective conversion and/or
  altered clearance ratio in adults.
- `Fm = 0.70` lies inside the upper end of the Mizuno bootstrap interval for
  the pediatric estimate and is a conservative adult central value.

### Risk

Adult `Fm` is inferred, not re-estimated from adult parent-metabolite data in
this repository. The impact on calibrated PFS is limited because the resistance
parameters are fitted to clinical endpoints, but adult PK sampling would be
needed to validate exposure predictions.

## 3. Pediatric Dose Covariate Disabled

### Deviation

Mizuno et al. included a dose covariate on `CL_TEM`, `Q2`, and `Q3`. This model
sets the dose covariate multiplier to 1.0.

### Justification

- The numeric dose-per-kg reference used to center the covariate was not
  published.
- The adult RCC model uses one fixed clinical dose: 25 mg IV weekly.
- Applying an uncentered pediatric dose covariate to a single adult dose would
  introduce a hidden arbitrary assumption.

### Risk

Low for the current single-dose model. If future work evaluates dose escalation
or dose-response, the covariate should be revisited and refit or used as a
prior.

## 4. Week-4 Steady State Approximated by 1.5x Accumulation

### Deviation

Instead of explicitly simulating repeated weekly dosing for several weeks, the
adult model scales the week-1 profile by an accumulation factor of 1.5.

### Justification

- The requested PD model specified this simplification.
- The primary validation target is aggregate PFS, not exact week-specific PK.
- The approximation keeps the model auditable and fast while preserving the
  exposure-to-biomarker link.

### Risk

Moderate for exact trough prediction. Low for the current clinical PFS
calibration because the resistance parameters absorb aggregate exposure-effect
differences.

## 5. S6K1 Target Engagement Threshold of 80%

### Deviation

The source literature describes strong or near-complete S6K1 suppression but
does not define a universal 80% threshold. This repository reports 80% as the
target-engagement benchmark.

### Justification

- 80% is a practical pharmacology threshold for robust pathway engagement.
- The model's seeded median is 79.9%, within rounding and measurement noise of
  the benchmark.
- PFS calibration uses the continuous inhibition time course, not a binary 80%
  cutoff.

### Risk

Low for PFS. Moderate for biomarker-only claims. Prospective pS6K1 sampling
should refine the EC50, Emax, and target threshold.

## 6. Exponential Acquired Resistance Added to Tumor Growth

### Deviation

RECORD-1 did not publish a mechanistic acquired-resistance ODE. This model
adds:

```text
lambda_kill(t) = lambda_kill_0 * exp(-t / tau_resist)
```

### Justification

The constant-efficacy tumor model reproduced median PFS but overpredicted the
6- to 12-month PFS tail. Exponential decay of drug efficacy is a parsimonious
way to represent:

- clonal selection,
- feedback pathway activation,
- mTOR-independent escape,
- loss of durable tumor growth inhibition over weeks.

The final calibrated value is:

```text
tau_resist = 39.50 days
```

This timescale is biologically plausible for acquired pathway escape and is
constrained by RECORD-1 aggregate PFS endpoints.

### Risk

The resistance term is calibrated from aggregate endpoints, not individual
patient tumor trajectories. The model is suitable for cohort-level simulation
and hypothesis generation, not individual clinical prediction.

## 7. Responder/Resistant Mixture Model Tested and Rejected

### Deviation

A two-population mixture model was implemented as a diagnostic alternative but
not selected as the final structure.

### Findings

Using the proposed quick-path `pi = 0.35` produced:

```text
Median PFS = 1.05 months
6-month PFS = 10.0%
12-month PFS = 3.8%
```

Solving the supplied exponential anchor equation produced `pi` around 0.73,
not 0.35, and still failed to match median PFS in the full simulation:

```text
Median PFS = 2.74 months
6-month PFS = 28.0%
12-month PFS = 10.0%
```

### Justification for Rejection

The simple mixture hypothesis made the resistant branch progress too quickly
under the current tumor-growth parameters. RECORD-1 aggregate behavior is better
captured by a continuous time-varying efficacy model than by a discrete
two-population split.

## 8. Calibration Uses Aggregate RECORD-1 Endpoints

### Deviation

The model is calibrated to published aggregate endpoints, not individual-level
RECORD-1 data.

### Justification

Individual-level trial data and full Kaplan-Meier coordinates were not available
in this workspace. The available benchmarks were:

- median PFS,
- 6-month PFS probability,
- approximate 12-month PFS probability.

The calibration objective is transparent and reproducible.

### Risk

Parameter uncertainty is understated relative to a full patient-level fit.
Future work should refit with digitized Kaplan-Meier coordinates or individual
patient data if available.

## Risk Summary

| Assumption | Risk to PFS | Mitigation |
| --- | --- | --- |
| Everolimus-to-sirolimus biomarker transposition | Moderate | Adult pS6K1 validation |
| Adult Fm = 0.70 | Low-moderate | Adult parent-metabolite PK sampling |
| Dose covariate disabled | Low for fixed dose | Refit for dose-ranging work |
| 1.5x accumulation approximation | Moderate for troughs | Explicit repeated-dose PK |
| 80% S6K1 threshold | Low for PFS | Prospective biomarker sampling |
| Exponential resistance | Moderate | Fit to KM coordinates or patient data |
| Aggregate endpoint calibration | Moderate | Use individual-level trial data |

## Conclusion

The final model is scientifically defensible as a mechanistic, cohort-level
PK/PD simulation calibrated to published RECORD-1 endpoints. The key unresolved
need is prospective adult RCC PK/PD validation.

