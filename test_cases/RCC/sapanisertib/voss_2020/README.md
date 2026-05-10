# Sapanisertib PK/PD Simulation - RCC
## Test Case: Voss et al. (2020) [PMCID: PMC7686313]

### Overview

This test case implements a deterministic mean-data PK/PD model for
sapanisertib (TAK-228), an oral mTORC1/2 inhibitor studied in advanced solid
tumors including RCC. The model uses only the numerical values from Voss et al.
2020, Supplementary Table 5, and visually read median/bin values from
Supplementary Figure 5.

Model blocks:

- PK: 1-compartment oral model with first-order absorption and first-order
  elimination.
- PD: inhibitory sigmoidal Emax concentration-response models for p4EBP1,
  pS6, pNDRG1, and pPRAS40.
- PK/PD integration: continuous 28-day pathway inhibition profiles for QD, QW,
  QD3dQW, and QD5dQW schedules.
- Sensitivity analysis: EC50 values of 3, 6, 10, and 15 ng/mL.

No population PK, random effects, stochastic simulation, individual-patient
simulation, or external literature EC50 values are used.

### Reference

Voss MH, Gordon MS, Mita M, et al. Phase 1 study of mTORC1/2 inhibitor
sapanisertib (TAK-228) in advanced solid tumours, with an expansion phase in
renal, endometrial or bladder cancer. *Br J Cancer.* 2020;123:1590-1598.
PMCID: PMC7686313.

### Folder Structure

```text
sapanisertib/
  voss_2020/
    sapanisertib_pkpd_model.py
    requirements.txt
    README.md
    results/
      accumulation_check.csv
      comparison_5mg_qd_vs_30mg_qw.csv
      ec50_sensitivity.csv
      pd_fitted_parameters.csv
      pk_validation.csv
      pkpd_summary.csv
      qd_dose_coverage.csv
      figures/
        01_pk_auc_primary_cmax_diagnostic.png
        02_pk_profiles_28d.png
        ...
        13_combined_pk_pd_story_p4ebp1.png
```

### How To Run

From repository root:

```bash
python test_cases/RCC/sapanisertib/voss_2020/sapanisertib_pkpd_model.py
```

The script regenerates all CSV outputs under `results/` and all PNG figures
under `results/figures/`.

### PK Parameterization

Fixed PK parameters:

| Parameter | Value |
|---|---:|
| Ka | 1.2 h^-1 |
| Cl/F | 19.0 L/h |
| Vd/F | 200 L |
| Ke | 0.095 h^-1 |
| F | 1.0 relative |

The model predicts dose-proportional Cmax and AUC. AUC is treated as the
primary PK validation endpoint because it is more robust than Cmax to sparse
sampling, small sample sizes, and outlier-driven means. Cmax is retained as a
diagnostic of observed variability.

Accumulation is minimal:

| Schedule | Predicted C2D1/C1D1 AUC ratio |
|---|---:|
| QD | 1.125 |
| QW | 1.000 |

### PD Fit Summary

Emax is constrained to a maximum of 100% inhibition.

| Biomarker | Emax % | EC50 ng/mL | Hill | R2 | RMSE % |
|---|---:|---:|---:|---:|---:|
| pS6 | 94.12 | 2.84 | 0.97 | 0.982 | 1.51 |
| p4EBP1 | 100.00 | 3.24 | 1.95 | 0.999 | 0.38 |
| pNDRG1 | 96.02 | 3.31 | 1.79 | 0.997 | 0.66 |
| pPRAS40 | 94.26 | 3.64 | 1.05 | 0.985 | 1.67 |

R2 and RMSE are calculated against binned median values, not raw patient
scatter. Because each fit uses only 4-7 median/bin points and 3 free
parameters, R2 should not be interpreted as strong external validation.

### Key PK/PD Finding

The model identifies a pharmacological QD dose-coverage threshold between
4 and 6 mg QD for p4EBP1/TORC1 inhibition:

| QD dose | Ctrough ng/mL | ft > p4EBP1 EC50 |
|---:|---:|---:|
| 2 mg | 1.24 | 56.9% |
| 4 mg | 2.47 | 87.7% |
| 5 mg | 3.09 | 97.7% |
| 6 mg | 3.71 | ~100% |

The recommended Phase 2 dose of 5 mg QD sits at the edge of continuous pathway
coverage: predicted Ctrough is 3.09 ng/mL versus fitted p4EBP1 EC50 of
3.24 ng/mL, giving a trough-to-EC50 ratio of 0.95.

This is a retrospective model-derived finding. The clinical study selected
5 mg QD for tolerability after 6 mg QD was the MTD and was poorly tolerated
beyond the DLT evaluation window. The model suggests that this tolerability
constraint coincided closely with the pharmacological threshold for sustained
TORC1 pathway coverage.

### 5 mg QD vs 30 mg QW

Despite similar average plasma concentrations, the schedules differ strongly in
pathway coverage because the half-life is short relative to the 168-hour weekly
dosing interval.

| Scenario | Biomarker | Cmax | Cavg | Ctrough | ft > EC50 | AUEC |
|---|---|---:|---:|---:|---:|---:|
| 5 mg QD | p4EBP1 | 22.60 | 10.92 | 3.09 | 97.7% | 55,867 |
| 5 mg QD | pNDRG1 | 22.60 | 10.92 | 3.09 | 96.5% | 52,509 |
| 30 mg QW | p4EBP1 | 120.61 | 9.40 | ~0.00 | 24.5% | 16,470 |
| 30 mg QW | pNDRG1 | 120.61 | 9.40 | ~0.00 | 24.4% | 15,736 |

Using fitted EC50 values, 5 mg QD gives about 3.4-fold greater p4EBP1 AUEC than
30 mg QW over 28 days.

### EC50 Sensitivity

The near-continuous coverage claim depends on the low fitted skin-biopsy EC50.
The broader conclusion that QD gives greater cumulative inhibition than QW is
robust across assumed EC50 values of 3-15 ng/mL.

At higher assumed EC50 values, such as 10-15 ng/mL, 5 mg QD no longer remains
above EC50 for most of the cycle, but it still provides higher AUEC than
30 mg QW.

### Interpretation Limits

- This model describes pharmacology, not clinical efficacy.
- It cannot establish that greater AUEC translates to greater antitumor
  response.
- Skin biopsy EC50 may differ from tumor EC50 because of tissue penetration,
  pathway dependency, and assay differences.
- The biopsy PD data were measured 2-4 hours post-dose, near peak exposure.
  Trough and off-period inhibition are model extrapolations and were not
  directly validated by trough biopsies.
- The trough-to-EC50 ratio of 0.95 for 5 mg QD should be interpreted as
  EC50-dependent, not as a definitive claim of continuous tumor inhibition.

### Main Outputs

- `pk_validation.csv`: AUC primary validation and Cmax diagnostic residuals.
- `accumulation_check.csv`: predicted and observed C2D1/C1D1 AUC ratios.
- `pd_fitted_parameters.csv`: Emax, EC50, Hill coefficient, 95% CI, R2, RMSE.
- `pkpd_summary.csv`: Cmax, Cavg, Ctrough, ft>EC50, and AUEC by scenario.
- `comparison_5mg_qd_vs_30mg_qw.csv`: p4EBP1/pNDRG1 comparison.
- `ec50_sensitivity.csv`: ft>EC50 and AUEC at assumed EC50 values.
- `qd_dose_coverage.csv`: QD dose threshold analysis.
- `results/figures/*.png`: validation, PK, PD, PK/PD, rebound, sensitivity,
  dose-coverage, and combined story figures.
