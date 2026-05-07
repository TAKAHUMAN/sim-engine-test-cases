# Cediranib PK/PD Simulation - RCC
## Test Case: Li et al. (2017) [PMCID: PMC5510068]

### Overview
This test case implements a Cediranib (AZD2171) PK/PD validation workflow centered on the Li et al. 2017 population PK parameters and VEGFR-2 target coverage.

Model blocks:

- Allometric scaling check: rat-to-human CL, Vd, and terminal half-life.
- PK: 2-compartment oral model with finite zero-order input into an absorption depot followed by first-order absorption.
- PD: direct VEGFR-2 Emax inhibition linked to free plasma concentration.
- Biomarker: indirect-response sVEGFR-2 production inhibition model.
- Dose comparison: 15, 20, and 30 mg once daily steady-state target coverage.

### Reference

Li J, Al-Huniti N, Henningsson A, Tang W, Masson E. Population pharmacokinetic and exposure simulation analysis for cediranib (AZD2171) in pooled Phase I/II studies in patients with cancer. *Br J Clin Pharmacol.* 2017;83(9):1969–1980. doi: 10.1111/bcp.13266. PMCID: PMC5510068.

### Source Split (PK vs PD)

- PK source paper: Li et al. 2017 (PMCID: PMC5510068)
- PD/biomarker source: separate VEGFR-2 and sVEGFR-2 literature used for mechanistic linkage (not from the Li et al. popPK paper alone)

### Folder Structure

```text
cediranib/
  li_2017/
    cediranib_pkpd.py
    requirements.txt
    README.md
    results/
      allometric_scaling.csv
      dose_comparison.csv
      pkpd_20mg_timeseries.csv
      svegfr2_timeseries.csv
      figures/
        figure_1_pk_14day_total_free.png
        figure_2_pk_steady_state_total_free.png
        figure_3_vegfr2_inhibition_14day.png
        figure_4_vegfr2_inhibition_steady_state.png
        figure_5_svegfr2_treated_vs_untreated.png
        figure_6_dose_comparison_free_cp.png
        figure_7_dose_comparison_vegfr2_inhibition.png
        figure_8_allometric_scaling_summary.png
```

### Files

- `cediranib_pkpd.py`: Full allometry, PK, VEGFR-2 inhibition, sVEGFR-2, dose-comparison, and plotting workflow.
- `requirements.txt`: Minimal Python package requirements.
- `results/*.csv`: Current deterministic simulation tables.
- `results/figures/*.png`: Current generated validation plots at 300 dpi.

### Current Parameterization

Clinical PK defaults from Li et al. 2017:

- CL/F = 26.3 L/h
- Vc/F = 489 L
- Vp/F = 213 L
- Q/F = 11.8 L/h
- Ka = 2.70 1/h
- D1 = 1.68 h
- Typical patient: 73 kg, 59 years

Target and biomarker assumptions:

- Cediranib molecular weight = 450.5 g/mol
- Human free fraction = 0.05
- VEGFR-2 IC50 = 0.5 nM free drug
- VEGFR-3 IC50 = 3 nM free drug
- VEGFR-1 IC50 = 5 nM free drug
- Human sVEGFR-2 baseline = 10,000 pg/mL
- Human sVEGFR-2 turnover scaled from mouse 3-day turnover half-life using BW^-0.25

### How To Run

From repository root:

```bash
python test_cases/RCC/cediranib/li_2017/cediranib_pkpd.py
```

This regenerates all CSV outputs under `results/` and all figures under `results/figures/`.

### Validation Checkpoints

Current deterministic 20 mg once-daily results:

| Metric | Value |
|---|---:|
| AUCss total | 0.760 mg*h/L |
| Cmax,ss total | 0.0524 mg/L |
| Cmin,ss total | 0.0179 mg/L |
| Cmax,ss free | 5.81 nM |
| Cmin,ss free | 1.99 nM |
| Terminal half-life | 24.4 h |
| Mean VEGFR-2 inhibition | 86.5% |
| Min VEGFR-2 inhibition | 79.9% |
| Max VEGFR-2 inhibition | 92.1% |
| Time above 50% inhibition | 24.0 h/24 h |
| Time above 90% inhibition | 5.6 h/24 h |
| sVEGFR-2 day 7 | 8408 pg/mL |
| sVEGFR-2 day 14 | 7082 pg/mL |
| sVEGFR-2 reduction day 14 | 29.2% |

Dose comparison at steady state:

| Dose | Cmax free | Cmin free | Mean VEGFR-2 inhibition | Time above VEGFR-2 IC50 |
|---|---:|---:|---:|---:|
| 15 mg qd | 4.36 nM | 1.49 nM | 82.9% | 24.0 h/24 h |
| 20 mg qd | 5.81 nM | 1.99 nM | 86.5% | 24.0 h/24 h |
| 30 mg qd | 8.72 nM | 2.98 nM | 90.6% | 24.0 h/24 h |

Allometric scaling checkpoint:

| Parameter | Predicted from rat | Observed human | Error |
|---|---:|---:|---:|
| CL | 14.13 L/h | 15.8 L/h | -10.6% |
| Vd | 758.44 L | 421 L | +80.2% |
| t1/2 | 37.20 h | 24 h | +55.0% |

### Notes

- The concentration conversion used in the script is `nM = mg/L * 1e6 / MW`, matching the IC50 conversion `mg/L = nM * MW * 1e-6`.
- The allometric Vd comparison is sensitive to apparent-volume versus F-corrected-volume conventions. The F-corrected observed Vss target is 421 L; comparing against Vss/F = 702 L gives a much smaller discrepancy.
- The sVEGFR-2 block is a mechanistic placeholder based on assumed soluble receptor turnover, not a refit to individual biomarker observations.
