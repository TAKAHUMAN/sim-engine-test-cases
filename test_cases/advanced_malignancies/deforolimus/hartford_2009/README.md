# Deforolimus PK/PD Simulation - Advanced Malignancies
## Test Case: Hartford et al. (2009) [DOI: 10.1158/1078-0432.CCR-08-2076]

### Overview

This test case implements a deterministic deforolimus (AP23573; ridaforolimus) PK/PD workflow for the weekly IV phase I study in patients with advanced malignancies.

Model blocks:

- PK: 2-compartment IV infusion model with dose-dependent clearance and volume terms.
- PD: effect-compartment Emax model for 4E-BP1 phosphorylation inhibition.
- Dosing: 30-minute IV infusion, weekly schedule, with 75 mg explored as the MTD anchor.
- TGI: exploratory AUC-response checks using the reported aggregate AUC/tumor-size correlation; no individual tumor data are invented or fitted.

**Not for clinical decision-making.**

### Reference

Hartford CM, Desai AA, Janisch L, Karrison T, Rivera VM, Berk L, Loewy JW, Kindler H, Stadler WM, Knowles HL, Bedrosian C, Ratain MJ. A phase I trial to determine the safety, tolerability, and maximum tolerated dose of deforolimus in patients with advanced malignancies. *Clinical Cancer Research.* 2009;15(4):1428-1434. doi: 10.1158/1078-0432.CCR-08-2076.

### Source Split

- PK source: Hartford et al. 2009 summary NCA table for weekly IV deforolimus.
- PD source: Hartford et al. 2009 median 4E-BP1 phosphorylation inhibition values.
- TGI source: Hartford et al. 2009 aggregate AUC versus tumor-size correlation (`n=32`, `r=-0.43`, `P=0.015`).

### Folder Structure

```text
deforolimus/
  hartford_2009/
    deforolimus_pkpd.py
    requirements.txt
    README.md
    results/
      final_model_parameters_table.csv
      model_qualification_table.csv
      pk_goodness_of_fit.csv
      pd_goodness_of_fit.csv
      tgi_stochastic_variability_summary.csv
      figures/
        clinical_simulation_4panel.png
        PK_goodness_of_fit.png
        PD_goodness_of_fit.png
        TGI_stochastic_r_scan.png
```

### How To Run

From repository root:

```bash
python test_cases/advanced_malignancies/deforolimus/hartford_2009/deforolimus_pkpd.py
```

This regenerates CSV outputs under `results/` and figures under `results/figures/`.

### Current Validation Checkpoints

| Metric | Reference | Model | Error | Status |
|---|---:|---:|---:|---|
| Cmax at 75 mg | 1.195 ug/mL | 0.986 ug/mL | 17.47% | PASS |
| AUC at 75 mg | 12.7 ug*h/mL | 11.732 ug*h/mL | 7.62% | PASS |
| t1/2 at 75 mg | 47.3 h | 48.382 h | 2.29% | PASS |
| Cmax at 50 mg | 0.982 ug/mL | 0.840 ug/mL | 14.48% | PASS |
| AUC at 50 mg | 10.7 ug*h/mL | 9.890 ug*h/mL | 7.57% | PASS |
| Inhibition at 1.5 h | 95.0% | 92.45% | 2.68% | PASS |
| Inhibition at 168 h | 75.0% | 75.20% | 0.27% | PASS |

### Notes

- The model intentionally uses summary-level published values only.
- The TGI block is a correlation-preserving exploratory simulation, not a fitted tumor model.
- The final fixed-Imax PD parameterization is used for biological interpretability.
