# Axitinib PK/PD Simulation - mRCC
## Test Case: Schindler et al. (2017)

### Overview
This test case contains an axitinib literature-style PK/PD simulation chain for metastatic renal cell carcinoma (mRCC):

- PK: oral 2-compartment model with lag time and BID dosing
- Exposure: rolling 24 h AUC
- PD: VEGF, sVEGFR-1, sVEGFR-2, sVEGFR-3 indirect-response models
- Safety marker: diastolic blood pressure (dBP)
- Tumor dynamics: SLD model driven by relative sVEGFR-3

Primary references:
- Rini et al. 2013 (axitinib population PK)
- Schindler et al. 2017 (biomarkers, dBP, tumor-size, OS linkage)

### Folder Structure

```
axitinib/
  schindler_2017/
    axitinib_model.py
    literature_defaults.yaml
    README.md
    results/
      figures/
        axitinib_baseline_pkpd_chain.png
```

### Files
- `axitinib_model.py`: Core model implementation (`simulate_axitinib_pk`, `simulate_axitinib_pkpd_chain`, hazard helper)
- `literature_defaults.yaml`: Default literature parameterization and regimen
- `results/figures/axitinib_baseline_pkpd_chain.png`: Baseline run figure generated from defaults

### How To Run
From your simulation repository root (where `microservice/axitinib/baseline_plots.py` exists):

```bash
python microservice/axitinib/baseline_plots.py axitinib_plots_latest
```

This writes:
- `axitinib_plots_latest/axitinib_baseline_pkpd_chain.png`

### Output Notes
The included baseline figure in this test case was generated from literature defaults and copied into:

- `results/figures/axitinib_baseline_pkpd_chain.png`

### Known Scope
- The OS hazard utility is provided as a model helper and is not yet a full survival simulator with cumulative hazard integration.
