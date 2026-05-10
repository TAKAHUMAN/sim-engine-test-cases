# Dovitinib PK/PD Simulation - RCC
## Test Case: Escudier et al. (2014) [PMID: 24691021; no PMCID found]

### Overview

This test case implements a dovitinib (TKI258) PK/PD validation workflow for
patients with metastatic renal cell carcinoma from Escudier et al. 2014. The
available source values are aggregate paper summaries, so this is an
aggregate-calibrated validation case rather than a raw individual population
model fit.

Model blocks:

- PK: 2-compartment oral model with first-order absorption.
- Elimination: Michaelis-Menten saturable metabolism.
- Autoinduction: time-varying low-concentration clearance from day 1 to day 15.
- Population scaffold: 65-patient sparse pseudo-dataset with log-normal IIV.
- PD: concentration-driven turnover Emax models for sVEGFR1, sVEGFR2, FGF23,
  and VEGF.

### Reference

Escudier B, Grunwald V, Ravaud A, Ou YC, Castellano D, Lin CC, Gschwend JE,
Harzstark A, Beall S, Pirotta N, Squires M, Shi M, Angevin E. Phase II results
of Dovitinib (TKI258) in patients with metastatic renal cell cancer. *Clinical
Cancer Research.* 2014;20(11):3012-3022. doi:
10.1158/1078-0432.CCR-13-3006. PMID: 24691021.

### Folder Structure

```text
dovitinib/
  escudier_2014/
    dovitinib_pkpd.py
    requirements.txt
    README.md
    data/
      paper_pk_targets.csv
      paper_pd_targets.csv
    pkpd_model/
      fit.py
      model.py
      plots.py
    results/
      paper_pk_comparison.csv
      paper_pd_comparison.csv
      pk_parameter_estimates.csv
      pd_parameter_estimates.csv
      model_equations.md
      figures/
        concentration_profiles_day1_day15.png
        observed_vs_predicted.png
        residuals_vs_predicted.png
        random_effects_distribution.png
        vpc_concentration.png
        individual_predictions.png
        svegfr_response.png
```

### How To Run

From repository root:

```bash
python test_cases/RCC/dovitinib/escudier_2014/dovitinib_pkpd.py
```

This regenerates all CSV outputs under `results/` and all figures under
`results/figures/`.

### Validation Checkpoints

Current aggregate PK calibration:

| Metric | Paper | Model | Error |
|---|---:|---:|---:|
| Day 1 Cmax | 326.3 ng/mL | 315.7 ng/mL | -3.25% |
| Day 1 AUC | 5576.4 h*ng/mL | 5750.0 h*ng/mL | +3.11% |
| Day 1 t1/2 | 24.0 h | 23.6 h | -1.70% |
| Day 15 Cmax | 263.5 ng/mL | 268.7 ng/mL | +1.96% |
| Day 15 AUC | 3933.3 h*ng/mL | 3869.8 h*ng/mL | -1.61% |
| Day 15 t1/2 | 11.0 h | 11.2 h | +2.14% |

Current aggregate PD calibration:

| Marker | Day | Paper | Model |
|---|---:|---:|---:|
| sVEGFR1 | 15 | -17.0% | -17.00% |
| sVEGFR1 | 26 | -26.0% | -26.00% |
| sVEGFR2 | 15 | -24.0% | -24.00% |
| sVEGFR2 | 26 | -17.0% | -17.00% |
| FGF23 | 26 | +90.0% | +90.00% |
| VEGF | 26 | +20.0% | +20.00% |

### Notes

- The paper does not appear to have a PMCID; this case is indexed by PMID and
  DOI.
- Apparent oral parameters are estimated because absolute bioavailability is not
  identifiable from oral aggregate summaries alone.
- The initial clearance values in the source build specification conflict with
  the stated autoinduction direction. This implementation enforces the paper
  signal: lower day 15 exposure and shorter half-life imply higher apparent
  clearance by day 15.
- Day 15 Tmax is lower than the paper median in the mechanistic simulation
  because the model carries over the 5-days-on/2-days-off dosing state while the
  paper provides sparse aggregate occasion summaries.
- IIV, residual error, shrinkage, and covariate effects are simulation
  scaffolding unless raw patient-level concentration, biomarker, and covariate
  data become available.
