# Belzutifan PK/PD/Safety Simulation - RCC
## Test Case: Choueiri et al. (2021) and Marathe et al. (2023)

### Overview

This test case implements a belzutifan PK/PD/safety validation workflow for renal
cell carcinoma using aggregate phase 1 and population PK literature anchors.

Model blocks:

- PK: 2-compartment oral model with first-order absorption and lag time.
- Covariates: weight, age, formulation, fed state, UGT2B17 phenotype, and
  CYP2C19 phenotype.
- PD: direct inhibitory Emax/Hill model linking belzutifan exposure to
  erythropoietin suppression.
- Safety: exploratory hemoglobin turnover model calibrated to phase 1 grade 3
  anemia incidence.
- Validation: aggregate comparison against published PK exposure, digitized EPO
  suppression, and anemia incidence.

### References

Marathe P, et al. Population pharmacokinetics and exposure-response analyses of
belzutifan. PMCID: PMC10583240.

Choueiri TK, et al. Inhibition of hypoxia-inducible factor-2alpha in renal cell
carcinoma with belzutifan: a phase 1 trial. *Nature Medicine.* 2021. PMCID:
PMC9128828.

### Source Split

- PK source: PMC10583240 Tables 2, 3, and 5.
- PD biomarker source: PMC9128828 Extended Data Fig. 4, represented by
  `data/digitized_epo_fig4.csv`.
- Safety source: PMC9128828 adverse event table, using grade 3 anemia incidence
  as the clinical calibration anchor.

### Folder Structure

```text
belzutifan/
  choueiri_marathe_2021_2023/
    belzutifan_pkpd.py
    requirements.txt
    README.md
    data/
      digitized_epo_fig4.csv
    results/
      validation_table.csv
      epo_emax_fit.csv
      anemia_summary.csv
      clinical_scenarios.csv
      population_pk_percentiles.csv
      population_epo_percentiles.csv
      figures/
        pk_concentration_percentiles.png
        epo_emax_validation.png
        epo_suppression_percentiles.png
        hemoglobin_response.png
        dose_response.png
```

Long per-subject concentration and PK/PD profile tables are intentionally not
tracked; they are regenerated in memory by the script and summarized as
percentile outputs to keep the test case lightweight.

### How To Run

From repository root:

```bash
python test_cases/RCC/belzutifan/choueiri_marathe_2021_2023/belzutifan_pkpd.py
```

This regenerates CSV outputs under `results/` and figures under
`results/figures/`.

### Current Validation Checkpoints

PK validation against Table 5 FFP Study 4 exposure anchors:

| Metric | Paper | Model | Error |
|---|---:|---:|---:|
| AUC0-24h | 16.71 ug*h/mL | 16.56 ug*h/mL | -0.90% |
| Cmax | 1362.54 ng/mL | 1444.54 ng/mL | +6.02% |
| Cmin | 306.66 ng/mL | 293.49 ng/mL | -4.29% |
| Tmax | 1.42 h | 1.50 h | +5.63% |

PD and safety validation:

| Layer | Target | Model |
|---|---:|---:|
| EPO Emax | digitized Extended Data Fig. 4 | RMSE 13.38% |
| Emax | fitted | 0.666 |
| EC50 | fitted Cavg driver | 207.8 ng/mL |
| Hill gamma | fitted | 1.243 |
| Grade 3 anemia | 27.0% | 27.0% |

### Notes

- The validation baseline uses Study 4 FFP formulation assumptions because the
  patients received FFP during PK sampling. FMF is retained as a separate
  prediction scenario in `clinical_scenarios.csv`.
- The Table 5 VHL-RCC geometric mean CL/F of 7.25 L/h is used for the aggregate
  validation baseline; the Table 2 fixed-effect CL/F of 5.63 L/h remains
  documented in the generated parameter table.
- The EPO fit uses `cavg_to_time_ng_ml` as the PD driver. AUC-scale EC50 values
  must not be applied directly as instantaneous concentration EC50 values.
- The hemoglobin model is an exploratory safety layer calibrated to aggregate
  grade 3 anemia incidence. Patient-level hemoglobin, rescue EPO/transfusion,
  and censoring data would be required for confirmatory safety modeling.
