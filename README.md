# Simulation Engine Test Cases

Validation test cases for the TAKAHUMAN simulation engine, organized by indication and drug.

## Repository Structure

```
test_cases/
  <INDICATION>/
    <drug_name>/
      <author_year>/
        sunitinib_pkpd.py       # Simulation script
        README.md               # Model description and validation
        results/
          figures/              # Output plots
          *.csv                 # Simulation result tables
```

## Indications

| Indication | Drug | Reference | Status |
|---|---|---|---|
| RCC | Sunitinib | Khosravan 2016 (PMCID: PMC5526090) | Complete |
| RCC | Sunitinib | Diekstra 2017 (PMCID: PMC5613186) | Complete |
| RCC | Axitinib | PK: Rini 2013; PD: Schindler 2017 (PMCID: PMC5488123) | Complete |
| RCC | Everolimus | PK: Pawaskar 2013 (PMCID: PMC3755750); PD: RECORD-1 (no PMCID) | Complete |
| RCC | Pazopanib | Tan 2025 (PMCID: PMC12064635) | Complete |
| RCC | Sorafenib | PK: Jain 2011 (PMCID: PMC3162659); PD: Wilhelm 2008 (PMCID: PMC12261297) | Complete |
| RCC | Lenvatinib | Majid et al. 2024 (PMCID: PMC11179699, RCC covariate) | Complete |
| RCC | Cediranib | PK: Li et al. 2017 (PMCID: PMC5510068); PD: separate biomarker source | Complete |

## Adding a New Test Case

1. Create folder: `test_cases/<INDICATION>/<drug>/<author_year>/`
2. Add simulation script and `README.md`
3. Run simulation, save figures to `results/figures/` and CSVs to `results/`
4. Submit a PR to `main`

## Current Coverage

- RCC (Renal Cell Carcinoma): 8 models
  - Sunitinib PK/PD - Khosravan et al., Clin Pharmacokinet 2016;55:1251-1269 (PMCID: PMC5526090)
  - Sunitinib PK/PD - Diekstra et al. 2017 (PMCID: PMC5613186)
  - Axitinib PK/PD chain - PK source: Rini et al. 2013; PD source: Schindler et al. 2017 (PMCID: PMC5488123)
  - Everolimus PBPK/PD chain - PK source: Pawaskar et al. 2013 (PMCID: PMC3755750); PD source: RECORD-1 linkage (no PMCID found)
  - Pazopanib PK/PD chain - Tan et al. 2025 (real-world mRCC and STS, PMCID: PMC12064635)
  - Sorafenib PK/PD - PK source: Jain et al. 2011 (*Br J Clin Pharmacol*, PMCID: PMC3162659); PD source: Wilhelm et al. 2008 RCC xenograft PD (*Mol Cancer Ther*, PMCID: PMC12261297)
  - Lenvatinib integrated PK/biomarker/TGI - Majid et al. 2024 (CL/F x 0.851 for RCC, PMCID: PMC11179699)
  - Cediranib PK/PD - PK source: Li et al. 2017 population PK (PMCID: PMC5510068); PD source: separate VEGFR-2/sVEGFR-2 biomarker literature
